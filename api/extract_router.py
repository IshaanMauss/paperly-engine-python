# File: api/extract_router.py
import asyncio
import base64
import io
import time
import hashlib
import json
import os
import re
import concurrent.futures
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from google.genai import types
import fitz

from schemas.ingestion_schema import SlicedQuestionsResponse
from services.extraction_cost import reset_cost_ledger, start_cost_ledger, summarize_cost_ledger
from services.gemini_pdf_service import extract_pdf_native_gemini, rescue_missing_qp_questions, _build_local_qp_page_hints
from services.gemini_runtime import run_gemini_async
from services.gemini_slicer import _get_client
from services.groq_slicer import slice_and_format_questions
from services.pipeline_errors import PipelineServiceError, build_error_detail
from services.pdf_processor import crop_and_compress_diagram_async, pdf_base64_to_vision_pages_async
from services.pix2text_ocr import extract_latex_from_image

router = APIRouter(prefix="/extract", tags=["extract"])
BATCH_SIZE = 10

# Cache for storing extraction results
# Using a simple TTL cache with a 4-hour expiration
EXTRACTION_CACHE = {}
CACHE_TTL = 14400  # 4 hours in seconds
PERSISTENT_CACHE_DIR = Path(
    os.getenv("EXTRACTION_CACHE_DIR", ".extraction_cache")
)
PERSISTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = os.getenv("EXTRACTION_CACHE_VERSION", "v3_math_text_guard")

# Queue system for managing concurrent requests.
# Keep general request concurrency modest, but gate full-PDF extraction more
# tightly. A QP can render 20 pages and schedule many Gemini tasks; letting five
# such documents start at once can spike memory and provider-side 503s even
# though Gemini calls are rate-limited deeper in the stack.
MAX_CONCURRENT_REQUESTS = int(os.getenv("PAPERLY_MAX_CONCURRENT_REQUESTS", "5"))
MAX_CONCURRENT_PDF_EXTRACTIONS = int(os.getenv("PAPERLY_MAX_CONCURRENT_PDF_EXTRACTIONS", "1"))
REQUEST_SEMAPHORE = asyncio.Semaphore(max(1, MAX_CONCURRENT_REQUESTS))
PDF_EXTRACTION_SEMAPHORE = asyncio.Semaphore(max(1, MAX_CONCURRENT_PDF_EXTRACTIONS))

# Thread pool for processing requests
THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

# Job status tracking
JOBS_STATUS: Dict[str, Dict[str, Any]] = {}

# Schema matches Node.js JSON.stringify({ image, mime_type, document_type, file_name, board, page1_image })
class ExtractRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Base64-encoded image payload")
    mime_type: str = Field(default="image/png", description="Incoming payload mime type")
    document_type: str = Field(default="Question Paper", description="Question Paper or Marking Scheme")
    file_name: Optional[str] = Field(default="", description="Original filename for paper_reference_key generation")
    board: str = Field(default="IGCSE", description="Education board (IGCSE or IB)")
    page1_image: Optional[str] = Field(default=None, description="Base64-encoded first page image (for IB extraction)")
    use_cache: bool = Field(default=False, description="Set to true to use cached results")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional trusted metadata from Node")


class RescueMissingRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Base64-encoded source PDF")
    mime_type: str = Field(default="application/pdf", description="Must be application/pdf")
    document_type: str = Field(default="Question Paper", description="Targeted rescue is for QP rows")
    file_name: Optional[str] = Field(default="", description="Original filename")
    board: str = Field(default="IGCSE", description="Education board")
    missing_ids: List[str] = Field(default_factory=list, description="Canonical IDs missing in this upload")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Trusted metadata and expected IDs")


class RepairRowRequest(BaseModel):
    row: Dict[str, Any] = Field(default_factory=dict, description="Current review row to repair")
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Current review rows for sibling context")
    row_index: Optional[int] = Field(default=None, description="Current row index in the review list")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Current paper metadata")
    image: Optional[str] = Field(default=None, description="Optional original PDF payload for PDF-aware repair")
    mime_type: str = Field(default="application/pdf", description="Payload mime type when image is supplied")
    file_name: Optional[str] = Field(default="", description="Original filename for repair context")
    board: str = Field(default="IGCSE", description="Education board")
    repair_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional QA/reviewer context for why repair was requested")


def _hash_image(image_data: str) -> str:
    """Create a hash of the image data for caching"""
    return hashlib.sha256(str(image_data or "").encode()).hexdigest()


def _cache_path(cache_key: str) -> Path:
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", cache_key)
    return PERSISTENT_CACHE_DIR / f"{safe_key}.json"


def _response_to_cache_payload(result: SlicedQuestionsResponse) -> dict:
    return {
        "timestamp": time.time(),
        "result": result.model_dump(mode="json"),
    }


def _read_persistent_cache(cache_key: str) -> Optional[SlicedQuestionsResponse]:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = float(payload.get("timestamp") or 0)
        if time.time() - timestamp >= CACHE_TTL:
            try:
                path.unlink()
            except Exception:
                pass
            return None
        return SlicedQuestionsResponse.model_validate(payload.get("result") or {})
    except Exception as exc:
        print(f"⚠️ [Persistent Cache] Ignoring corrupt cache file {path.name}: {exc}")
        return None


def _write_persistent_cache(cache_key: str, result: SlicedQuestionsResponse) -> None:
    try:
        path = _cache_path(cache_key)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(_response_to_cache_payload(result), ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception as exc:
        print(f"⚠️ [Persistent Cache] Failed to write cache entry: {exc}")


def _response_log_summary(result: SlicedQuestionsResponse) -> dict:
    questions = list(result.questions_array or [])
    diagram_count = 0
    diagram_bytes_estimate = 0
    for question in questions:
        urls = getattr(question, "diagram_urls", []) or []
        if not isinstance(urls, list):
            continue
        for url in urls:
            if isinstance(url, str) and url:
                diagram_count += 1
                diagram_bytes_estimate += len(url)

    metadata = result.metadata.model_dump(mode="json") if result.metadata else {}
    return {
        "question_count": len(questions),
        "diagram_count": diagram_count,
        "diagram_payload_chars": diagram_bytes_estimate,
        "validation_status": metadata.get("validation_status"),
        "paper_reference_key": metadata.get("paper_reference_key"),
        "unified_paper_key": metadata.get("unified_paper_key"),
    }


def _compact_cost_summary() -> dict:
    summary = summarize_cost_ledger()
    return {
        "gemini_calls": summary.get("gemini_calls", 0),
        "gemini_failures": summary.get("gemini_failures", 0),
        "estimated_inr": summary.get("estimated_inr", 0),
        "by_model": summary.get("by_model", {}),
        "sample_failures": summary.get("failures", []),
    }


def _canonical_parts(value: Any) -> List[str]:
    return [part for part in str(value or "").strip().lower().split(".") if part]


def _display_label_from_canonical(canonical_id: str) -> str:
    parts = _canonical_parts(canonical_id)
    if not parts:
        return ""
    return f"{parts[0]}{''.join(f'({part})' for part in parts[1:])}"


def _canonical_parent(canonical_id: str) -> str:
    parts = _canonical_parts(canonical_id)
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _row_canonical(row: Dict[str, Any]) -> str:
    return str(row.get("canonical_question_id") or row.get("question_id") or "").strip().lower()


def _strip_leading_label(text: str) -> str:
    return re.sub(
        r"^\s*\d{1,2}\s*(?:\(\s*(?:[a-z]|[ivxlcdm]+)\s*\)\s*){0,5}",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    ).strip()


def _repair_row_from_sibling_context(
    row: Dict[str, Any],
    rows: List[Dict[str, Any]],
    row_index: Optional[int] = None,
) -> Dict[str, Any]:
    canonical = _row_canonical(row)
    parts = _canonical_parts(canonical)
    if len(parts) < 2:
        return {"applied": False, "reason": "Row has no usable nested canonical ID.", "proposal": row}

    label = _display_label_from_canonical(canonical)
    parent = _canonical_parent(canonical)
    target_token = parts[-1]
    text = str(row.get("question_latex") or "")
    lower = " ".join(text.lower().split())
    current_tail = " ".join(_strip_leading_label(text).split())
    short_math_tail = bool(re.fullmatch(r"[A-Za-z]{1,4}|\\?vec\{?[A-Za-z]{1,4}\}?", current_tail.strip()))
    corrupt_text = (
        "this is a blank page" in lower
        or lower.strip() == "blank page"
        or (len(current_tail) < 12 and not short_math_tail)
    )

    siblings = [
        candidate for candidate in rows
        if isinstance(candidate, dict)
        and _row_canonical(candidate) != canonical
        and _canonical_parent(_row_canonical(candidate)) == parent
    ]
    context_candidates = siblings + [
        candidate for candidate in rows
        if isinstance(candidate, dict)
        and _row_canonical(candidate) in {parent, _canonical_parent(parent)}
    ]
    context_texts = [
        str(candidate.get("question_latex") or "").strip()
        for candidate in context_candidates
        if str(candidate.get("question_latex") or "").strip()
    ]
    if text.strip():
        context_texts.append(text.strip())

    marker_pattern = r"\(\s*(?:[a-z]|[ivxlcdm]+)\s*\)"
    best_shared = ""
    best_target = ""
    for source_text in context_texts:
        cleaned = _strip_leading_label(source_text)
        if not cleaned:
            continue
        target_match = re.search(
            rf"\(\s*{re.escape(target_token)}\s*\)\s*(.+?)(?=(?:\s+{marker_pattern}\s+)|$)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        first_marker = re.search(marker_pattern, cleaned, flags=re.IGNORECASE)
        shared = cleaned[: first_marker.start()].strip() if first_marker else cleaned.strip()
        if len(shared) > len(best_shared):
            best_shared = shared
        if target_match:
            target = " ".join(target_match.group(1).split())
            if len(target) > len(best_target):
                best_target = target

    if not best_target and not corrupt_text:
        if len(current_tail) >= 2:
            best_target = current_tail

    if corrupt_text and not best_target:
        return {
            "applied": False,
            "reason": "Shared stem found, but the actual target subpart text is missing. Repair manually from the PDF.",
            "proposal": row,
        }

    if not best_target and not corrupt_text:
        return {"applied": False, "reason": "No safer sibling repair found.", "proposal": row}

    repaired_body = " ".join(
        part for part in [best_shared, f"({target_token}) {best_target}".strip()] if part
    ).strip()
    if not repaired_body:
        return {"applied": False, "reason": "Could not reconstruct row text.", "proposal": row}

    warnings = row.get("validation_warnings")
    if not isinstance(warnings, list):
        warnings = []
    proposal = {
        **row,
        "question_id": label,
        "canonical_question_id": canonical,
        "parent_canonical_id": parts[0],
        "question_latex": f"{label} {repaired_body}".strip(),
        "needs_review": True,
        "validation_warnings": list(dict.fromkeys([
            *warnings,
            "Row repair proposed text from sibling shared-stem context; verify against PDF before saving.",
        ])),
    }
    return {
        "applied": True,
        "reason": "Rebuilt row from sibling shared-stem context.",
        "proposal": proposal,
        "source": "sibling_shared_stem",
        "row_index": row_index,
    }


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def _repair_candidate_pages(pdf_base64: str, target_id: str, rows: List[Dict[str, Any]]) -> List[int]:
    max_pages = max(1, int(os.getenv("PAPERLY_ROW_REPAIR_MAX_PAGES", "3")))
    target_parts = _canonical_parts(target_id)
    if not target_parts:
        return []

    root = target_parts[0]
    expected_ids = [target_id]
    for row in rows or []:
        row_id = _row_canonical(row)
        if row_id.startswith(f"{root}.") and row_id not in expected_ids:
            expected_ids.append(row_id)

    selected: List[int] = []
    try:
        hints = _build_local_qp_page_hints(pdf_base64, expected_ids=expected_ids)
        for hint in hints:
            page_index = hint.get("page_index")
            if not isinstance(page_index, int):
                continue
            hint_expected = {str(value).strip().lower() for value in (hint.get("expected_ids") or [])}
            likely_root = str(hint.get("likely_root") or "").strip().lower()
            if target_id in hint_expected or likely_root == root:
                selected.append(page_index)
    except Exception as exc:
        print(f"[RowRepair][PDF] Page hint selection failed: {exc}")

    if not selected:
        for idx, row in enumerate(rows or []):
            if _row_canonical(row) == target_id:
                selected.append(max(0, idx // 3))
                break

    deduped: List[int] = []
    for page_index in selected:
        if page_index not in deduped:
            deduped.append(page_index)
    text_matched = [
        page_index for page_index in deduped
        if _pdf_page_text_matches_target(pdf_base64, page_index, target_id)
    ]
    if not text_matched:
        try:
            pdf_bytes = base64.b64decode(str(pdf_base64 or "").split(",", 1)[-1])
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                for page_index in range(doc.page_count):
                    if _pdf_page_text_matches_target(pdf_base64, page_index, target_id):
                        text_matched.append(page_index)
            finally:
                doc.close()
        except Exception as exc:
            print(f"[RowRepair][PDF] full page scan failed: {exc}")
    return (text_matched or deduped)[:max_pages]


def _pdf_page_text_matches_target(pdf_base64: str, page_index: int, target_id: str) -> bool:
    parts = _canonical_parts(target_id)
    if not parts:
        return False
    try:
        pdf_bytes = base64.b64decode(str(pdf_base64 or "").split(",", 1)[-1])
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page_index < 0 or page_index >= doc.page_count:
                return False
            text = " ".join(doc[page_index].get_text("text").lower().split())
        finally:
            doc.close()
    except Exception:
        return False

    root = parts[0]
    if not re.search(rf"\b{re.escape(root)}\b", text):
        return False
    if len(parts) >= 2:
        first_child = parts[1]
        if not re.search(rf"\(\s*{re.escape(first_child)}\s*\)", text):
            return False
    if len(parts) >= 3:
        leaf = parts[2]
        if not re.search(rf"\(\s*{re.escape(leaf)}\s*\)", text):
            return False
    return True


def _valid_pdf_repair_text(target_id: str, text: str) -> bool:
    body = " ".join(_strip_leading_label(text).split())
    lower = body.lower()
    if not body or "blank page" in lower:
        return False
    if len(body) > 1800:
        return False
    if len(body) >= 6:
        return True
    return bool(re.search(r"[A-Za-z0-9]|\\vec|→|⃗", body))


def _repair_target_guidance(canonical: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    parts = _canonical_parts(canonical)
    parent = _canonical_parent(canonical)
    terminal = parts[-1] if parts else ""
    siblings: List[Dict[str, str]] = []
    for row in rows or []:
        sibling_id = _row_canonical(row)
        sibling_parts = _canonical_parts(sibling_id)
        if not sibling_id or sibling_id == canonical:
            continue
        if _canonical_parent(sibling_id) != parent or not sibling_parts:
            continue
        siblings.append({
            "id": sibling_id,
            "terminal": sibling_parts[-1],
            "label": _display_label_from_canonical(sibling_id),
        })
    return {
        "label": _display_label_from_canonical(canonical),
        "parent_id": parent,
        "parent_label": _display_label_from_canonical(parent) if parent else "",
        "terminal": terminal,
        "terminal_marker": f"({terminal})" if terminal else "",
        "sibling_markers": [
            f"{item['label']} terminal ({item['terminal']})"
            for item in siblings[:8]
        ],
    }


def _repair_sibling_terminal_violation(
    canonical: str,
    text: str,
    rows: List[Dict[str, Any]],
) -> str:
    parts = _canonical_parts(canonical)
    parent = _canonical_parent(canonical)
    if len(parts) < 3 or not parent:
        return ""
    sibling_terms: List[str] = []
    for row in rows or []:
        sibling_id = _row_canonical(row)
        sibling_parts = _canonical_parts(sibling_id)
        if not sibling_id or sibling_id == canonical or len(sibling_parts) < 2:
            continue
        if _canonical_parent(sibling_id) == parent:
            sibling_terms.append(sibling_parts[-1])
    if len(parts[-1]) == 1 and "a" <= parts[-1] <= "z":
        sibling_terms.extend(chr(code) for code in range(ord("a"), ord("f") + 1) if chr(code) != parts[-1])
    if not sibling_terms:
        return ""

    body = _strip_leading_label(text)
    window = body[:420]
    for term in sorted(set(sibling_terms), key=len, reverse=True):
        marker = rf"(?<![A-Za-z0-9])\(\s*{re.escape(term)}\s*\)"
        if re.search(marker, window, flags=re.IGNORECASE):
            return (
                f"Candidate for {canonical} includes sibling terminal ({term}) "
                f"near the start. Extract only the target terminal ({parts[-1]})."
            )
    return ""


def _log_row_repair_result(canonical: str, result: Dict[str, Any]) -> None:
    try:
        proposal = result.get("proposal") if isinstance(result, dict) else {}
        text = ""
        diagrams = 0
        if isinstance(proposal, dict):
            text = str(
                proposal.get("question_latex")
                or proposal.get("final_answer")
                or proposal.get("official_marking_scheme_latex")
                or ""
            )
            diagrams = len(proposal.get("diagram_urls") or []) if isinstance(proposal.get("diagram_urls"), list) else 0
        text_preview = " ".join(text.split())
        if len(text_preview) > 600:
            text_preview = text_preview[:600] + "..."
        print(
            "[RowRepair][Result] "
            f"id={canonical!r} applied={bool(result.get('applied'))} "
            f"source={result.get('source')!r} diagrams={diagrams} "
            f"reason={str(result.get('reason') or '')[:220]!r} "
            f"text={text_preview!r}"
        )
    except Exception as exc:
        print(f"[RowRepair][ResultLogError] {exc}")


def _extracted_question_to_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump(mode="json")
    if isinstance(row, dict):
        return dict(row)
    return {}


def _apply_row_update(row: Any, payload: Dict[str, Any]) -> Any:
    if hasattr(row, "model_copy"):
        return row.model_copy(update=payload)
    if isinstance(row, dict):
        updated = dict(row)
        updated.update(payload)
        return updated
    return row


def _compact_math_noise_score(text: str) -> float:
    body = " ".join(str(text or "").split())
    if not body:
        return 0.0
    tokens = body.split()
    if len(tokens) < 12:
        return 0.0
    one_char = sum(1 for token in tokens if re.fullmatch(r"[A-Za-z0-9=+\-(){}.,]", token))
    mathish = sum(1 for token in tokens if re.search(r"[=+\-^]|\\|[(){}]", token))
    return (one_char + 0.5 * mathish) / max(1, len(tokens))


def _row_is_auto_repair_candidate(row: Dict[str, Any]) -> bool:
    doc_type = str(row.get("document_type") or "Question Paper").strip().lower()
    if doc_type == "marking scheme":
        return False
    canonical = _row_canonical(row)
    if len(_canonical_parts(canonical)) < 2:
        return False

    text = str(row.get("question_latex") or "")
    lower = " ".join(text.lower().split())
    warnings = " ".join(str(item) for item in (row.get("validation_warnings") or [])).lower()
    diagram_count = len(row.get("diagram_urls") or []) if isinstance(row.get("diagram_urls"), list) else 0
    needs_review = bool(row.get("needs_review"))

    local_warning = "local qp skeleton created" in warnings
    visual_without_image = diagram_count == 0 and any(
        token in lower
        for token in (
            "diagram shows", "not to scale", "graph", "grid", "table shows",
            "cumulative frequency", "histogram", "venn diagram", "pie chart",
            "triangle", "quadrilateral", "circle", "sector",
        )
    )
    diagram_label_pollution = diagram_count == 0 and any(
        token in lower
        for token in (" scale ", "not to scale", " north ", "angle ", "°")
    ) and len(text) > 80
    math_noise = _compact_math_noise_score(text) >= float(
        os.getenv("PAPERLY_QP_AUTO_REPAIR_MATH_NOISE_SCORE", "0.42")
    )
    repeated_parent_label = bool(re.search(r"\b\d+\([a-z]\).*\b\d+\([a-z]\)", lower))

    return bool(
        local_warning
        or (needs_review and (visual_without_image or diagram_label_pollution or math_noise or repeated_parent_label))
    )


async def _repair_qp_rows_on_page_batch(
    *,
    client: Any,
    model: str,
    page_image_bytes: bytes,
    page_index: int,
    targets: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    file_name: str,
) -> Dict[str, Any]:
    target_lines = []
    for target in targets:
        canonical = str(target.get("canonical") or "")
        row = target.get("row") or {}
        target_lines.append({
            "canonical_question_id": canonical,
            "printed_label": _display_label_from_canonical(canonical),
            "current_text": str(row.get("question_latex") or "")[:900],
            "warnings": row.get("validation_warnings") or [],
            "target_guidance": _repair_target_guidance(canonical, rows),
        })

    prompt = f"""
You are repairing multiple suspicious IGCSE Question Paper rows from ONE PDF page image.
Return ONLY valid JSON.

File: {file_name or "unknown"}
Page: {page_index + 1}
Targets:
{json.dumps(target_lines, ensure_ascii=False)[:9000]}

Return JSON:
{{
  "repairs": [
    {{
      "canonical_question_id": "<one target canonical id>",
      "found": true|false,
      "question_latex": "<clean text beginning with the printed label>",
      "diagram_required": true|false,
      "diagram_region": {{
        "x_start_pct": 0-100,
        "x_end_pct": 0-100,
        "y_start_pct": 0-100,
        "y_end_pct": 0-100
      }} | null,
      "confidence": "high|medium|low",
      "reason": "<short reason>"
    }}
  ]
}}

Rules:
- Repair every target independently. Do not skip a visible target because another target is nearby.
- Use the target printed label and target_guidance. Extract ONLY that target row.
- Include shared parent/stem context when needed to understand the row:
  definitions, formulas, table headings, diagram descriptions, named points, conditions, and values above the leaf.
- Do not include later sibling subparts as part of the target.
- If text in a diagram was polluted into the current row, replace it with the actual printed question sentence.
- If a connected diagram/table/graph is required for the row and not just decorative text, set diagram_required=true and give a tight region around the connected visual plus labels.
- Do not select page numbers, answer lines only, headers, footers, or unrelated neighboring diagrams.
- If no connected image is needed, set diagram_required=false.
- Never invent text. If the target is not visible on this page, found=false.
""".strip()

    try:
        async with REQUEST_SEMAPHORE:
            response = await run_gemini_async(
                lambda: asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=page_image_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            )
        return _parse_json_object(getattr(response, "text", "") or "")
    except Exception as exc:
        print(f"[QPAutoRepair][PageBatch] page={page_index + 1} failed: {exc}")
        return {"repairs": [], "error": str(exc)}


async def _auto_repair_qp_response(
    response: SlicedQuestionsResponse,
    *,
    pdf_base64: str,
    mime_type: str,
    file_name: str,
    board: str,
) -> SlicedQuestionsResponse:
    if str(os.getenv("PAPERLY_QP_AUTO_REPAIR_ENABLED", "false")).strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return response
    if (mime_type or "").lower() != "application/pdf" or not pdf_base64:
        return response
    metadata = response.metadata.model_dump(mode="json") if hasattr(response.metadata, "model_dump") else (response.metadata or {})
    if str(metadata.get("document_type") or "Question Paper").strip().lower() == "marking scheme":
        return response

    rows_raw = list(response.questions_array or [])
    rows = [_extracted_question_to_dict(row) for row in rows_raw]
    candidates: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if _row_is_auto_repair_candidate(row):
            canonical = _row_canonical(row)
            if canonical:
                candidates.append({"index": idx, "canonical": canonical, "row": row})

    if not candidates:
        print("[QPAutoRepair] no risky rows selected")
        return response

    max_rows = max(0, int(os.getenv("PAPERLY_QP_AUTO_REPAIR_MAX_ROWS", "4")))
    max_pages = max(1, int(os.getenv("PAPERLY_QP_AUTO_REPAIR_MAX_PAGES", "2")))
    max_rows_per_page = max(1, int(os.getenv("PAPERLY_QP_AUTO_REPAIR_MAX_ROWS_PER_PAGE", "2")))
    if max_rows <= 0:
        print(f"[QPAutoRepair] candidates={len(candidates)} but max_rows=0")
        return response

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        if sum(len(v) for v in grouped.values()) >= max_rows:
            break
        pages = _repair_candidate_pages(pdf_base64, str(candidate["canonical"]), rows)
        if not pages:
            continue
        page_index = pages[0]
        if page_index not in grouped and len(grouped) >= max_pages:
            continue
        bucket = grouped.setdefault(page_index, [])
        if len(bucket) < max_rows_per_page:
            bucket.append(candidate)

    if not grouped:
        print(f"[QPAutoRepair] selected 0 groups from {len(candidates)} candidates")
        return response

    rendered_pages = await pdf_base64_to_vision_pages_async(
        pdf_base64,
        dpi=int(os.getenv("PAPERLY_QP_AUTO_REPAIR_DPI", "180")),
    )
    client = _get_client()
    model = os.getenv("PAPERLY_QP_AUTO_REPAIR_MODEL", os.getenv("PAPERLY_ROW_REPAIR_MODEL", "gemini-2.5-flash-lite"))
    applied = 0
    attempted = 0

    for page_index, page_targets in sorted(grouped.items()):
        if page_index < 0 or page_index >= len(rendered_pages):
            continue
        attempted += len(page_targets)
        image_bytes = base64.b64decode(rendered_pages[page_index])
        result = await _repair_qp_rows_on_page_batch(
            client=client,
            model=model,
            page_image_bytes=image_bytes,
            page_index=page_index,
            targets=page_targets,
            rows=rows,
            file_name=file_name,
        )
        repairs = result.get("repairs") if isinstance(result, dict) else []
        if not isinstance(repairs, list):
            continue
        by_id = {
            str(item.get("canonical_question_id") or "").strip().lower(): item
            for item in repairs
            if isinstance(item, dict)
        }

        for target in page_targets:
            canonical = str(target["canonical"])
            repair = by_id.get(canonical)
            if not repair or not repair.get("found"):
                continue
            repaired_text = str(repair.get("question_latex") or "").strip()
            label = _display_label_from_canonical(canonical)
            if label and repaired_text and not repaired_text.lower().startswith(label.lower()):
                repaired_text = f"{label} {repaired_text}".strip()
            if not _valid_pdf_repair_text(canonical, repaired_text):
                continue
            sibling_violation = _repair_sibling_terminal_violation(canonical, repaired_text, rows)
            if sibling_violation:
                print(f"[QPAutoRepair][Reject] {canonical}: {sibling_violation}")
                continue

            row_idx = int(target["index"])
            current_row = rows[row_idx]
            diagram_urls = current_row.get("diagram_urls") if isinstance(current_row.get("diagram_urls"), list) else []
            if not diagram_urls and repair.get("diagram_required") is True:
                crop = await _crop_pdf_repair_diagram(
                    pdf_base64,
                    page_index,
                    repair.get("diagram_region"),
                ) if isinstance(repair.get("diagram_region"), dict) else None
                if not crop:
                    crop = await _fallback_pdf_repair_context_crop(pdf_base64, page_index, canonical)
                if crop:
                    diagram_urls = [crop]

            warnings = current_row.get("validation_warnings")
            if not isinstance(warnings, list):
                warnings = []
            next_warnings = list(dict.fromkeys([
                *warnings,
                (
                    f"Auto page-batch repair rebuilt this row from PDF page {page_index + 1}; "
                    "verify before saving."
                ),
            ]))
            if repair.get("diagram_required") is True and not diagram_urls:
                next_warnings.append(
                    "Auto repair detected a likely visual dependency but could not safely crop it; paste diagram manually."
                )

            update = {
                "question_latex": repaired_text,
                "diagram_urls": diagram_urls,
                "needs_review": True,
                "validation_warnings": next_warnings,
            }
            rows_raw[row_idx] = _apply_row_update(rows_raw[row_idx], update)
            rows[row_idx] = {**current_row, **update}
            applied += 1
            print(
                f"[QPAutoRepair][Applied] id={canonical!r} page={page_index + 1} "
                f"diagrams={len(diagram_urls)} text={repaired_text[:220]!r}"
            )

    response.questions_array = rows_raw
    print(
        f"[QPAutoRepair] candidates={len(candidates)} attempted={attempted} "
        f"applied={applied} groups={len(grouped)} model={model}"
    )
    return response


def _safe_percent(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, numeric))


def _normalize_region_percent(region: Dict[str, Any]) -> Dict[str, float]:
    keys = ("x_start_pct", "x_end_pct", "y_start_pct", "y_end_pct")
    raw_values: List[float] = []
    for key in keys:
        try:
            raw_values.append(float(region.get(key)))
        except (TypeError, ValueError):
            pass
    scale = 1.0
    if raw_values:
        max_value = max(raw_values)
        # Gemini sometimes returns 0-1000 page/image coordinates despite the pct schema.
        if 100.0 < max_value <= 1000.0:
            scale = 10.0
        elif 1000.0 < max_value <= 3000.0:
            scale = max_value / 100.0
    return {
        key: _safe_percent((float(region.get(key)) / scale) if scale != 1.0 else region.get(key), -1.0)
        for key in keys
    }


async def _crop_pdf_repair_diagram(
    pdf_base64: str,
    page_index: int,
    region: Any,
) -> Optional[str]:
    if not isinstance(region, dict):
        return None
    try:
        normalized = _normalize_region_percent(region)
        y_start = normalized["y_start_pct"]
        y_end = normalized["y_end_pct"]
        x_start = normalized["x_start_pct"]
        x_end = normalized["x_end_pct"]
        if y_start < 0 or y_end < 0 or y_end <= y_start:
            return None
        if (y_end - y_start) < 3 or (x_end - x_start) < 5:
            return None
        if (y_end - y_start) > 85 and (x_end - x_start) > 95:
            return None
        cropped = await crop_and_compress_diagram_async(
            pdf_base64=pdf_base64,
            page_num=page_index,
            y_start_pct=y_start,
            y_end_pct=y_end,
            x_start_pct=x_start,
            x_end_pct=x_end,
            dpi=int(os.getenv("PAPERLY_ROW_REPAIR_DIAGRAM_DPI", "220")),
            padding_pct=float(os.getenv("PAPERLY_ROW_REPAIR_DIAGRAM_PADDING", "0.02")),
        )
        if cropped and not _image_crop_has_content(cropped):
            return None
        return f"data:image/jpeg;base64,{cropped}" if cropped else None
    except Exception as exc:
        print(f"[RowRepair][PDF] Diagram crop failed page={page_index + 1}: {exc}")
        return None


def _image_crop_has_content(image_b64: str) -> bool:
    try:
        from PIL import Image, ImageStat

        image_bytes = base64.b64decode(str(image_b64 or "").split(",", 1)[-1])
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        width, height = img.size
        if width < 60 or height < 40:
            return False
        stat = ImageStat.Stat(img)
        if not stat.var or stat.var[0] < float(os.getenv("PAPERLY_ROW_REPAIR_MIN_CROP_VARIANCE", "18")):
            return False
        pixels = img.resize((80, 80))
        non_white = sum(1 for value in pixels.getdata() if value < 245)
        return non_white >= int(os.getenv("PAPERLY_ROW_REPAIR_MIN_NONWHITE_PIXELS", "35"))
    except Exception:
        return True


def _repair_context_requests_image(repair_context: Optional[Dict[str, Any]]) -> bool:
    text = json.dumps(repair_context or {}, ensure_ascii=False).lower()
    return any(token in text for token in ("diagram", "graph", "grid", "image", "visual", "picture"))


def _text_mentions_visual(text: str) -> bool:
    lower = (text or "").lower()
    return any(token in lower for token in ("diagram", "graph", "grid", "table", "venn", "chart", "figure", "not to scale"))


def _line_text_from_block(line: Dict[str, Any]) -> str:
    return " ".join(str(span.get("text") or "") for span in line.get("spans") or []).strip()


def _pct_from_rect(rect: Any, page_rect: Any) -> Dict[str, float]:
    return {
        "x_start_pct": max(0.0, min(100.0, 100.0 * rect.x0 / page_rect.width)),
        "x_end_pct": max(0.0, min(100.0, 100.0 * rect.x1 / page_rect.width)),
        "y_start_pct": max(0.0, min(100.0, 100.0 * rect.y0 / page_rect.height)),
        "y_end_pct": max(0.0, min(100.0, 100.0 * rect.y1 / page_rect.height)),
    }


async def _fallback_pdf_repair_context_crop(
    pdf_base64: str,
    page_index: int,
    target_id: str,
) -> Optional[str]:
    """Crop the visible question block around a target label when model crop drifts."""
    parts = _canonical_parts(target_id)
    if len(parts) < 2:
        return None
    try:
        pdf_bytes = base64.b64decode(str(pdf_base64 or "").split(",", 1)[-1])
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page_index < 0 or page_index >= doc.page_count:
                return None
            page = doc[page_index]
            page_rect = page.rect
            text_dict = page.get_text("dict")
            lines: List[Dict[str, Any]] = []
            for block in text_dict.get("blocks") or []:
                for line in block.get("lines") or []:
                    line_text = _line_text_from_block(line)
                    if not line_text:
                        continue
                    bbox = fitz.Rect(line.get("bbox"))
                    lines.append({"text": line_text, "bbox": bbox})
        finally:
            doc.close()
    except Exception as exc:
        print(f"[RowRepair][FallbackCrop] layout read failed page={page_index + 1}: {exc}")
        return None

    if not lines:
        return None

    root = parts[0]
    child = parts[1] if len(parts) >= 2 else ""
    leaf = parts[2] if len(parts) >= 3 else ""
    root_pat = re.compile(rf"^\s*{re.escape(root)}\b")
    child_pat = re.compile(rf"\(\s*{re.escape(child)}\s*\)", re.IGNORECASE) if child else None
    leaf_pat = re.compile(rf"\(\s*{re.escape(leaf)}\s*\)", re.IGNORECASE) if leaf else None

    root_y = None
    child_y = None
    target_y = None
    next_sibling_y = None
    for item in lines:
        text = item["text"]
        y0 = item["bbox"].y0
        if root_y is None and root_pat.search(text):
            root_y = y0
        if child_y is None and child_pat and child_pat.search(text):
            child_y = y0
        if target_y is None and leaf_pat and leaf_pat.search(text):
            target_y = y0
        elif target_y is None and not leaf_pat and child_pat and child_pat.search(text):
            target_y = y0
        elif target_y is not None and next_sibling_y is None:
            if leaf_pat and re.search(r"\(\s*(?:ii|iii|iv|v|b|c|d)\s*\)", text, flags=re.IGNORECASE) and y0 > target_y + 8:
                next_sibling_y = y0
            elif not leaf_pat and re.search(r"\(\s*(?:b|c|d)\s*\)", text, flags=re.IGNORECASE) and y0 > target_y + 8:
                next_sibling_y = y0

    if target_y is None:
        return None

    start_y = root_y if root_y is not None and root_y <= target_y else (child_y if child_y is not None and child_y <= target_y else target_y)
    end_y = next_sibling_y if next_sibling_y is not None else min(page_rect.height, target_y + page_rect.height * 0.16)
    if end_y <= start_y + page_rect.height * 0.05:
        end_y = min(page_rect.height, target_y + page_rect.height * 0.20)

    # Keep a zoomed-out row block, but avoid headers/footers and later unrelated questions.
    y_start = max(0.0, start_y - page_rect.height * 0.035)
    y_end = min(page_rect.height, end_y + page_rect.height * 0.015)
    x_start = page_rect.width * 0.02
    x_end = page_rect.width * 0.98
    region = _pct_from_rect(fitz.Rect(x_start, y_start, x_end, y_end), page_rect)
    cropped = await _crop_pdf_repair_diagram(pdf_base64, page_index, region)
    return cropped


async def _score_pdf_repair_candidate(
    *,
    client: Any,
    model: str,
    page_image_bytes: bytes,
    canonical: str,
    label: str,
    doc_type: str,
    current_text: str,
    repair_context: Optional[Dict[str, Any]],
    target_guidance: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    context_text = json.dumps(repair_context or {}, ensure_ascii=False)[:1400]
    guidance_text = json.dumps(target_guidance or {}, ensure_ascii=False)[:1200]
    candidate_text = json.dumps(candidate or {}, ensure_ascii=False)[:2500]
    prompt = f"""
You are the strict evaluator for a row repair proposal.
Score the proposal against the PDF page image and return ONLY JSON.

Target canonical ID: {canonical}
Target printed label: {label}
Document type: {doc_type}
Target hierarchy guidance:
{guidance_text or "none"}
Current broken extraction:
Omitted intentionally. The saved row may be copied from a different question/page.
QA/reviewer context:
{context_text or "none"}

Candidate proposal:
{candidate_text}

Return JSON:
{{
  "score": 0-10,
  "pass": true|false,
  "critique": "<short concrete feedback for retry>",
  "text_ok": true|false,
  "parent_context_ok": true|false,
  "diagram_ok": true|false
}}

Scoring rules:
- The current broken extraction is NOT ground truth. It may be copied from a different page or sibling. Judge against the PDF image and the target printed label only.
- For deep nested targets, the PDF may show a parent label plus terminal labels. The candidate must contain the requested terminal marker from Target hierarchy guidance and must not contain sibling terminal markers as if they were part of the answer.
- 10 means text is exact, target ID matches, required parent/shared stem is included, no sibling subparts are included, and diagram decision is correct.
- If the target row depends on a diagram/stem above it, parent_context_ok must be true only when candidate question_latex includes enough parent/stem text to understand the row. A diagram_region alone is not enough.
- For example, if the page says "The diagram shows..." with values/conditions above (i), the candidate text must include that stem plus (i), not only the (i) sentence.
- If QA says missing diagram/image, diagram_ok must be true only when diagram_required=true and the region surrounds the connected diagram/stem, not an answer line, page number, header/footer, or unrelated question.
- If the candidate includes later sibling subparts like (ii), (b), or (c) after the target, text_ok=false.
- pass=true only when score >= 8 and text_ok=true and parent_context_ok=true and diagram_ok=true.
""".strip()
    try:
        async with REQUEST_SEMAPHORE:
            response = await run_gemini_async(
                lambda: asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=page_image_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            )
        parsed = _parse_json_object(getattr(response, "text", "") or "")
        parsed["score"] = int(float(parsed.get("score") or 0))
        return parsed
    except Exception as exc:
        print(f"[RowRepair][Judge] failed: {exc}")
        return {
            "score": 0,
            "pass": False,
            "critique": f"Evaluator failed: {exc}",
            "text_ok": False,
            "parent_context_ok": False,
            "diagram_ok": False,
        }


async def _score_pdf_repair_crop(
    *,
    client: Any,
    model: str,
    crop_data_url: str,
    canonical: str,
    label: str,
    repaired_text: str,
    repair_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context_text = json.dumps(repair_context or {}, ensure_ascii=False)[:1200]
    prompt = f"""
You are validating the ACTUAL CROPPED IMAGE that would be pasted onto one repaired row.
Return ONLY JSON.

Target canonical ID: {canonical}
Target printed label: {label}
Repaired row text:
{repaired_text[:1400]}
QA/reviewer context:
{context_text or "none"}

Return JSON:
{{
  "score": 0-10,
  "pass": true|false,
  "critique": "<short concrete reason>",
  "is_connected_visual": true|false,
  "is_wrong_answer_line_or_page_artifact": true|false
}}

Rules:
- pass=true only if the crop contains the diagram/table/graph/visual directly needed by this target row.
- Reject answer lines, page number/header/footer, mark brackets, unrelated later subparts, and neighboring question visuals.
- If repaired text mentions a triangle/graph/table/grid, the crop must show that connected object and its labels/values.
- If the crop only shows text like an answer blank or another question's answer line, score <= 2 and pass=false.
""".strip()
    try:
        image_bytes = base64.b64decode(str(crop_data_url or "").split(",", 1)[-1])
        async with REQUEST_SEMAPHORE:
            response = await run_gemini_async(
                lambda: asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            )
        parsed = _parse_json_object(getattr(response, "text", "") or "")
        parsed["score"] = int(float(parsed.get("score") or 0))
        return parsed
    except Exception as exc:
        print(f"[RowRepair][CropJudge] failed: {exc}")
        return {
            "score": 0,
            "pass": False,
            "critique": f"Crop evaluator failed: {exc}",
            "is_connected_visual": False,
            "is_wrong_answer_line_or_page_artifact": True,
        }


async def _repair_row_from_pdf_context(
    row: Dict[str, Any],
    rows: List[Dict[str, Any]],
    row_index: Optional[int],
    metadata: Optional[Dict[str, Any]],
    pdf_base64: Optional[str],
    mime_type: str,
    file_name: str,
    board: str,
    repair_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    canonical = _row_canonical(row)
    parts = _canonical_parts(canonical)
    if len(parts) < 2:
        return {"applied": False, "reason": "PDF repair needs a nested canonical ID.", "proposal": row}
    if (mime_type or "").lower() != "application/pdf" or not pdf_base64:
        return {"applied": False, "reason": "Original PDF is not available for PDF-aware repair.", "proposal": row}

    pages = _repair_candidate_pages(pdf_base64, canonical, rows)
    if not pages:
        return {"applied": False, "reason": "Could not locate likely PDF page for this row.", "proposal": row}

    rendered_pages = await pdf_base64_to_vision_pages_async(pdf_base64, dpi=int(os.getenv("PAPERLY_ROW_REPAIR_DPI", "180")))
    label = _display_label_from_canonical(canonical)
    sibling_context = []
    parent = _canonical_parent(canonical)
    for candidate in rows or []:
        candidate_id = _row_canonical(candidate)
        if candidate_id == canonical or _canonical_parent(candidate_id) == parent:
            sibling_context.append(f"{candidate_id}: {str(candidate.get('question_latex') or '')[:500]}")

    above_context = []
    below_context = []
    if isinstance(row_index, int):
        for offset in (2, 1):
            idx = row_index - offset
            if 0 <= idx < len(rows):
                item = rows[idx]
                above_context.append(f"{_row_canonical(item)}: {str(item.get('question_latex') or '')[:450]}")
        for offset in (1, 2):
            idx = row_index + offset
            if 0 <= idx < len(rows):
                item = rows[idx]
                below_context.append(f"{_row_canonical(item)}: {str(item.get('question_latex') or '')[:450]}")

    doc_type = str(row.get("document_type") or (metadata or {}).get("document_type") or "Question Paper").strip()
    is_ms = doc_type.lower() == "marking scheme"
    current_text = ""
    existing_diagram_count = len(row.get("diagram_urls") or []) if isinstance(row.get("diagram_urls"), list) else 0
    reviewer_context_text = json.dumps(repair_context or {}, ensure_ascii=False)[:1500]
    target_guidance = _repair_target_guidance(canonical, rows)
    target_guidance_text = json.dumps(target_guidance, ensure_ascii=False)[:1500]

    prompt = f"""
You are repairing ONE IGCSE {doc_type} extraction row from the PDF image.

Target canonical ID: {canonical}
Target printed label: {label}
Target hierarchy guidance:
{target_guidance_text}
File: {file_name or "unknown"}
Current extracted text:
Omitted intentionally because this row may be copied from a different question/page.
Existing diagram/image count on this row: {existing_diagram_count}
Reviewer/QA reason for repair:
{reviewer_context_text or "none"}

Return ONLY valid JSON:
{{
  "found": true|false,
  "question_latex": "<for QP: only the target subpart text, beginning with {label}; for MS: marking answer/steps for this exact row>",
  "diagram_required": true|false,
  "diagram_region": {{
    "x_start_pct": 0-100,
    "x_end_pct": 0-100,
    "y_start_pct": 0-100,
    "y_end_pct": 0-100
  }} | null,
  "confidence": "high|medium|low",
  "reason": "<short reason>"
}}

Rules:
- The current extracted text is not provided because it may be copied from a different question/page. Trust ONLY the PDF image, target printed label, and QA reason.
- Extract ONLY the target subpart {label}; do not include sibling subparts.
- If the target hierarchy says terminal_marker is "(a)", extract the row under that "(a)" marker only. Do not accept sibling markers listed in sibling_markers, such as "(b)", as the target.
- For a deep nested ID like 4.a.iii.a, the PDF may show parent "4(a)(iii)" and then terminal "(a)". Include the parent/shared stem only if needed, then the exact terminal "(a)" text. Exclude terminal "(b)" text.
- If you see later subparts such as (ii), (b), or (c), remove them unless they are shared context before the target.
- Use a zoomed-out view of the whole page: include shared stem, table, graph, diagram instruction, or definitions above/below the target ONLY when required to understand this target.
- If a stem above the target contains values, definitions, named points, diagram description, or setup needed to solve the target, question_latex MUST include that stem as text before the target subpart. Do not rely on image alone for parent context.
- Do not duplicate the same shared stem twice.
- Never copy text from memory or from prior extraction attempts. Read the target from the PDF image.
- Never write "This is a blank page" unless that exact target is truly blank.
- If the target label/text is not visible on this page, return found=false.
- For vector-only rows like (a) AC, returning "{label} AC" is valid.
- If Existing diagram/image count is greater than 0, do not request another image unless the QA reason explicitly says the existing image is wrong.
- If the Reviewer/QA reason says the row mentions a diagram/graph/grid but no image is attached, actively inspect above and below the target label for a connected visual before deciding diagram_required.
- Set diagram_required=true ONLY when a diagram/table/graph/visual is directly connected to this exact target row and is not already just a page number, blank answer space, unrelated neighboring question, or generic page crop.
- If a visual is needed, provide one tight diagram_region around that connected visual, including nearby labels. Otherwise set diagram_required=false and diagram_region=null.
- Do not invent missing content.

Sibling/context rows for orientation:
{chr(10).join(sibling_context[:8])}

Rows immediately above:
{chr(10).join(above_context) or "none"}

Rows immediately below:
{chr(10).join(below_context) or "none"}
""".strip()

    model = os.getenv("PAPERLY_ROW_REPAIR_MODEL", "gemini-2.5-flash-lite")
    client = _get_client()
    min_score = int(os.getenv("PAPERLY_ROW_REPAIR_MIN_SCORE", "9"))
    max_attempts = max(1, int(os.getenv("PAPERLY_ROW_REPAIR_MAX_ATTEMPTS", "3")))
    repair_trace: List[Dict[str, Any]] = []
    for page_index in pages:
        if page_index < 0 or page_index >= len(rendered_pages):
            continue
        image_bytes = base64.b64decode(rendered_pages[page_index])
        feedback = ""
        for attempt_num in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if feedback:
                attempt_prompt += f"""

PREVIOUS ATTEMPT WAS REJECTED BY THE STRICT EVALUATOR.
Fix exactly this:
{feedback[:1200]}
"""
            try:
                async with REQUEST_SEMAPHORE:
                    response = await run_gemini_async(
                        lambda: asyncio.to_thread(
                            client.models.generate_content,
                            model=model,
                            contents=[
                                attempt_prompt,
                                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            ],
                            config=types.GenerateContentConfig(response_mime_type="application/json"),
                        )
                    )
                parsed = _parse_json_object(getattr(response, "text", "") or "")
            except Exception as exc:
                print(f"[RowRepair][PDF] page={page_index + 1} attempt={attempt_num} model={model} failed: {exc}")
                repair_trace.append({"page": page_index + 1, "attempt": attempt_num, "score": 0, "critique": str(exc)})
                continue

            if not parsed.get("found"):
                repair_trace.append({"page": page_index + 1, "attempt": attempt_num, "score": 0, "critique": parsed.get("reason") or "not found"})
                feedback = str(parsed.get("reason") or "Target not found; inspect the whole page for the visible target label.")
                continue
            repaired_text = str(parsed.get("question_latex") or "").strip()
            if label and not repaired_text.lower().startswith(label.lower()):
                repaired_text = f"{label} {repaired_text}".strip()
            if not _valid_pdf_repair_text(canonical, repaired_text):
                repair_trace.append({"page": page_index + 1, "attempt": attempt_num, "score": 0, "critique": "invalid or blank repair text"})
                feedback = "The repaired text was blank/invalid. Read the exact target row and include required parent stem."
                continue
            sibling_violation = _repair_sibling_terminal_violation(canonical, repaired_text, rows)
            if sibling_violation:
                repair_trace.append({
                    "page": page_index + 1,
                    "attempt": attempt_num,
                    "score": 0,
                    "critique": sibling_violation,
                    "candidate_text": repaired_text[:500],
                })
                feedback = sibling_violation
                continue
            parsed["question_latex"] = repaired_text

            evaluation = await _score_pdf_repair_candidate(
                client=client,
                model=os.getenv("PAPERLY_ROW_REPAIR_EVAL_MODEL", model),
                page_image_bytes=image_bytes,
                canonical=canonical,
                label=label,
                doc_type=doc_type,
                current_text=current_text,
                repair_context=repair_context,
                target_guidance=target_guidance,
                candidate=parsed,
            )
            score = int(evaluation.get("score") or 0)
            critique = str(evaluation.get("critique") or "")
            repair_trace.append({
                "page": page_index + 1,
                "attempt": attempt_num,
                "score": score,
                "critique": critique,
                "candidate_text": str(parsed.get("question_latex") or "")[:500],
                "candidate_region": parsed.get("diagram_region") if isinstance(parsed.get("diagram_region"), dict) else None,
            })
            if not evaluation.get("pass") or score < min_score:
                feedback = critique or f"Score {score}/10. Include exact target, required parent stem, and correct connected diagram only."
                continue

            existing_diagrams = row.get("diagram_urls") if isinstance(row.get("diagram_urls"), list) else []
            diagram_urls = existing_diagrams
            cropped_diagram = None
            should_try_image = (
                not existing_diagrams
                and (
                    parsed.get("diagram_required") is True
                    or _repair_context_requests_image(repair_context)
                    or _text_mentions_visual(repaired_text)
                )
            )
            if should_try_image:
                crop_attempts: List[Dict[str, Any]] = []
                cropped_diagram = await _crop_pdf_repair_diagram(
                    pdf_base64,
                    page_index,
                    parsed.get("diagram_region"),
                ) if isinstance(parsed.get("diagram_region"), dict) else None
                if cropped_diagram:
                    crop_attempts.append({"source": "model_region", "data_url": cropped_diagram})
                fallback_crop = await _fallback_pdf_repair_context_crop(pdf_base64, page_index, canonical)
                if fallback_crop:
                    crop_attempts.append({"source": "layout_question_block", "data_url": fallback_crop})

                accepted_crop = None
                accepted_crop_source = None
                last_crop_feedback = ""
                for crop_attempt in crop_attempts:
                    crop_eval = await _score_pdf_repair_crop(
                        client=client,
                        model=os.getenv("PAPERLY_ROW_REPAIR_CROP_EVAL_MODEL", os.getenv("PAPERLY_ROW_REPAIR_EVAL_MODEL", model)),
                        crop_data_url=str(crop_attempt.get("data_url") or ""),
                        canonical=canonical,
                        label=label,
                        repaired_text=repaired_text,
                        repair_context=repair_context,
                    )
                    crop_score = int(crop_eval.get("score") or 0)
                    last_crop_feedback = str(crop_eval.get("critique") or "")
                    repair_trace.append({
                        "page": page_index + 1,
                        "attempt": attempt_num,
                        "crop_source": crop_attempt.get("source"),
                        "crop_score": crop_score,
                        "crop_critique": str(crop_eval.get("critique") or ""),
                    })
                    if crop_eval.get("pass") and crop_score >= int(os.getenv("PAPERLY_ROW_REPAIR_CROP_MIN_SCORE", "8")):
                        accepted_crop = str(crop_attempt.get("data_url") or "")
                        accepted_crop_source = crop_attempt.get("source")
                        break

                if accepted_crop:
                    cropped_diagram = accepted_crop
                    diagram_urls = [accepted_crop]
                    repair_trace.append({
                        "page": page_index + 1,
                        "attempt": attempt_num,
                        "crop_accepted_source": accepted_crop_source,
                    })
                elif parsed.get("diagram_required") is True:
                    cropped_diagram = None
                    feedback = (
                        "No crop was safely accepted for this visual repair. "
                        f"Last crop feedback: {last_crop_feedback or 'wrong/unconnected visual'}. "
                        "Retry with a crop around the connected diagram/stem for the exact target row."
                    )
                    if attempt_num < max_attempts:
                        continue
                    feedback = "Evaluator accepted text but diagram crop failed content checks. Retry with a tighter region around the connected diagram, not answer lines/header/page number."
                    repair_trace.append({
                        "page": page_index + 1,
                        "attempt": attempt_num,
                        "score": score,
                        "critique": "Accepted candidate needed an image, but crop failed safety/content checks.",
                    })
                    return {
                        "applied": False,
                        "reason": "Repair found the right row, but could not safely crop the connected diagram. Paste/crop the diagram manually.",
                        "proposal": row,
                        "source": "pdf_ai_repair",
                        "candidate_pages": [page + 1 for page in pages],
                        "repair_trace": repair_trace,
                    }
                    continue
                else:
                    cropped_diagram = None

            warnings = row.get("validation_warnings")
            if not isinstance(warnings, list):
                warnings = []
            next_warnings = [
                *warnings,
                f"Row repair proposed text from original PDF page; evaluator score {score}/10. Verify before saving.",
            ]
            if parsed.get("diagram_required") and not cropped_diagram and not existing_diagrams:
                next_warnings.append("Row repair saw a likely visual dependency but could not safely crop it; paste/crop diagram manually.")
            proposal = {
                **row,
                "question_id": label,
                "canonical_question_id": canonical,
                "parent_canonical_id": parts[0],
                "diagram_urls": diagram_urls,
                "needs_review": True,
                "validation_warnings": list(dict.fromkeys(next_warnings)),
            }
            if is_ms:
                proposal["question_latex"] = label
                proposal["final_answer"] = _strip_leading_label(repaired_text)
                proposal["official_marking_scheme_latex"] = _strip_leading_label(repaired_text)
            else:
                proposal["question_latex"] = repaired_text
            return {
                "applied": True,
                "reason": f"Rebuilt row from original PDF page {page_index + 1}; evaluator score {score}/10.",
                "proposal": proposal,
                "source": "pdf_ai_repair",
                "row_index": row_index,
                "page_index": page_index,
                "confidence": parsed.get("confidence") or "medium",
                "repair_trace": repair_trace,
            }

    return {
        "applied": False,
        "reason": "PDF-aware repair could not confidently isolate this subpart. Repair manually from the PDF.",
        "proposal": row,
        "source": "pdf_ai_repair",
        "candidate_pages": [page + 1 for page in pages],
        "repair_trace": repair_trace,
    }


async def _process_single_page(page_image: str, page_number: int, document_type: str, file_name: str = "", board: str = "IGCSE"):
    # Check cache first
    image_hash = _hash_image(page_image[:1000])  # Use first 1000 chars for faster hashing
    cache_key = f"{image_hash}_{document_type}_{board}"
    
    current_time = time.time()
    if cache_key in EXTRACTION_CACHE and current_time - EXTRACTION_CACHE[cache_key]["timestamp"] < CACHE_TTL:
        print(f"🔄 [Cache Hit] Using cached result for page {page_number}")
        return EXTRACTION_CACHE[cache_key]["result"]
    
    # Process in parallel with semaphore to limit concurrent API calls
    async with REQUEST_SEMAPHORE:
        # Use ThreadPool for CPU-intensive image processing
        raw_latex = await asyncio.to_thread(extract_latex_from_image, page_image)
        print(f"\n====== 🟢 DEBUG: RAW LATEX FROM GEMINI (PAGE {page_number}) ======")
        print(raw_latex)
        print("===============================================================\n")
    
        # Use ThreadPool for AI processing as well
        page_questions = await asyncio.to_thread(slice_and_format_questions, raw_latex, document_type)
        print(f"\n====== 🔵 DEBUG: SLICED ARRAY FROM GROQ (PAGE {page_number}) ======")
        print(page_questions)
        print("================================================================\n")
        
        # Cache the result
        EXTRACTION_CACHE[cache_key] = {"result": page_questions, "timestamp": current_time}
        
        return page_questions


@router.post("", response_model=SlicedQuestionsResponse)
async def process_image(request: ExtractRequest, background_tasks: BackgroundTasks) -> SlicedQuestionsResponse:
    try:
        # Generate a unique job ID for this request
        job_id = f"job_{int(time.time())}_{hash(request.image[:100])}"
        JOBS_STATUS[job_id] = {"status": "processing", "start_time": time.time(), "progress": 0}
        
        print(f"📥 [API] Processing pipeline starting... type={request.document_type!r} file={request.file_name!r} job={job_id}")
        mime_type = (request.mime_type or "").lower()
        file_name = (request.file_name or "").strip()

        # Check cache first for the full document
        # document_hash = _hash_image(request.image[:2000])  # Old hash logic
        document_hash = _hash_image(request.image)  # Use the new hashing logic
        extra_cache_fingerprint = ""
        if request.extra_metadata:
            extra_cache_fingerprint = hashlib.sha256(
                json.dumps(request.extra_metadata, sort_keys=True, ensure_ascii=True).encode("utf-8")
            ).hexdigest()[:16]
        cache_key = (
            f"{CACHE_VERSION}_{document_hash}_{request.document_type}_{request.board}_"
            f"{mime_type}_{file_name}_{extra_cache_fingerprint}"
        )
        print(
            f"[API] Cache policy: use_cache={request.use_cache} "
            f"cache_key_prefix={cache_key[:32]} file={file_name!r}"
        )
        
        current_time = time.time()
        if request.use_cache and cache_key in EXTRACTION_CACHE and current_time - EXTRACTION_CACHE[cache_key]["timestamp"] < CACHE_TTL:
            print(f"🔄 [Cache Hit] Using cached result for document")
            JOBS_STATUS[job_id]["status"] = "completed"
            return EXTRACTION_CACHE[cache_key]["result"]
        if request.use_cache:
            persistent_hit = _read_persistent_cache(cache_key)
            if persistent_hit is not None:
                print(f"Persistent cache hit for {file_name or request.document_type}")
                EXTRACTION_CACHE[cache_key] = {"result": persistent_hit, "timestamp": current_time}
                JOBS_STATUS[job_id]["status"] = "completed"
                return persistent_hit

        if mime_type == "application/pdf":
            # Process within a semaphore to limit concurrent API calls
            async with PDF_EXTRACTION_SEMAPHORE:
                cost_token = start_cost_ledger()
                try:
                    # Use a shorter timeout to improve user experience
                    # The extract_pdf_native_gemini is already an async function that uses asyncio.to_thread internally
                    # so we don't need to wrap it again with asyncio.to_thread
                    questions_array = await asyncio.wait_for(
                        extract_pdf_native_gemini(
                            pdf_base64=request.image,
                            document_type=request.document_type,
                            filename=file_name,
                            board=request.board,
                            page1_base64=request.page1_image,
                            extra_metadata=request.extra_metadata or None,
                        ),
                        timeout=float(os.getenv("PAPERLY_PDF_EXTRACTION_TIMEOUT_SECONDS", "600")),
                    )
                    if (request.document_type or "").strip().lower() == "question paper":
                        try:
                            questions_array = await _auto_repair_qp_response(
                                questions_array,
                                pdf_base64=request.image,
                                mime_type=mime_type,
                                file_name=file_name,
                                board=request.board,
                            )
                        except Exception as auto_repair_exc:
                            print(f"[QPAutoRepair] skipped after error: {auto_repair_exc}")
                    # Cache the result
                    EXTRACTION_CACHE[cache_key] = {"result": questions_array, "timestamp": current_time}
                    background_tasks.add_task(_write_persistent_cache, cache_key, questions_array)

                    response_summary = _response_log_summary(questions_array)
                    response_summary["gemini_cost"] = _compact_cost_summary()
                    print(f"[API] PDF extraction response summary: {response_summary}")

                    JOBS_STATUS[job_id]["status"] = "completed"
                    return questions_array
                except Exception:
                    print(f"[API] PDF extraction cost summary before failure: {_compact_cost_summary()}")
                    raise
                finally:
                    reset_cost_ledger(cost_token)

        # Process page images in parallel with controlled concurrency
        page_images = [request.image]
        questions_array = []
        extracted_metadata = {}
        
        # Process batches in parallel with controlled concurrency
        for batch_start in range(0, len(page_images), BATCH_SIZE):
            batch = page_images[batch_start : batch_start + BATCH_SIZE]
            tasks = [
                _process_single_page(page_image, batch_start + index + 1, request.document_type, file_name, request.board)
                for index, page_image in enumerate(batch)
            ]
            # Execute tasks with controlled concurrency
            batch_results = await asyncio.gather(*tasks)
            for page_response in batch_results:
                if page_response.metadata:
                    extracted_metadata = page_response.metadata
                questions_array.extend(page_response.questions_array)
            
            JOBS_STATUS[job_id]["progress"] = min(100, int((batch_start + len(batch)) / len(page_images) * 100))

        result = SlicedQuestionsResponse(metadata=extracted_metadata, questions_array=questions_array)
        
        # Cache the result
        EXTRACTION_CACHE[cache_key] = {"result": result, "timestamp": current_time}
        JOBS_STATUS[job_id]["status"] = "completed"
        
        return result
    except asyncio.TimeoutError as exc:
        if job_id in JOBS_STATUS:
            JOBS_STATUS[job_id]["status"] = "failed"
            JOBS_STATUS[job_id]["error"] = "timeout"
            
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "type": "timeout_error",
                    "stage": "pdf_native_gemini",
                    "message": "PDF processing timed out before Gemini returned a response. Try splitting the document into smaller parts.",
                }
            },
        ) from exc
    except PipelineServiceError as exc:
        if job_id in JOBS_STATUS:
            JOBS_STATUS[job_id]["status"] = "failed"
            JOBS_STATUS[job_id]["error"] = exc.message
            
        print(f"❌ [Pipeline Error:{exc.stage}] {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=build_error_detail(exc)) from exc
    except HTTPException:
        if job_id in JOBS_STATUS:
            JOBS_STATUS[job_id]["status"] = "failed"
        raise
    except Exception as e:
        if job_id in JOBS_STATUS:
            JOBS_STATUS[job_id]["status"] = "failed"
            JOBS_STATUS[job_id]["error"] = str(e)
            
        print(f"❌ [Critical API Error]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_error",
                    "stage": "api",
                    "message": "Unexpected error while processing extraction request.",
                    "details": {"reason": str(e)},
                }
            },
        ) from e


@router.post("/rescue-missing")
async def rescue_missing_questions(request: RescueMissingRequest) -> Dict[str, Any]:
    mime_type = (request.mime_type or "").lower()
    if mime_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Targeted rescue requires the original PDF payload.")
    if (request.document_type or "").strip().lower() == "marking scheme":
        raise HTTPException(status_code=400, detail="Targeted rescue is only for Question Paper missing rows.")

    async with PDF_EXTRACTION_SEMAPHORE:
        cost_token = start_cost_ledger()
        try:
            result = await asyncio.wait_for(
                rescue_missing_qp_questions(
                    pdf_base64=request.image,
                    missing_ids=request.missing_ids or [],
                    filename=request.file_name or "",
                    board=request.board or "IGCSE",
                    extra_metadata=request.extra_metadata or None,
                ),
                timeout=float(os.getenv("PAPERLY_RESCUE_TIMEOUT_SECONDS", "180")),
            )
            result["gemini_cost"] = _compact_cost_summary()
            print(
                "[API] Rescue missing response summary: "
                f"{result.get('rescue_report')} cost={result.get('gemini_cost')}"
            )
            return result
        except PipelineServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=build_error_detail(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="Targeted rescue timed out. Try full redo only if the missing rows are important.",
            ) from exc
        except Exception as exc:
            print(f"[API] Rescue missing cost summary before failure: {_compact_cost_summary()}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            reset_cost_ledger(cost_token)


@router.post("/repair-row")
async def repair_extracted_row(request: RepairRowRequest) -> Dict[str, Any]:
    result: Dict[str, Any]
    if request.image:
        result = await _repair_row_from_pdf_context(
            row=request.row or {},
            rows=request.rows or [],
            row_index=request.row_index,
            metadata=request.metadata or None,
            pdf_base64=request.image,
            mime_type=request.mime_type or "application/pdf",
            file_name=request.file_name or "",
            board=request.board or "IGCSE",
            repair_context=request.repair_context or None,
        )
        if not result.get("applied"):
            sibling_result = _repair_row_from_sibling_context(
                row=request.row or {},
                rows=request.rows or [],
                row_index=request.row_index,
            )
            if sibling_result.get("applied"):
                result = sibling_result
    else:
        result = _repair_row_from_sibling_context(
            row=request.row or {},
            rows=request.rows or [],
            row_index=request.row_index,
        )
    _log_row_repair_result(_row_canonical(request.row or {}), result)
    return {
        "success": True,
        "repair": result,
    }


@router.get("/job/{job_id}", response_model=dict)
async def get_job_status(job_id: str):
    """Get the status of a specific extraction job"""
    if job_id not in JOBS_STATUS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up old jobs (more than 1 hour)
    current_time = time.time()
    to_delete = []
    for jid, data in JOBS_STATUS.items():
        if current_time - data.get("start_time", 0) > 3600:  # 1 hour
            to_delete.append(jid)
    
    for jid in to_delete:
        if jid != job_id:  # Don't delete the job we're returning
            JOBS_STATUS.pop(jid, None)
    
    return JOBS_STATUS[job_id]


@router.get("/cache/clear")
async def clear_cache():
    """Administrative endpoint to clear the extraction cache"""
    global EXTRACTION_CACHE
    EXTRACTION_CACHE = {}
    for path in PERSISTENT_CACHE_DIR.glob("*.json"):
        try:
            path.unlink()
        except Exception:
            pass
    return {"message": "Cache cleared", "status": "success"}
