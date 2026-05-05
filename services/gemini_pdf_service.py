import asyncio
import base64
import json
import os
import re
import tempfile
import time

import fitz
from google import genai

from schemas.ingestion_schema import ExtractedQuestion, ExtractedPaperMetadata, SlicedQuestionsResponse
from services.pipeline_errors import PipelineServiceError
from extractors.ref_code_extractor import regex_extract_ref_code
from builders.key_builder import build_paper_reference_key


# ---------------------------------------------------------------------------
# paper_reference_key generator - IGCSE
# ---------------------------------------------------------------------------

def _generate_igcse_paper_reference_key(filename: str) -> str:
    """
    Derive a canonical slug from the IGCSE filename.

    Supported patterns (case-insensitive):
      0607_s18_ms_22   ->  igcse_0607_s18_ms_22
      0580_w21_qp_41   ->  igcse_0580_w21_qp_41
      0580_m22_ms_12   ->  igcse_0580_m22_ms_12

    Season codes: s = May/June, w = Oct/Nov, m = Feb/March
    Falls back to empty string if the filename doesn't match.
    """
    if not filename:
        return ""

    match = re.search(
        r"(\d{4})_([smwSMW])(\d{2})_((?:ms|qp|er|gt))_(\d)(\d)",
        filename,
        re.IGNORECASE,
    )
    if match:
        subject_code, season, year_suffix, doc_type, paper_number, variant = match.groups()
        return f"igcse_{subject_code}_{season.lower()}{year_suffix}_{doc_type.lower()}_{paper_number}{variant}"

    match2 = re.search(r"(\d{4})_([smwSMW])(\d{2})", filename, re.IGNORECASE)
    if match2:
        subject_code, season, year_suffix = match2.groups()
        return f"igcse_{subject_code}_{season.lower()}{year_suffix}"

    return ""


# ---------------------------------------------------------------------------
# paper_reference_key generator - IB
# ---------------------------------------------------------------------------

def _generate_ib_paper_reference_key(subject: str, level: str, paper_number: str,
                                     timezone: str, session: str, year: str,
                                     document_type: str) -> str:
    subject       = str(subject       or "").strip()
    level         = str(level         or "").strip()
    paper_number  = str(paper_number  or "").strip()
    timezone      = str(timezone      or "").strip()
    session       = str(session       or "").strip()
    year          = str(year          or "").strip()
    document_type = str(document_type or "").strip()

    if not all([subject, paper_number, session, year, document_type]):
        return ""

    subject_map = {
        "mathematics: analysis and approaches": "aa",
        "mathematics: applications and interpretation": "ai",
        "mathematics": "math",
        "physics": "phys",
        "chemistry": "chem",
        "biology": "bio",
    }

    subject_code = subject_map.get(subject.lower(), subject.lower())
    level_code   = level.lower() if level else "sl"
    paper_code   = f"p{paper_number}" if paper_number else "p1"
    tz_code      = f"tz{timezone}" if timezone else ""
    session_code = session.lower().replace("/", "").replace(" ", "")
    year_code    = year
    doc_code     = "qp" if document_type.lower() == "question paper" else "ms"

    parts = ["ib", subject_code, level_code, paper_code]
    if tz_code:
        parts.append(tz_code)
    parts.append(f"{session_code}{year_code}")
    parts.append(doc_code)

    return "_".join(parts)


# ---------------------------------------------------------------------------
# IGCSE Page Verification using Regex
# ---------------------------------------------------------------------------

def _verify_igcse_metadata_from_text(text: str, paper_reference_key: str) -> dict:
    """
    Validate IGCSE paper metadata from the first page text against the filename.
    """
    result = {
        "match_status": True,
        "mismatches": [],
        "extracted": {}
    }

    if not text or not paper_reference_key:
        result["match_status"] = False
        result["mismatches"].append("no_text_or_key")
        return result

    if not paper_reference_key.startswith("igcse_"):
        return result

    text = " ".join(text.split())
    key_parts = paper_reference_key.split('_')
    if len(key_parts) < 3:
        result["match_status"] = False
        result["mismatches"].append("invalid_key_format")
        return result

    expected_subject = key_parts[1]

    season_year_match = re.search(r"[smw](\d{2})", paper_reference_key)
    expected_year = f"20{season_year_match.group(1)}" if season_year_match else None

    season_match = re.search(r"_([smw])\d{2}_", paper_reference_key)
    expected_session_code = season_match.group(1) if season_match else None

    session_map = {
        "s": ["may", "june", "summer", "may/june", "june/july"],
        "w": ["october", "november", "winter", "oct", "nov", "oct/nov", "october/november"],
        "m": ["february", "march", "feb", "mar", "feb/mar", "february/march", "march/april", "mar/apr"]
    }
    expected_session_names = session_map.get(expected_session_code, []) if expected_session_code else []

    subject_match = re.search(r"mathematics\s*\(?(\d{4})\)?", text, re.IGNORECASE)
    if subject_match:
        result["extracted"]["subject_code"] = subject_match.group(1)

    year_match = re.search(r"(20\d{2})", text)
    if year_match:
        result["extracted"]["year"] = year_match.group(1)

    session_pattern = r"(may|june|october|november|february|march|winter|summer)"
    session_match_text = re.search(session_pattern, text, re.IGNORECASE)
    if session_match_text:
        result["extracted"]["session"] = session_match_text.group(1).lower()

    if "subject_code" in result["extracted"]:
        if result["extracted"]["subject_code"] != expected_subject:
            result["match_status"] = False
            result["mismatches"].append("subject_code")

    if "year" in result["extracted"] and expected_year:
        if result["extracted"]["year"] != expected_year:
            result["match_status"] = False
            result["mismatches"].append("year")

    if "session" in result["extracted"] and expected_session_names:
        if result["extracted"]["session"] not in expected_session_names:
            result["match_status"] = False
            result["mismatches"].append("session")

    return result


# ---------------------------------------------------------------------------
# IB Extractor using LLM
# ---------------------------------------------------------------------------

def _extract_ib_metadata_from_page(page_base64: str) -> dict:
    try:
        client = _get_client()
        model = _pick_available_model(client)

        pdf_bytes = base64.b64decode(page_base64.split(",", 1)[1] if "," in page_base64 else page_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_file_path = tmp.name

        uploaded_file = client.files.upload(file=temp_file_path)
        _wait_for_file_ready(client, uploaded_file.name, timeout_seconds=240)

        system_prompt = """
You are an IB mathematics document classifier with high precision.
Analyze ONLY the first page of this IB document and extract the following metadata STRICTLY and NOTHING ELSE:
OUTPUT FORMAT — return ONLY a JSON object with these fields (no prose, no explanations):
{
  "subject_name": "<full subject name>",
  "level": "<SL or HL>",
  "paper_number": "<paper number>",
  "timezone": "<timezone number if present>",
  "session": "<month or season>",
  "year": "<4-digit year>",
  "document_type": "<Question Paper or Marking Scheme>"
}
CRITICAL RULES:
1. Extract ONLY what you can see with HIGH CONFIDENCE.
2. Only analyze the FIRST PAGE.
3. Set field to null if you cannot find it with certainty.
4. DO NOT invent or assume any value.
"""
        try:
            response = client.models.generate_content(
                model=model,
                contents=[system_prompt, uploaded_file],
                config={"response_mime_type": "application/json"},
            )
            raw_text = getattr(response, "text", "") or ""
            parsed = _parse_json_payload(raw_text)

            client.files.delete(name=uploaded_file.name)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return parsed
        except Exception as e:
            print(f"⚠️  [IB Metadata Extraction] Failed: {str(e)}")
            return None
    except Exception as e:
        print(f"❌ [IB Metadata Extraction Error] {type(e).__name__}: {e!r}")
        return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_pdf_system_prompt(document_type: str, paper_reference_key: str = "") -> str:
    LATEX_RULES = """
STRICT LATEX ESCAPING (MANDATORY)
- Extract ALL math as raw LaTeX. Use $...$ for inline and $$...$$ for block.
- NEVER use Unicode math characters (², ³, ±, α, ∫). Use LaTeX equivalents (^2, ^3, \\pm, \\alpha, \\int).
- Double-escape every backslash for JSON: write \\\\frac{}{}, NOT \\frac{}{}.
""".strip()

    prk_instruction = (
        f'\n- "paper_reference_key": set to "{paper_reference_key}" in BOTH metadata and every question object.'
        if paper_reference_key else '\n- "paper_reference_key": set to "" if you cannot determine it.'
    )

    # ── Marking-scheme branch ────────────────────────────────────────────────
    if (document_type or "").strip().lower() == "marking scheme":
        return f"""
You are an IGCSE/IB mathematics MARKING SCHEME extraction engine.
OUTPUT FORMAT — return ONLY the following JSON object:
{{
  "metadata": {{
    "curriculum": "<string>", "program": "<string or null>", "subjectCode": "<string>", "tier": "<string or null>",
    "paperNumber": <integer, 0 if unknown>, "session": "<string or null>", "year": <integer, 0 if unknown>, "paper_reference_key": "<string>"
  }},
  "questions_array": [
    {{
      "document_type": "Marking Scheme",
      "curriculum": "<string>", "program": "<string or null>", "subjectCode": "<string>", "tier": "<string or null>",
      "paperNumber": <integer>, "session": "<string or null>", "year": <integer>, "paper_reference_key": "<string>",
      "isTemplatizable": false, "variables": [],
      "question_latex": "<question number as a string>",
      "question_id": "<question number as a string>",
      "final_answer": "<concise final answer>",
      "total_marks": <integer>,
      "method_steps": [ {{ "type": "<mark type>", "description": "<description>" }} ],
      "official_marking_scheme_latex": "<full marking scheme answer in LaTeX>",
      "diagram_urls": [], "needs_review": false
    }}
  ]
}}
CRITICAL RULES:
1. ALWAYS extract ALL mark points from the marking scheme into "method_steps".
2. Question number MUST include the top-level integer (e.g., "3(a)(i)").
{prk_instruction}
{LATEX_RULES}
""".strip()

    # ── Regular question-paper branch ────────────────────────────────────────
    return f"""
You are an IGCSE/IB mathematics question extraction engine.

TARGET
- You MUST read the ENTIRE document, not just the first page. Extract EVERY math question. Analyze the FIRST PAGE to extract metadata.

OUTPUT FORMAT — return ONLY the following JSON object:
{{
  "metadata": {{
    "curriculum": "<string>", "program": "<string or null>", "subjectCode": "<string>", "tier": "<string or null>",
    "paperNumber": <integer, 0 if unknown>, "session": "<string or null>", "year": <integer, 0 if unknown>, "paper_reference_key": "<string>"
  }},
  "questions_array": [
    {{
      "document_type": "Question Paper",
      "curriculum": "<same as metadata>", "program": "<same as metadata>", "subjectCode": "<same as metadata>", "tier": "<same as metadata>",
      "paperNumber": <same as metadata>, "session": "<same as metadata>", "year": <same as metadata>, "paper_reference_key": "<same as metadata>",
      "isTemplatizable": <true | false>, "variables": [],
      "question_latex": "<full question text in LaTeX>",
      "official_marking_scheme_latex": null,
      "diagram_urls": [],
      "diagram_page_number": <integer, the actual page number (1-indexed) where the diagram is located. 0 if none>,
      "diagram_y_range": [<float, top Y percentage 0.0-1.0>, <float, bottom Y percentage 0.0-1.0>],
      "needs_review": false
    }}
  ]
}}
CRITICAL RULES:
1. "diagram_urls": If a question contains a diagram, graph, or illustration, output ["[NEEDS_CROP]"].
2. "diagram_page_number": CRITICAL. If you output [NEEDS_CROP], you MUST provide the exact page number (1, 2, 3...) where the diagram is.
3. "diagram_y_range": CRITICAL. Give the approximate vertical location of the diagram on the page as a ratio from 0.0 (top) to 1.0 (bottom). Example: [0.20, 0.45]. Empty array [] if no diagram.
4. Duplicate ALL metadata fields inside EVERY question object.
{prk_instruction}
{LATEX_RULES}
""".strip()


# ---------------------------------------------------------------------------
# Normalization / defensive mapping
# ---------------------------------------------------------------------------

_QUESTION_FIELD_ALIASES: dict[str, str] = {
    "question_text": "question_latex", "latex": "question_latex", "question_content": "question_latex", "text": "question_latex",
    "marking_scheme_latex": "official_marking_scheme_latex", "answer": "official_marking_scheme_latex", "mark_scheme": "official_marking_scheme_latex",
    "questionNumber": "question_latex", "question_number": "question_latex",
    "diagrams": "diagram_urls", "images": "diagram_urls",
    "templateable": "isTemplatizable", "is_templateable": "isTemplatizable", "is_templatizable": "isTemplatizable",
    "subject_code": "subjectCode", "subject": "subjectCode", "paper": "paperNumber", "paper_number": "paperNumber",
}

_METADATA_FIELD_ALIASES: dict[str, str] = {
    "subject_code": "subjectCode", "subject": "subjectCode", "paper": "paperNumber", "paper_number": "paperNumber",
}

_QUESTION_DEFAULTS: dict = {
    "document_type": "Question Paper", "curriculum": "", "program": None, "subjectCode": "", "tier": None,
    "paperNumber": 0, "session": None, "year": 0, "paper_reference_key": "", "ref_code_base": "", "ref_code_full": "",
    "isTemplatizable": False, "variables": [], "question_latex": "", "question_id": "", "final_answer": "",
    "total_marks": 0, "method_steps": [], "official_marking_scheme_latex": None, "diagram_urls": [], "needs_review": False,
}

_METADATA_DEFAULTS: dict = {
    "curriculum": "", "program": None, "subjectCode": "", "tier": None, "paperNumber": 0, "session": None,
    "year": 0, "paper_reference_key": "", "ref_code_base": "", "ref_code_full": "",
}


def _normalize_tier(tier: str | None) -> str:
    if not tier or not isinstance(tier, str):
        return "N/A"
    tier_lower = tier.lower().strip()
    if "higher" in tier_lower or tier_lower == "hl": return "HL"
    if "standard" in tier_lower or tier_lower == "sl": return "SL"
    if "core" in tier_lower: return "Core"
    if "extended" in tier_lower: return "Extended"
    return "N/A"


def _remap_keys(raw: dict, alias_map: dict[str, str]) -> dict:
    out = {}
    for k, v in raw.items():
        canonical = alias_map.get(k, k)
        if canonical not in out:
            out[canonical] = v
        else:
            out[k] = v
    return out


def _coerce_int(value, default: int = 0) -> int:
    if value is None: return default
    try: return int(value)
    except (ValueError, TypeError): return default


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, int): return bool(value)
    return default


def _coerce_list(value, default=None) -> list:
    if default is None: default = []
    if isinstance(value, list): return value
    if value is None: return default
    return [str(value)]


def _normalize_method_steps(raw_steps) -> list:
    if not isinstance(raw_steps, list): return []
    result = []
    for step in raw_steps:
        if isinstance(step, dict):
            result.append({
                "type": str(step.get("type", "")).strip(),
                "description": str(step.get("description", step.get("desc", ""))).strip(),
            })
        elif isinstance(step, str):
            result.append({"type": "note", "description": step.strip()})
    return result


def _normalize_metadata(raw: dict | None, filename: str, board: str, generated_key_override: str = "") -> dict:
    if not isinstance(raw, dict): raw = {}

    extracted_curr = str(raw.get("curriculum", "")).upper()
    if "INTERNATIONAL BACCALAUREATE" in extracted_curr or "IB" in extracted_curr:
        raw["curriculum"] = "IB"
    elif "CAMBRIDGE" in extracted_curr or "IGCSE" in extracted_curr:
        raw["curriculum"] = "IGCSE"

    if board and ("INTERNATIONAL BACCALAUREATE" in board.upper() or board.upper() == "IB"):
        board = "IB"
    elif board and ("CAMBRIDGE" in board.upper() or board.upper() == "IGCSE"):
        board = "IGCSE"

    raw = _remap_keys(raw, _METADATA_FIELD_ALIASES)
    result = dict(_METADATA_DEFAULTS)
    result.update({k: v for k, v in raw.items() if k in result})
    result["paperNumber"] = _coerce_int(result["paperNumber"], 0)
    result["year"] = _coerce_int(result["year"], 0)
    result["tier"] = _normalize_tier(result.get("tier"))

    generated_key = ""
    if board.upper() == "IGCSE":
        generated_key = _generate_igcse_paper_reference_key(filename)
    else:
        generated_key = generated_key_override or result.get("paper_reference_key", "")

    result["paper_reference_key"] = generated_key or result.get("paper_reference_key", "")
    result["curriculum"] = board.upper()
    return result


def _normalize_question(raw: dict, fallback_metadata: dict, document_type: str) -> dict:
    if not isinstance(raw, dict): return dict(_QUESTION_DEFAULTS)

    raw = _remap_keys(raw, _QUESTION_FIELD_ALIASES)
    result = dict(_QUESTION_DEFAULTS)

    for k in result:
        if k in raw: result[k] = raw[k]

    result["document_type"] = document_type
    result["tier"] = _normalize_tier(result.get("tier"))

    for meta_key in ("curriculum", "program", "subjectCode", "tier", "paperNumber", "session", "year", "ref_code_base", "ref_code_full"):
        if not result.get(meta_key) and fallback_metadata.get(meta_key):
            result[meta_key] = fallback_metadata[meta_key]

    if fallback_metadata.get("curriculum"):
        result["curriculum"] = fallback_metadata["curriculum"]
    if not result.get("paper_reference_key") and fallback_metadata.get("paper_reference_key"):
        result["paper_reference_key"] = fallback_metadata["paper_reference_key"]

    if document_type.strip().lower() == "marking scheme":
        if not result.get("question_id"): result["question_id"] = result.get("question_latex", "")
        if not result.get("final_answer"): result["final_answer"] = ""
        result["total_marks"] = _coerce_int(result.get("total_marks"), 0)
        result["method_steps"] = _normalize_method_steps(result.get("method_steps", []))

    result["paperNumber"] = _coerce_int(result["paperNumber"], 0)
    result["year"] = _coerce_int(result["year"], 0)
    result["isTemplatizable"] = _coerce_bool(result["isTemplatizable"], False)
    result["variables"] = _coerce_list(result["variables"], [])

    raw_diagrams = _coerce_list(result["diagram_urls"], [])
    valid_urls = []
    has_diagram_indicator = False

    flattened_diagrams = []
    for item in raw_diagrams:
        if item is None: continue
        if isinstance(item, list):
            flattened_diagrams.extend([str(subitem).strip() for subitem in item if subitem])
        else:
            flattened_diagrams.append(str(item).strip())

    for item_str in flattened_diagrams:
        if not item_str: continue
        if item_str.startswith("http") or item_str.startswith("data:image") or item_str == "[NEEDS_CROP]":
            valid_urls.append(item_str)
        elif item_str != "[]" and item_str not in ["null", "undefined"]:
            has_diagram_indicator = True

    result["diagram_urls"] = valid_urls
    result["needs_review"] = _coerce_bool(result["needs_review"], False)

    if not result["diagram_urls"]:
        q_latex = (result.get("question_latex") or "").lower()
        if has_diagram_indicator or "diagram" in q_latex or "graph" in q_latex or "figure" in q_latex:
            result["diagram_urls"] = []

    if not isinstance(result["diagram_urls"], list): result["diagram_urls"] = []
    result["curriculum"] = result["curriculum"] or ""
    result["subjectCode"] = result["subjectCode"] or ""
    result["question_latex"] = result["question_latex"] or ""

    return result


def _normalize_response(
    parsed: dict,
    filename: str,
    document_type: str,
    board: str,
    generated_paper_reference_key: str = "",
    extra_metadata: dict = None,
) -> SlicedQuestionsResponse:
    meta_raw = parsed.get("metadata") or {}
    if extra_metadata:
        meta_raw.update(extra_metadata)
    meta_normalized = _normalize_metadata(meta_raw, filename, board, generated_paper_reference_key)

    questions_raw = parsed.get("questions_array") or []
    if not isinstance(questions_raw, list): questions_raw = []

    questions: list[ExtractedQuestion] = []
    for i, q in enumerate(questions_raw):
        try:
            normalized = _normalize_question(q, meta_normalized, document_type)
            schema_fields = set(ExtractedQuestion.model_fields.keys())
            filtered = {k: v for k, v in normalized.items() if k in schema_fields}
            questions.append(ExtractedQuestion(**filtered))
        except Exception as exc:
            print(f"⚠️  [normalize] Skipping question {i} due to validation error: {exc}")
            try:
                safe = dict(_QUESTION_DEFAULTS)
                safe.update({k: v for k, v in meta_normalized.items() if k in safe})
                safe["document_type"] = document_type
                safe["question_latex"] = str(q) if not isinstance(q, dict) else q.get("question_latex", "")
                safe["needs_review"] = True
                schema_fields = set(ExtractedQuestion.model_fields.keys())
                filtered_safe = {k: v for k, v in safe.items() if k in schema_fields}
                questions.append(ExtractedQuestion(**filtered_safe))
            except Exception:
                pass

    return SlicedQuestionsResponse(
        metadata=ExtractedPaperMetadata(**meta_normalized),
        questions_array=questions,
    )


def _parse_json_payload(content: str) -> dict:
    if not content or not content.strip():
        return {"metadata": {}, "questions_array": []}
    try:
        cleaned_content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        cleaned_content = re.sub(r'^```\s*', '', cleaned_content, flags=re.MULTILINE)
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = content[start: end + 1]
            json_str = re.sub(r'[\x00-\x1F]+', ' ', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as final_err:
                print(f"CRITICAL PARSE FAIL: {final_err}")
                return {"metadata": {}, "questions_array": []}
        return {"metadata": {}, "questions_array": []}


# ---------------------------------------------------------------------------
# Gemini client helpers
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    return genai.Client(api_key=api_key)


def _wait_for_file_ready(client: genai.Client, file_name: str, timeout_seconds: int = 240):
    deadline = time.time() + timeout_seconds
    state_code_map = {0: "STATE_UNSPECIFIED", 1: "PROCESSING", 2: "ACTIVE", 3: "FAILED"}

    def _normalize_state(state_value) -> str:
        if state_value is None: return "UNKNOWN"
        name = getattr(state_value, "name", None)
        if isinstance(name, str) and name: return name.upper()
        if isinstance(state_value, int): return state_code_map.get(state_value, str(state_value))
        try: return state_code_map.get(int(state_value), str(state_value)).upper()
        except Exception: return str(state_value).upper() or "UNKNOWN"

    last_state = "UNKNOWN"
    while time.time() < deadline:
        remote_file = client.files.get(name=file_name)
        last_state = _normalize_state(getattr(remote_file, "state", None))
        if "ACTIVE" in last_state: return remote_file
        if "FAILED" in last_state: raise RuntimeError(f"Uploaded file entered FAILED state: {last_state}")
        time.sleep(1.5)
    raise TimeoutError(f"File not ACTIVE before timeout. Last state: {last_state}")


def _pick_available_model(client: genai.Client) -> str:
    # FIX: Updated preferred list with current, valid Gemini model IDs
    preferred = [
        "models/gemini-2.5-flash-preview-05-20",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
    ]
    try:
        available = {m.name for m in client.models.list()}
    except Exception:
        return "gemini-2.0-flash"

    for model_name in preferred:
        if model_name in available:
            return model_name.replace("models/", "")
    for model_name in available:
        if model_name.startswith("models/gemini-"):
            return model_name.replace("models/", "")
    return "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# FIX: Retry helper for transient 503 / rate-limit errors
# ---------------------------------------------------------------------------

def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: dict,
    retries: int = 3,
    delay: float = 5.0,
):
    """Wraps generate_content with simple retry logic for transient server errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_exc = e
            err_str = str(e)
            is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_transient and attempt < retries - 1:
                wait = delay * (attempt + 1)
                print(f"⚠️  [Gemini] Transient error on attempt {attempt + 1}/{retries}, retrying in {wait:.0f}s… ({e})")
                time.sleep(wait)
                continue
            raise
    raise last_exc


# ---------------------------------------------------------------------------
# Core extraction (sync, runs in a thread)
# ---------------------------------------------------------------------------

def _extract_pdf_native_sync(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
) -> SlicedQuestionsResponse:
    if not pdf_base64 or not pdf_base64.strip():
        empty_meta = ExtractedPaperMetadata(**_METADATA_DEFAULTS)
        return SlicedQuestionsResponse(metadata=empty_meta, questions_array=[])

    normalized_b64 = pdf_base64.strip()
    if "," in normalized_b64:
        normalized_b64 = normalized_b64.split(",", 1)[1]

    uploaded_file = None
    temp_file_path = None
    client = None
    extra_metadata = {}
    validation_result = {"match_status": True, "mismatches": []}
    paper_reference_key = ""

    try:
        client = _get_client()
        pdf_bytes = base64.b64decode(normalized_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_file_path = tmp.name

        if board.upper() == "IGCSE":
            paper_reference_key = _generate_igcse_paper_reference_key(filename)
            print(f"ℹ️  [Gemini Native PDF] IGCSE paper_reference_key: {paper_reference_key!r}")
        else:
            ib_metadata = {}
            if page1_base64:
                ib_metadata = _extract_ib_metadata_from_page(page1_base64) or {}
            ref_code, method = regex_extract_ref_code(temp_file_path)
            ref_code_base = ref_code.base if ref_code else ""
            if ref_code:
                session = ib_metadata.get("session", "")
                year = ib_metadata.get("year", "")
                if not session or not year:
                    prefix = ref_code.session_prefix
                    if len(prefix) == 4:
                        year = "20" + prefix[:2]
                        sess_digits = prefix[2:]
                        session = "may" if sess_digits == "25" else "november"
                paper_reference_key = build_paper_reference_key(
                    curriculum="ib", subject=ib_metadata.get("subject_name", ""),
                    tier=ib_metadata.get("level", ""), session=session, year=year,
                    ref_code_base=ref_code_base,
                )
                extra_metadata["ref_code_base"] = ref_code_base
                extra_metadata["ref_code_full"] = ref_code.raw
                print(f"ℹ️  [Gemini Native PDF] IB paper_reference_key: {paper_reference_key!r} via {method}")
            else:
                print("⚠️  [Gemini Native PDF] Could not extract IB reference code via regex")

        needs_review = board.upper() == "IGCSE" and not validation_result["match_status"]
        if needs_review:
            print(f"⚠️  [Gemini Native PDF] Metadata verification failed: {validation_result['mismatches']}")

        uploaded_file = client.files.upload(file=temp_file_path)
        _wait_for_file_ready(client, uploaded_file.name, timeout_seconds=240)

        system_prompt = _build_pdf_system_prompt(document_type, paper_reference_key)

        # FIX: Dynamically pick the primary model instead of hardcoding a deprecated one.
        # FIX: raw_text is now always assigned after the try/except block (not inside except).
        # FIX: Both model calls use _generate_with_retry for 503/429 resilience.
        primary_model = _pick_available_model(client)
        response = None

        try:
            response = _generate_with_retry(
                client, primary_model,
                contents=[system_prompt, uploaded_file],
                config={"response_mime_type": "application/json"},
            )
        except Exception as primary_exc:
            print(f"⚠️  [Gemini Native PDF] Primary model '{primary_model}' failed ({primary_exc}). Trying fallback…")
            fallback_model = _pick_available_model(client)
            try:
                response = _generate_with_retry(
                    client, fallback_model,
                    contents=[system_prompt, uploaded_file],
                    config={"response_mime_type": "application/json"},
                )
            except Exception as fallback_exc:
                raise PipelineServiceError(
                    stage="pdf_native_gemini",
                    message="All models failed.",
                    details={
                        "provider": "gemini",
                        "reason": str(fallback_exc),
                        "exception_type": type(fallback_exc).__name__,
                    },
                ) from fallback_exc

        # FIX: raw_text is assigned here — outside all try/except blocks — so it is
        # always defined whether the primary or fallback model was used.
        raw_text = getattr(response, "text", "") or ""
        parsed_dict = _parse_json_payload(raw_text)

        if needs_review:
            for question in parsed_dict.get("questions_array", []):
                if isinstance(question, dict):
                    question["needs_review"] = True

        # ---------------------------------------------------------------------------
        # Diagram Crop Logic — runs on parsed_dict BEFORE Pydantic models
        # ---------------------------------------------------------------------------
        try:
            questions_list = parsed_dict.get("questions_array", [])
            needs_crop = any(
                (isinstance(q, dict) and (
                    (isinstance(q.get("diagram_urls"), list) and "[NEEDS_CROP]" in q.get("diagram_urls", []))
                    or q.get("diagram_urls") == "[NEEDS_CROP]"
                ))
                for q in questions_list
            )

            if needs_crop:
                crop_bytes = base64.b64decode(normalized_b64)
                doc = fitz.open(stream=crop_bytes, filetype="pdf")
                try:
                    for q in questions_list:
                        if not isinstance(q, dict):
                            continue
                        urls = q.get("diagram_urls", [])
                        if (isinstance(urls, list) and "[NEEDS_CROP]" in urls) or urls == "[NEEDS_CROP]":
                            page_number = max(0, int(q.get("diagram_page_number", 1) or 1) - 1)
                            page_number = min(page_number, len(doc) - 1)
                            page = doc[page_number]

                            y_range = q.get("diagram_y_range") or []
                            if isinstance(y_range, list) and len(y_range) == 2:
                                try:
                                    y0 = float(y_range[0]) * page.rect.height
                                    y1 = float(y_range[1]) * page.rect.height
                                    clip = fitz.Rect(page.rect.x0, y0, page.rect.x1, y1)
                                except Exception:
                                    clip = page.rect
                            else:
                                clip = page.rect

                            pix = page.get_pixmap(clip=clip, colorspace=fitz.csRGB, alpha=False)
                            image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                            q["diagram_urls"] = [f"data:image/png;base64,{image_b64}"]
                finally:
                    doc.close()
        except Exception as e:
            print(f"⚠️  [Gemini Native PDF] Diagram crop failed: {e}")
        # ---------------------------------------------------------------------------

        normalized_response = _normalize_response(
            parsed_dict,
            filename,
            document_type,
            board,
            paper_reference_key,
            extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
        )
        return normalized_response

    except PipelineServiceError:
        raise
    except Exception as exc:
        print(f"❌ [Gemini Native PDF Error] {type(exc).__name__}: {exc!r}")
        raise PipelineServiceError(
            stage="pdf_native_gemini",
            message="Failed to extract structured questions from PDF.",
            details={
                "provider": "gemini",
                "reason": str(exc),
                "exception_type": type(exc).__name__,
            },
        ) from exc
    finally:
        if client is not None and uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public async entry-point
# ---------------------------------------------------------------------------

async def extract_pdf_native_gemini(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
) -> SlicedQuestionsResponse:
    """Extract structured questions from a PDF using Gemini vision models."""
    return await asyncio.to_thread(
        _extract_pdf_native_sync, pdf_base64, document_type, filename, board, page1_base64
    )


__all__ = ["extract_pdf_native_gemini"]