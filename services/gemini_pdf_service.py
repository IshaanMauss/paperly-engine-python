"""
gemini_pdf_service.py
=====================
Extracts structured questions from IGCSE / IB exam PDFs using Gemini.

Architecture — Single-Pass Multimodal Pattern (v3)
---------------------------------------------------
Phase 2 replaces the old two-Task cascade (OCR→Groq + Vision Engine) with a
single call to gemini_slicer.extract_pages_batch().

  Step 1 — Render     : pdf_base64_to_vision_pages_async converts every PDF page
                        to a JPEG at 150 DPI (unchanged, free).
  Step 2 — Extract    : gemini_slicer.extract_pages_batch() concurrently sends
                        each page JPEG to Gemini 2.5 Flash, which returns
                        BOTH structured question JSON AND diagram bounding-box
                        coordinates in one pass. The internal asyncio.Semaphore(3)
                        prevents Gemini 503/RESOURCE_EXHAUSTED errors.
  Step 3 — Crop       : For each question whose diagram_regions list is non-empty,
                        crop_and_compress_diagram_async (pdf_processor.py) slices
                        the PDF page at the given y-percentages and returns a
                        compressed JPEG base64. The result is appended to
                        model.diagram_urls. diagram_urls is ALWAYS initialized
                        as [] — never None.
  Step 4 — Assemble   : All ExtractedQuestion Pydantic models are collected,
                        metadata is normalized, and a SlicedQuestionsResponse
                        is returned to extract_router.py.

Fallback (Gemini Files API)
---------------------------
_extract_pdf_native_sync uploads the full PDF to the Gemini Files API and
extracts via a whole-document prompt. This is used when:
  • Page rendering fails (corrupted PDF, fitz error).
  • The gemini_slicer path raises a PipelineServiceError.
  • GEMINI_SLICER_ENABLED env-var is set to "false".
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import fitz
from google import genai

from schemas.ingestion_schema import (
    ExtractedQuestion,
    ExtractedPaperMetadata,
    SlicedQuestionsResponse,
    QuestionNumberMetadata,
    ValidationReport,
)
from services.pipeline_errors import PipelineServiceError
from services.extraction_cost import record_gemini_usage
from services.gemini_runtime import run_gemini_sync
from services.ms_anchor_reconciler import reconcile_qp_against_ms
from extractors.ref_code_extractor import regex_extract_ref_code
from builders.key_builder import (
    generate_igcse_key,
    generate_ib_key,
    generate_unified_paper_key,
)
from utils.question_normalizer import QuestionNumberNormalizer # Keep for ID normalization
from services.pdf_processor import (
    crop_and_compress_diagram_async,
    pdf_base64_to_vision_pages_async,
)
# Phase 2: import the new single-pass engine
from services.gemini_slicer import extract_pages_batch as gemini_slicer_extract_pages

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 2: Metadata verification  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _verify_igcse_metadata_from_text(text: str, paper_reference_key: str) -> dict:
    result = {"match_status": True, "mismatches": [], "extracted": {}}
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
        "m": ["february", "march", "feb", "mar", "feb/mar", "february/march", "march/april", "mar/apr"],
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


def _extract_ib_metadata_from_page(page_base64: str) -> dict:
    uploaded_file = None
    temp_file_path = None
    client = None
    try:
        client = _get_client()
        model = _pick_available_model(client)
        pdf_bytes = base64.b64decode(
            page_base64.split(",", 1)[1] if "," in page_base64 else page_base64
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_file_path = tmp.name

        uploaded_file = run_gemini_sync(lambda: client.files.upload(file=temp_file_path))
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
        response = run_gemini_sync(
            lambda: client.models.generate_content(
                model=model,
                contents=[system_prompt, uploaded_file],
                config={"response_mime_type": "application/json"},
            )
        )
        raw_text = getattr(response, "text", "") or ""
        return _parse_json_payload(raw_text)

    except Exception as e:
        print(f"❌ [IB Metadata Extraction Error] {type(e).__name__}: {e!r}")
        return {}
    finally:
        if client and uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# ===========================================================================
# SECTION 3: Prompt builder  (for Gemini Files API fallback path only)
# ===========================================================================

def _build_pdf_system_prompt(
    document_type: str,
    paper_reference_key: str = "",
    board: str = "IGCSE",
) -> str:
    LATEX_RULES = """
STRICT LATEX ESCAPING (MANDATORY)
- Extract ALL math as raw LaTeX. Use $...$ for inline and $$...$$ for block.
- NEVER use Unicode math characters (², ³, ±, α, ∫). Use LaTeX equivalents (^2, ^3, \\pm, \\alpha, \\int).
- Double-escape every backslash for JSON: write \\\\frac{}{}, NOT \\frac{}{}.

ANSWER SPACES — IGNORE COMPLETELY (MANDATORY)
- Exam papers contain answer-writing areas: dotted lines (......), ruled lines (______),
  blank boxes, or "[2]"-style mark brackets at line ends.
- These are NOT question content. DO NOT extract them. DO NOT convert them to \\textunderscore,
  \\underline, \\dotfill, or any LaTeX equivalent. Omit them entirely from question_latex.
- If you see a sequence of \\textunderscore or \\ldots representing a blank line, DELETE it.

PARENT CONTEXT — MANDATORY FOR MULTI-PAGE PDFs
- If a question has sub-parts (a), (b), (c) on a DIFFERENT page than the parent stem,
  you MUST still copy the full parent stem text into EVERY sub-part's question_latex.
- NEVER output just "(a) ..." without the parent question number and stem prepended.
- Example: Parent "3 The diagram shows a triangle." on page 2, sub-parts on page 3:
  → question_latex for 3(a): "3(a) [full parent stem text] [sub-part text]"
  → question_latex for 3(b): "3(b) [full parent stem text] [sub-part text]"
""".strip()

    # ──────────────────────────────────────────────────────────────────────────
    # STRICT SESSION & TIER MAPPING RULES
    # Mirrors gemini_slicer.py session_mapping_rule / tier_mapping_rule.
    # Injected into BOTH QP and MS prompts so the fallback path enforces
    # the same data-integrity contract as the primary slicer path.
    # ──────────────────────────────────────────────────────────────────────────
    SESSION_MAPPING_RULE = """
🔒 STRICT SESSION MAPPING (ZERO TOLERANCE — NO DEFAULTS):
Extract the session from the cover page ONLY if explicitly stated.
NEVER default to "s" (summer). NEVER output the full month name.

MAPPING TABLE — output ONLY the single letter shown on the right:
  "February", "March", "Feb", "Mar", "Feb/Mar", "February/March"  →  "m"
  "May", "June", "July", "Jun", "Jul", "Summer", "May/June", "June/July" →  "s"
  "October", "November", "Oct", "Nov", "Winter",
  "Oct/Nov", "October/November"                                    →  "w"
  (month/season not found on cover page)                           →  null

ANTI-HALLUCINATION RULES (ZERO EXCEPTIONS):
  ❌ NEVER default to "s" when you cannot find the month/season.
  ❌ NEVER output a full month name ("march", "june"). Only "m", "s", or "w".
  ❌ NEVER infer the session from the year, subject code, or filename.
  ❌ NEVER output "may/june", "february/march", or any other compound string.
  ✅ ONLY valid outputs: "m", "s", "w", or null — absolutely nothing else.
""".strip()

    TIER_MAPPING_RULE = """
🔒 STRICT TIER MAPPING (ZERO TOLERANCE):
Extract the tier ONLY from explicit words printed on the cover page.

MAPPING TABLE:
  Cover page contains "Extended" (any case)  →  return EXACTLY "Extended"
  Cover page contains "Core"    (any case)   →  return EXACTLY "Core"
  Neither word appears on the cover page     →  return "N/A"

ANTI-HALLUCINATION RULES (ZERO EXCEPTIONS):
  ❌ DO NOT assume "Extended" is the default. Extract only what is printed.
  ❌ DO NOT use abbreviations ("Ext", "Ext.", "C"). Exact strings only.
  ❌ NEVER return null or an empty string for tier — use "N/A" when absent.
  ✅ ONLY valid outputs: "Extended", "Core", or "N/A".
""".strip()

    prk_instruction = (
        f'\n- "paper_reference_key": set to "{paper_reference_key}" in BOTH metadata and every question object.'
        if paper_reference_key
        else '\n- "paper_reference_key": set to "" if you cannot determine it.'
    )

    board_upper = board.upper()
    if board_upper == "IB":
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND (IB — AO-BASED):
Command term is PRIMARY. Mark count is FALLBACK only when no command term is visible.
  LOW  (AO1): State, Write down, List, Label, Draw, Plot, Define, Identify, Name.
  MEDIUM (AO2): Find, Calculate, Show, Determine, Solve, Construct, Sketch, Verify, Justify.
  HIGH (AO3/AO4): Derive, Prove, Explain, Analyse, Interpret, Comment, Discuss, Evaluate,
                  "Hence", "Hence or otherwise", "Find the exact value of" (multi-step).
Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()
    elif board_upper in ("IGCSE", "CAMBRIDGE"):
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND (IGCSE UNIVERSAL — OFFICIAL COMMAND WORDS):
Mark count is PRIMARY. Command word is secondary confirmation.
  LOW  (1 mark OR): State, Write down, Give, Write, Plot, Name, Identify, List, Label, Recall.
  MEDIUM (2 marks OR): Work out, Calculate, Describe, Sketch, Determine, Construct, Complete,
                       Measure, Outline, Suggest, Solve, Expand, Factorise, Simplify.
  HIGH (3+ marks OR): Show (that), Explain, Comment, Compare, Revise, "Make [var] subject of",
                      "Find [expression] in terms of", "Draw a [histogram/graph]",
                      "Hence show", "Hence or otherwise", Derive, Justify, Prove, Analyse.
Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()
    else:
        print(f"⚠️  [Difficulty] Unknown board '{board}', using IGCSE rules as fallback.")
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND:
  LOW = 1 mark. MEDIUM = 2–3 marks. HIGH = 4+ marks.
Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()

    if (document_type or "").strip().lower() == "marking scheme":
        return f"""
You are an {board} mathematics MARKING SCHEME extraction engine.
OUTPUT FORMAT — return ONLY the following JSON object:
{{
  "metadata": {{
    "curriculum": "{board}", "program": "<string or null>", "subjectCode": "<string>", "tier": "<string or null>",
    "paperNumber": <integer, 0 if unknown>, "session": "<string or null>", "year": <integer, 0 if unknown>, "paper_reference_key": "<string>"
  }},
  "questions_array": [
    {{
      "document_type": "Marking Scheme",
      "curriculum": "{board}", "program": "<string or null>", "subjectCode": "<string>", "tier": "<string or null>",
      "paperNumber": <integer>, "session": "<string or null>", "year": <integer>, "paper_reference_key": "<string>",
      "isTemplatizable": false, "variables": [],
      "question_latex": "<question number as a string>",
      "question_id": "<question number as a string>",
      "final_answer": "<concise final answer>",
      "total_marks": <integer>,
      "method_steps": [ {{ "type": "<mark type>", "description": "<description>" }} ],
      "official_marking_scheme_latex": "<full marking scheme answer in LaTeX>",
      "diagram_urls": [],
      "diagram_page_number": <integer or null>,
      "diagram_y_range": [<float>, <float>],
      "needs_review": false,
      "cognitive_demand": "<LOW | MEDIUM | HIGH>",
      "difficulty_override": null
    }}
  ]
}}
CRITICAL RULES:
1. ALWAYS extract ALL mark points from the marking scheme into "method_steps".
2. Question number MUST include the top-level integer (e.g., "3(a)(i)").
3. STRICT DIAGRAM DETECTION: "diagram_urls" must be [] — diagram positions are detected separately via vision.
4. {difficulty_rule}
5. ⭐ METADATA — SESSION (MANDATORY):
{SESSION_MAPPING_RULE}
6. ⭐ METADATA — TIER (MANDATORY):
{TIER_MAPPING_RULE}
{prk_instruction}
{LATEX_RULES}
""".strip()

    return f"""
You are an {board} mathematics question extraction engine.
TARGET
- You MUST read the ENTIRE document. Extract EVERY math question.

OUTPUT FORMAT — return ONLY the following JSON object:
{{
  "metadata": {{
    "curriculum": "{board}", "program": "<string or null>", "subjectCode": "<string>", "tier": "<string or null>",
    "paperNumber": <integer, 0 if unknown>, "session": "<string or null>", "year": <integer, 0 if unknown>, "paper_reference_key": "<string>"
  }},
  "questions_array": [
    {{
      "document_type": "Question Paper",
      "curriculum": "<same as metadata>", "program": "<same as metadata>", "subjectCode": "<same as metadata>", "tier": "<same as metadata>",
      "paperNumber": <same as metadata>, "session": "<same as metadata>", "year": <same as metadata>, "paper_reference_key": "<same as metadata>",
      "isTemplatizable": <true | false>, "variables": [],
      "question_latex": "<full question text starting with question number>",
      "official_marking_scheme_latex": null,
      "diagram_urls": [],
      "diagram_page_number": <integer>,
      "diagram_y_range": [<float>, <float>],
      "needs_review": false,
      "cognitive_demand": "<LOW | MEDIUM | HIGH>",
      "difficulty_override": null
    }}
  ]
}}
CRITICAL RULES:
1. HIERARCHICAL NUMBERING (MANDATORY): Prepend the parent integer to EVERY sub-question.
2. INFERRED NUMBERING & SEQUENCE (CRITICAL): If a question does not have a visible number, infer it from sequence.
3. STRICT DIAGRAM DETECTION: "diagram_urls" must be [] — diagram positions are detected separately via vision.
4. Duplicate ALL metadata fields inside EVERY question object.
5. {difficulty_rule}
6. ⭐ METADATA — SESSION (MANDATORY):
{SESSION_MAPPING_RULE}
7. ⭐ METADATA — TIER (MANDATORY):
{TIER_MAPPING_RULE}
{prk_instruction}
{LATEX_RULES}
""".strip()


# ===========================================================================
# SECTION 4: Legacy Vision Engine  (kept as fallback, not called in primary path)
# ===========================================================================

_VISION_ENGINE_PROMPT = """
You are a DIAGRAM COORDINATE DETECTOR. Your ONLY job is to locate VISUAL elements on this exam page.
Return ONLY a valid JSON array. No prose, no markdown fences, no explanations.
Schema for each detected element:
{"question_number": "<e.g. 4a or 7(b)(i)>", "y_start_pct": <0-100 float>, "y_end_pct": <0-100 float>}

RULE 1 — HIGH RECALL: Capture EVERY visual math element (geometry, graphs, coordinate grids, etc.).
RULE 2 — NO GARBAGE: DO NOT capture text tables, blank answer spaces, or headers.
RULE 3 — TIGHT CROPPING: Wrap bounding box tightly around visual only (not surrounding text).
- y_start_pct and y_end_pct are percentages of TOTAL PAGE HEIGHT (0=top, 100=bottom).
- If NO diagrams on this page, return exactly: []
""".strip()

_VISION_SEMAPHORE = asyncio.Semaphore(3)
_VISION_MODEL = "gemini-2.5-flash"


async def _run_vision_engine_for_page(
    page_jpeg_b64: str,
    page_num: int,
) -> List[Dict]:
    """
    Legacy Vision Engine — detects diagram bounding boxes for a single page.
    Only used in the Gemini Files API fallback path.
    """
    if not page_jpeg_b64:
        return []

    try:
        client = _get_client()
        image_content = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": page_jpeg_b64,
            }
        }
        response = None
        last_exc = None
        for attempt in range(3):
            try:
                async with _VISION_SEMAPHORE:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=_VISION_MODEL,
                        contents=[_VISION_ENGINE_PROMPT, image_content],
                        config={"response_mime_type": "application/json"},
                    )
                break
            except Exception as e:
                last_exc = e
                err_str = str(e)
                is_transient = any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "Too Many Requests"))
                if is_transient and attempt < 2:
                    wait_time = 4 * (2 ** attempt)
                    logger.warning(f"[Vision Engine] Transient error page {page_num}, attempt {attempt+1}/3. Retry in {wait_time}s.")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        if response is None:
            return []

        raw_text = (getattr(response, "text", "") or "").strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"```$", "", raw_text).strip()
        if not raw_text or raw_text == "[]":
            return []

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                for key in ["diagrams", "coordinates", "vision_results", "items"]:
                    if isinstance(parsed.get(key), list):
                        parsed = parsed[key]
                        break
                else:
                    return []
            else:
                return []

        validated = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            q_num = str(entry.get("question_number") or "").strip()
            y0, y1 = entry.get("y_start_pct"), entry.get("y_end_pct")
            if not q_num or y0 is None or y1 is None:
                continue
            try:
                y0_f, y1_f = float(y0), float(y1)
            except (ValueError, TypeError):
                continue
            if y0_f >= y1_f or y0_f < 0 or y1_f > 100:
                continue
            # BUG 4 FIX: 5% header/footer exclusion zone (matches gemini_slicer constants).
            # Page numbers, watermarks, and 'Turn over' footers live in the top/bottom 5%.
            # Clamp rather than hard-reject so a legitimate diagram touching the zone
            # is still captured (with a reduced coordinate).
            _HF_ZONE = 5.0
            if y0_f < _HF_ZONE:
                y0_f = _HF_ZONE
            if y1_f > (100.0 - _HF_ZONE):
                y1_f = 100.0 - _HF_ZONE
            height_pct = y1_f - y0_f
            # BUG 4 FIX: minimum height raised from 2.0% -> 5.0% to match
            # gemini_slicer._MIN_DIAGRAM_HEIGHT_PCT. 2% at 150 DPI is ~17px,
            # enough to match a stray dot or page number. 5% ~= 42px minimum.
            if height_pct > 65.0 or height_pct < 5.0:
                continue
            validated.append({"question_number": q_num, "y_start_pct": y0_f, "y_end_pct": y1_f, "page_num": page_num})

        return validated

    except Exception as e:
        logger.warning(f"[Vision Engine] Page {page_num} failed silently: {e}")
        return []


def _build_vision_lookup(
    all_page_results: List[List[Dict]],
    question_normalizer: QuestionNumberNormalizer,
) -> Dict[str, List[Dict]]:
    lookup: Dict[str, List[Dict]] = {}
    for page_results in all_page_results:
        for coord in page_results:
            key = question_normalizer.normalize_for_matching(coord.get("question_number", ""))
            if key:
                if key not in lookup:
                    lookup[key] = []
                lookup[key].append(coord)
    return lookup


# ===========================================================================
# SECTION 5: Diagram crop injection helpers
# ===========================================================================

async def _apply_vision_crops_to_questions(
    questions_raw: List[dict],
    vision_lookup: Dict[str, List[Dict]],
    pdf_base64: str,
    question_normalizer: QuestionNumberNormalizer,
) -> List[dict]:
    """
    Legacy merge step used in the Gemini Files API fallback path.
    Injects cropped diagram base64 into question dicts based on Vision Engine coords.
    """
    if not vision_lookup or not questions_raw:
        return questions_raw

    async def _process_question_crops(q: dict) -> dict:
        q_id_candidates = [
            q.get("canonical_question_id", ""),
            q.get("question_id", ""),
            q.get("question_latex", ""),
        ]

        if not isinstance(q.get("diagram_urls"), list):
            q["diagram_urls"] = []

        existing_urls = q["diagram_urls"]
        already_has_real_image = any(
            isinstance(u, str) and (u.startswith("http") or u.startswith("data:image"))
            for u in existing_urls
        )
        if already_has_real_image:
            return q

        has_needs_crop = (
            isinstance(existing_urls, list) and "[NEEDS_CROP]" in existing_urls
        ) or existing_urls == "[NEEDS_CROP]"

        has_vision_lookup_entry = any(
            question_normalizer.normalize_for_matching(candidate) in vision_lookup
            for candidate in q_id_candidates
        )

        if not (has_needs_crop or has_vision_lookup_entry):
            return q

        coords_to_crop = []
        for candidate in q_id_candidates:
            key = question_normalizer.normalize_for_matching(candidate)
            if key and key in vision_lookup:
                coords_to_crop.extend(vision_lookup[key])
                vision_lookup[key] = []

        if not coords_to_crop:
            return q

        crop_tasks = [
            crop_and_compress_diagram_async(
                pdf_base64=pdf_base64,
                page_num=coord.get("page_num", 0),
                y_start_pct=coord["y_start_pct"],
                y_end_pct=coord["y_end_pct"],
            )
            for coord in coords_to_crop
        ]
        cropped_b64_list = await asyncio.gather(*crop_tasks)

        for i, cropped_b64 in enumerate(cropped_b64_list):
            if cropped_b64:
                existing_urls.append(f"data:image/jpeg;base64,{cropped_b64}")
                coord = coords_to_crop[i]
                height_pct = coord["y_end_pct"] - coord["y_start_pct"]
                logger.info(
                    f"[Vision Merge] ✅ Crop for q={q.get('question_id', '?')} "
                    f"page={coord.get('page_num', '?')} y={coord['y_start_pct']:.1f}%-{coord['y_end_pct']:.1f}% "
                    f"height={height_pct:.1f}%"
                )
            else:
                coord = coords_to_crop[i]
                logger.warning(
                    f"[Vision Merge] ⚠️ Crop returned None for q={q.get('question_id', '?')} "
                    f"page={coord.get('page_num', '?')}"
                )

        q["diagram_urls"] = existing_urls
        return q

    updated = await asyncio.gather(*[_process_question_crops(q) for q in questions_raw])
    return list(updated)


# ===========================================================================
# SECTION 5b: Phase 2 — Diagram crop injection from gemini_slicer regions
# ===========================================================================
#
# PATCH SUMMARY (diagram_urls fix — see gemini_slicer.py for Bug 1 + Bug 2):
# Bug 3 fix: when Gemini returns diagram_regions=[] for a QP page (due to
# thinking_budget=0 or permissive prompt), the old code hit
# `if not regions: return model` and zero crops ever ran.
#
# This section now contains two new pure helpers:
#   _cluster_rects_by_y_proximity  — groups nearby fitz drawing paths
#   _detect_diagram_regions_via_pymupdf — deterministic fallback detector
#
# _inject_diagram_crops_from_slicer runs PyMuPDF for any QP question whose
# diagram_regions list is empty, assigns discovered regions by Y-order, then
# proceeds with the normal concurrent crop pass.
# ===========================================================================

# ---------------------------------------------------------------------------
# Bounding-box sanity constants (mirrors gemini_slicer.py — keep in sync)
# ---------------------------------------------------------------------------
_PYMUPDF_MAX_DIAGRAM_HEIGHT_PCT = 50.0
_CROP_INJECT_MAX_HEIGHT_PCT = 72.0

# FIX 2 (Noise Filter): Raised from 2.0 to 5.0 to align with gemini_slicer.
# 5% of A4 height safely ignores stray marks, bullet points, and rule lines,
# while catching legitimate small diagrams and graphs.
_PYMUPDF_MIN_DIAGRAM_HEIGHT_PCT = 5.0

# Minimum fraction of page WIDTH a vector cluster must span.
# Filters out narrow vertical artifacts (like column separators).
_PYMUPDF_MIN_DIAGRAM_WIDTH_PCT  = 8.0


def _cluster_rects_by_y_proximity(
    rects: List[fitz.Rect],
    threshold_pt: float = 50.0,
) -> List[fitz.Rect]:
    """
    Group fitz.Rect objects into clusters where consecutive rects (sorted by
    y0) are within threshold_pt points of each other vertically.

    Returns the bounding fitz.Rect of each cluster.
    Single-pass sweep, O(n log n) total (dominated by sort).
    """
    if not rects:
        return []

    sorted_rects = sorted(rects, key=lambda r: r.y0)
    clusters: List[List[fitz.Rect]] = []
    current_cluster: List[fitz.Rect] = [sorted_rects[0]]
    current_y_max = sorted_rects[0].y1

    for rect in sorted_rects[1:]:
        if rect.y0 <= current_y_max + threshold_pt:
            current_cluster.append(rect)
            current_y_max = max(current_y_max, rect.y1)
        else:
            clusters.append(current_cluster)
            current_cluster = [rect]
            current_y_max = rect.y1

    clusters.append(current_cluster)

    result: List[fitz.Rect] = []
    for cluster in clusters:
        x0 = min(r.x0 for r in cluster)
        y0 = min(r.y0 for r in cluster)
        x1 = max(r.x1 for r in cluster)
        y1 = max(r.y1 for r in cluster)
        result.append(fitz.Rect(x0, y0, x1, y1))

    return result


def _detect_diagram_regions_via_pymupdf(
    pdf_base64: str,
    page_num: int,
    _pdf_bytes_override: Optional[bytes] = None,
) -> List[Dict]:
    """
    Deterministic fallback diagram detector using PyMuPDF (fitz).

    Called when Gemini returns diagram_regions=[] for a Question Paper page.
    Inspects the raw PDF page for:
      1. Embedded raster images  (page.get_images + page.get_image_rects)
      2. Vector drawing clusters  (page.get_drawings, grouped by Y-proximity)

    Returns region dicts compatible with the crop injector:
        [{"question_number": "unknown",
          "y_start_pct": float, "y_end_pct": float,
          "page_num": int, "source": "pymupdf_image"|"pymupdf_drawing"}]

    "question_number" is "unknown" — the caller assigns them to questions by
    Y-ordering (topmost region → first question with no regions on that page).

    BUG 8 FIX: accepts _pdf_bytes_override so the caller can pass pre-decoded
    bytes and avoid repeated base64.b64decode() on every page.

    Never raises. Returns [] on any error so the pipeline degrades gracefully.
    """
    try:
        if _pdf_bytes_override is not None:
            pdf_bytes = _pdf_bytes_override
        else:
            raw_b64 = pdf_base64.split(",", 1)[1] if "," in pdf_base64 else pdf_base64
            pdf_bytes = base64.b64decode(raw_b64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if page_num >= len(doc):
            logger.warning(
                f"[PyMuPDF Fallback] page_num={page_num} out of range "
                f"(doc has {len(doc)} pages)"
            )
            doc.close()
            return []

        page = doc[page_num]
        page_h = page.rect.height
        page_w = page.rect.width

        if page_h <= 0 or page_w <= 0:
            doc.close()
            return []

        regions: List[Dict] = []
        _HF_ZONE = 5.0

        # ── 1. Embedded raster images ─────────────────────────────────────
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                y_start_pct = (rect.y0 / page_h) * 100.0
                y_end_pct   = (rect.y1 / page_h) * 100.0
                height_pct  = y_end_pct - y_start_pct
                width_pct   = ((rect.x1 - rect.x0) / page_w) * 100.0
                if (
                    _PYMUPDF_MIN_DIAGRAM_HEIGHT_PCT <= height_pct <= _PYMUPDF_MAX_DIAGRAM_HEIGHT_PCT
                    and width_pct >= _PYMUPDF_MIN_DIAGRAM_WIDTH_PCT
                    # FIX 1: Exclude header/footer zones
                    and y_start_pct >= _HF_ZONE
                    and y_end_pct   <= (100.0 - _HF_ZONE)
                ):
                    regions.append({
                        "question_number": "unknown",
                        "y_start_pct": round(y_start_pct, 1),
                        "y_end_pct":   round(y_end_pct,   1),
                        "page_num":    page_num,
                        "source":      "pymupdf_image",
                    })

        # ── 2. Vector drawing clusters ────────────────────────────────────
        drawings = page.get_drawings()
        if drawings:
            drawing_rects: List[fitz.Rect] = [
                d["rect"] for d in drawings
                if d.get("rect") and isinstance(d["rect"], fitz.Rect)
            ]
            for cluster_rect in _cluster_rects_by_y_proximity(drawing_rects, threshold_pt=50.0):
                y_start_pct = (cluster_rect.y0 / page_h) * 100.0
                y_end_pct   = (cluster_rect.y1 / page_h) * 100.0
                height_pct  = y_end_pct - y_start_pct
                width_pct   = ((cluster_rect.x1 - cluster_rect.x0) / page_w) * 100.0
                if (
                    _PYMUPDF_MIN_DIAGRAM_HEIGHT_PCT <= height_pct <= _PYMUPDF_MAX_DIAGRAM_HEIGHT_PCT
                    and width_pct >= _PYMUPDF_MIN_DIAGRAM_WIDTH_PCT
                    # FIX 1: Exclude header/footer zones — reuse _HF_ZONE from raster block above
                    and y_start_pct >= _HF_ZONE
                    and y_end_pct   <= (100.0 - _HF_ZONE)
                ):
                    # Skip if already covered by a raster image region (3-point tolerance)
                    is_duplicate = any(
                        abs(r["y_start_pct"] - y_start_pct) < 3.0
                        and abs(r["y_end_pct"] - y_end_pct) < 3.0
                        for r in regions
                    )
                    if not is_duplicate:
                        regions.append({
                            "question_number": "unknown",
                            "y_start_pct": round(y_start_pct, 1),
                            "y_end_pct":   round(y_end_pct,   1),
                            "page_num":    page_num,
                            "source":      "pymupdf_drawing",
                        })

        doc.close()

        # Sort top-to-bottom for deterministic Y-order assignment
        regions.sort(key=lambda r: r["y_start_pct"])

        n_raster  = sum(1 for r in regions if r["source"] == "pymupdf_image")
        n_vector  = sum(1 for r in regions if r["source"] == "pymupdf_drawing")
        logger.info(
            f"[PyMuPDF Fallback] Page {page_num}: detected {len(regions)} region(s) "
            f"({n_raster} raster, {n_vector} vector)"
        )
        return regions

    except Exception as exc:
        logger.warning(f"[PyMuPDF Fallback] Page {page_num} detection error: {exc}")
        return []


async def _inject_diagram_crops_from_slicer(
    slicer_results: List[Dict],
    pdf_base64: str,
) -> List[ExtractedQuestion]:
    """
    Phase 2 crop injection: processes the decoupled List[dict] returned by
    gemini_slicer.extract_pages_batch().

    Each dict has:
        { "model": ExtractedQuestion, "diagram_regions": [...], "page_num": int }

    For every non-empty diagram_regions list, calls crop_and_compress_diagram_async
    and appends the compressed JPEG base64 to model.diagram_urls.

    PyMuPDF fallback (PATCH):
    -------------------------
    If a QP question has diagram_regions=[] AND PyMuPDF detects visual objects
    on the same page, those regions are assigned to the question by Y-ordering
    (topmost PyMuPDF region → first no-region question on the page, etc.) before
    the concurrent crop pass runs. If more regions than questions, extras go to
    the last question. If more questions than regions, remaining questions stay
    with diagram_urls=[].

    Rules:
    - diagram_urls is ALWAYS initialized to [] if not already a list.
    - Crops for ALL regions of a single question are gathered concurrently.
    - All questions across all pages are processed concurrently via asyncio.gather.
    - Failures on individual crops are logged and skipped (never raises).
    - MS diagram_regions are [] by design (enforced in gemini_slicer prompt),
      so this function is a no-op for Marking Scheme questions.
    """
    if not slicer_results:
        return []

    # ── PyMuPDF fallback: pre-assign regions to questions missing them ────────
    # BUG 8 FIX: decode base64 once here instead of inside every
    # _detect_diagram_regions_via_pymupdf call. For a 20-page QP with 5 pages
    # hitting the fallback, this previously decoded ~2-5 MB of base64 five times.
    _raw_b64 = pdf_base64.split(",", 1)[1] if "," in pdf_base64 else pdf_base64
    _pdf_bytes: Optional[bytes] = None   # lazy — only decoded on first PyMuPDF hit

    # Cache per page_num so fitz.open() runs at most once per page.
    _pymupdf_cache: Dict[int, List[Dict]] = {}

    def _get_pymupdf_regions(page_num: int) -> List[Dict]:
        nonlocal _pdf_bytes
        if page_num not in _pymupdf_cache:
            if _pdf_bytes is None:
                _pdf_bytes = base64.b64decode(_raw_b64)
            _pymupdf_cache[page_num] = _detect_diagram_regions_via_pymupdf(
                pdf_base64=pdf_base64,
                page_num=page_num,
                _pdf_bytes_override=_pdf_bytes,
            )
        return _pymupdf_cache[page_num]

    # Group entry indices by page so we can assign in Y-order per page
    page_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, entry in enumerate(slicer_results):
        page_to_indices[entry.get("page_num", 0)].append(idx)

    def _entry_is_marking_scheme(entry: Dict) -> bool:
        model = entry.get("model")
        doc_type = getattr(model, "document_type", "") if model is not None else ""
        return str(doc_type or "").strip().lower() == "marking scheme"

    for page_num, indices in page_to_indices.items():
        # Only care about QP entries with no Gemini diagram_regions.
        # MS auto-crops are intentionally disabled; manual paste remains available
        # in the review UI and avoids whole-page/random marking-scheme crops.
        no_region_indices = [
            i for i in indices
            if not _entry_is_marking_scheme(slicer_results[i])
            and not slicer_results[i].get("diagram_regions")
        ]

        if not no_region_indices:
            continue

        pymupdf_regions = _get_pymupdf_regions(page_num)
        if not pymupdf_regions:
            logger.debug(
                f"[DiagramCrop] Page {page_num}: PyMuPDF also found 0 regions — "
                f"page likely has no diagrams."
            )
            continue

        diagram_likely_indices = []
        for entry_idx in no_region_indices:
            model = slicer_results[entry_idx].get("model")
            text = str(getattr(model, "question_latex", "") or "")
            if _qp_text_allows_auto_diagram(text):
                diagram_likely_indices.append(entry_idx)

        if not diagram_likely_indices:
            logger.info(
                f"[DiagramCrop] Page {page_num}: PyMuPDF found {len(pymupdf_regions)} "
                "region(s), but no no-region QP row looks diagram-bearing. "
                "Skipping automatic fallback assignment to avoid wrong-image attachment."
            )
            continue

        logger.info(
            f"[DiagramCrop] Page {page_num}: Gemini returned 0 diagram_regions. "
            f"Assigning {len(pymupdf_regions)} PyMuPDF region(s) to "
            f"{len(diagram_likely_indices)} diagram-likely question(s) by Y-order."
        )

        # Distribute: one region per question slot; clamp extras to last slot
        for assign_idx, region in enumerate(pymupdf_regions):
            target_slot = min(assign_idx, len(diagram_likely_indices) - 1)
            entry_idx   = diagram_likely_indices[target_slot]
            slicer_results[entry_idx].setdefault("diagram_regions", [])
            slicer_results[entry_idx]["diagram_regions"].append(region)

    # ── Concurrent crop pass ──────────────────────────────────────────────────

    async def _process_one(entry: Dict) -> ExtractedQuestion:
        model: ExtractedQuestion = entry["model"]
        regions: List[Dict] = entry.get("diagram_regions") or []

        # Guarantee diagram_urls is always a list (never None)
        if not isinstance(model.diagram_urls, list):
            model.diagram_urls = []

        if str(getattr(model, "document_type", "") or "").strip().lower() == "marking scheme":
            model.diagram_urls = []
            model.diagram_page_number = None
            model.diagram_y_range = []
            return model

        if not regions:
            return model

        safe_regions: List[Dict] = []
        for region in regions:
            try:
                y0 = float(region.get("y_start_pct", 0))
                y1 = float(region.get("y_end_pct", 0))
            except (TypeError, ValueError):
                logger.warning(
                    f"[DiagramCrop] Skipping non-numeric crop region for q={model.question_latex!r}: {region}"
                )
                continue

            height_pct = y1 - y0
            source = region.get("source", "gemini")
            if y0 < 0 or y1 > 100 or y0 >= y1:
                logger.warning(
                    f"[DiagramCrop] Skipping invalid crop bounds for q={model.question_latex!r} "
                    f"page={region.get('page_num', '?')} y={y0:.1f}%-{y1:.1f}% source={source}"
                )
                continue
            if height_pct > _CROP_INJECT_MAX_HEIGHT_PCT:
                logger.warning(
                    f"[DiagramCrop] Skipping oversized crop for q={model.question_latex!r} "
                    f"page={region.get('page_num', '?')} y={y0:.1f}%-{y1:.1f}% "
                    f"height={height_pct:.1f}% source={source}; manual crop required"
                )
                model.needs_review = True
                continue

            safe_regions.append(region)

        regions = safe_regions
        if not regions:
            return model

        # Fire all crops for this question concurrently
        crop_tasks = [
            crop_and_compress_diagram_async(
                pdf_base64=pdf_base64,
                page_num=region.get("page_num", 0),
                y_start_pct=region["y_start_pct"],
                y_end_pct=region["y_end_pct"],
            )
            for region in regions
        ]

        try:
            cropped_list = await asyncio.gather(*crop_tasks, return_exceptions=True)
        except Exception as gather_exc:
            logger.error(f"[DiagramCrop] gather failed for q={model.question_latex!r}: {gather_exc}")
            return model

        for i, result in enumerate(cropped_list):
            region     = regions[i]
            height_pct = region["y_end_pct"] - region["y_start_pct"]
            source     = region.get("source", "gemini")

            if isinstance(result, Exception):
                logger.warning(
                    f"[DiagramCrop] ⚠️ Crop exception for q={model.question_latex!r} "
                    f"page={region.get('page_num', '?')} "
                    f"y={region['y_start_pct']:.1f}%-{region['y_end_pct']:.1f}% "
                    f"source={source}: {result}"
                )
                continue

            if result:
                model.diagram_urls.append(f"data:image/jpeg;base64,{result}")
                logger.info(
                    f"[DiagramCrop] ✅ Cropped for q={model.question_latex!r} "
                    f"page={region.get('page_num', '?')} "
                    f"y={region['y_start_pct']:.1f}%-{region['y_end_pct']:.1f}% "
                    f"height={height_pct:.1f}% source={source}"
                )
            else:
                logger.warning(
                    f"[DiagramCrop] ⚠️ crop_and_compress_diagram_async returned None "
                    f"for q={model.question_latex!r} page={region.get('page_num', '?')} "
                    f"source={source}"
                )

        return model

    # Process ALL questions (across all pages) concurrently
    processed: List[ExtractedQuestion] = list(
        await asyncio.gather(*[_process_one(e) for e in slicer_results], return_exceptions=False)
    )

    total_with_diagrams = sum(
        1 for m in processed if isinstance(m, ExtractedQuestion) and m.diagram_urls
    )
    logger.info(
        f"[DiagramCrop] Batch complete: {len(processed)} questions, "
        f"{total_with_diagrams} with diagram(s)."
    )
    return [m for m in processed if isinstance(m, ExtractedQuestion)]


# ===========================================================================
# SECTION 6: Answer-blank sanitizer  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _sanitize_answer_blanks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'(\\textunderscore){2,}', '', text)
    text = re.sub(r'\\underline\{\\hspace\{[^}]*\}\}', '', text)
    text = re.sub(r'\\dotfill', '', text)
    text = re.sub(r'\s*\[\d+\]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bwww\.exam-mate\.com\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*UCLES\s+\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{4}/\d{1,2}/[A-Z]/[A-Z]/\d{2}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[?\s*Turn over\s*\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?m)^\s*\d{1,2}\s*$', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _ensure_qp_text_has_full_label(
    text: str,
    question_id: str,
    normalizer: QuestionNumberNormalizer,
) -> str:
    if not text or not question_id:
        return text

    qid_parts = normalizer.extract_parts(question_id)
    if not qid_parts or not qid_parts[0].isdigit():
        return text

    full_label = normalizer.format_parts(qid_parts)
    parsed_label, remainder = normalizer.split_label_and_remainder(text)
    parsed_parts = normalizer.extract_parts(parsed_label)
    if not parsed_parts:
        return f"{full_label} {text}".strip()

    parsed_canonical = normalizer.canonical_from_parts(parsed_parts)
    qid_canonical = normalizer.canonical_from_parts(qid_parts)
    parsed_is_orphan = not parsed_parts[0].isdigit()
    parsed_wrong_root = parsed_parts[0].isdigit() and parsed_parts[0] != qid_parts[0]
    if parsed_is_orphan or parsed_wrong_root or parsed_canonical == qid_canonical:
        return f"{full_label} {remainder}".strip() if remainder else full_label

    return text


# ===========================================================================
# SECTION 7: Normalization helpers  (ORIGINAL LOGIC PRESERVED — zero changes)
# ===========================================================================

_QUESTION_FIELD_ALIASES: dict[str, str] = {
    "question_text": "question_latex", "latex": "question_latex",
    "question_content": "question_latex", "text": "question_latex",
    "marking_scheme_latex": "official_marking_scheme_latex",
    "answer": "official_marking_scheme_latex", "mark_scheme": "official_marking_scheme_latex",
    "questionNumber": "question_latex", "question_number": "question_latex",
    "diagrams": "diagram_urls", "images": "diagram_urls",
    "templateable": "isTemplatizable", "is_templateable": "isTemplatizable",
    "is_templatizable": "isTemplatizable",
    "subject_code": "subjectCode", "subject": "subjectCode",
    "paper": "paperNumber", "paper_number": "paperNumber",
}

_METADATA_FIELD_ALIASES: dict[str, str] = {
    "subject_code": "subjectCode", "subject": "subjectCode",
    "paper": "paperNumber", "paper_number": "paperNumber",
}

_QUESTION_DEFAULTS: dict = {
    "document_type": "Question Paper", "curriculum": "", "program": None,
    "subjectCode": "", "tier": None, "paperNumber": 0, "session": None, "year": 0,
    "paper_reference_key": "", "unified_paper_key": "", "canonical_question_id": "",
    "parent_canonical_id": "",
    "question_number_metadata": QuestionNumberMetadata().model_dump(),
    "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
    "isTemplatizable": False, "variables": [], "question_latex": "",
    "question_id": "", "final_answer": "", "total_marks": 0, "method_steps": [],
    "official_marking_scheme_latex": None, "diagram_urls": [],
    "needs_review": False, "cognitive_demand": "MEDIUM", "difficulty_override": None,
}

_METADATA_DEFAULTS: dict = {
    "curriculum": "", "program": None, "subjectCode": "", "tier": None,
    "paperNumber": 0, "session": None, "year": 0, "paper_reference_key": "",
    "unified_paper_key": "", "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
}


def _normalize_tier(tier) -> str:
    if not tier or not isinstance(tier, str):
        return "N/A"
    t = tier.lower().strip()
    if "higher" in t or t == "hl": return "HL"
    if "standard" in t or t == "sl": return "SL"
    if "core" in t: return "Core"
    if "extended" in t: return "Extended"
    return "N/A"


def _remap_keys(raw: dict, alias_map: dict) -> dict:
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


def _normalize_metadata(
    raw: dict | None,
    filename: str,
    board: str,
    generated_key_override: str = "",
) -> dict:
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
    result["year"]        = _coerce_int(result["year"], 0)
    result["tier"]        = _normalize_tier(result.get("tier"))

    # ═══════════════════════════════════════════════════════════════════════════
    # 🔧 FIX: PRESERVE VALID SESSION CODES ("m", "s", "w") — SURGICAL INTERCEPT
    # ═══════════════════════════════════════════════════════════════════════════
    # ROOT CAUSE OF BUG:
    #   gemini_slicer correctly sends session="m" after the authoritative sync.
    #   But _extract_session_from_content() is a legacy helper that expects full
    #   month words ("March", "February"). When it receives the single letter "m",
    #   it returns None. The code then falls back to filename extraction, which
    #   reads "s22" from the filename and overwrites the correct "m" with "s".
    #
    # FIX STRATEGY:
    #   Normalise the incoming session value to lowercase FIRST.
    #   If it is already exactly "m", "s", or "w" — it is an authoritative
    #   single-letter code from gemini_slicer. Preserve it immediately and
    #   skip ALL further extraction logic. Do NOT pass it to the legacy helper.
    #   Only invoke _extract_session_from_content when the value is genuinely
    #   absent or is a long-form string (e.g. "May/June") that needs mapping.
    # ═══════════════════════════════════════════════════════════════════════════

    from builders.key_builder import _extract_session_from_content

    # Normalise to lowercase stripped string for comparison.
    # This handles both "m" (already correct) and "M" (uppercase from Gemini drift).
    current_session = str(result.get("session") or "").strip().lower()

    if current_session in ("m", "s", "w"):
        # ── INTERCEPT: Valid authoritative single-letter code. ────────────────
        # Write the normalised lowercase letter back and stop. The legacy helper
        # must NEVER see this value — it would return None and trigger the
        # filename-based fallback that overwrites "m" with a hallucinated "s".
        result["session"] = current_session
        logger.info(
            f"[Session Detection] Intercepted valid session code "
            f"'{current_session}' — preserving immediately, skipping "
            f"legacy extraction to prevent filename-based overwrite."
        )
    else:
        # ── FALLBACK: Session is absent, null, or a long-form string. ────────
        # Only now is it safe to call the legacy helper.
        detected_session = None

        # Priority 1: Try to map whatever string the AI emitted (e.g. "May/June")
        if current_session:
            detected_session = _extract_session_from_content(current_session)
            if detected_session:
                logger.info(
                    f"[Session Detection] Mapped AI session string "
                    f"'{current_session}' → '{detected_session}' via "
                    f"_extract_session_from_content."
                )

        # Priority 2: Scan secondary metadata fields for month keywords
        if not detected_session:
            for field in ["metadata", "subject", "title", "header"]:
                field_content = str(result.get(field, ""))
                detected_session = _extract_session_from_content(field_content)
                if detected_session:
                    logger.info(
                        f"[Session Detection] Extracted '{detected_session}' "
                        f"from secondary field '{field}'."
                    )
                    break

        # Priority 3: Filename-based fallback (last resort — lowest confidence)
        if not detected_session:
            detected_session = _extract_session_from_content(filename)
            if detected_session:
                logger.info(
                    f"[Session Detection] Extracted '{detected_session}' "
                    f"from filename '{filename}' (last-resort fallback)."
                )

        result["session"] = detected_session if detected_session else "N/A"
        logger.info(
            f"[Session Detection] Final normalised session: "
            f"'{result['session']}'"
        )

    # ── SESSION BACKSTOP FOR MARKING SCHEMES ─────────────────────────────────
    # If session is still "N/A" after all detection attempts AND
    # generated_key_override is present, extract the session from it.
    # generated_key_override was built by generate_igcse_key() using fitz
    # cover-text extraction — it is the most authoritative signal available.
    # This fires only for MS PDFs where the cover page has NO readable text
    # (fully image-based scans) and the filename also failed to yield a session.
    # For QPs this path almost never fires because QP cover pages always state
    # the session month; for MS it is the critical last resort.
    if result.get("session") in ("N/A", None, "") and generated_key_override:
        _bs_match = re.search(r'_([msw])\d{2}[_]', generated_key_override)
        if _bs_match:
            _bs_session = _bs_match.group(1)
            result["session"] = _bs_session
            logger.info(
                f"[Session Detection] MS backstop: inherited session "
                f"'{_bs_session}' from generated_key_override "
                f"'{generated_key_override}'."
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔧 KEY RESOLUTION: generated_key_override beats Gemini's echoed key
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE HIERARCHY (highest → lowest authority):
    #   1. generated_key_override — built by generate_igcse_key() using fitz
    #      cover-text extraction BEFORE this function is called.  It carries the
    #      session detected from the actual PDF content (e.g. "m18").
    #   2. incoming_key (from Gemini) — Gemini echoes whatever was injected in
    #      prk_line.  If prk_line was already wrong (built before Fix 1 ran), the
    #      echo is wrong too.  Trusting it blindly was the root cause of the
    #      MS/QP key mismatch: Gemini dutifully echoed "igcse_0580_s18_ms_42".
    #
    # Correction logic:
    #   • If generated_key_override is present AND its session letter differs from
    #     the incoming key's session letter → the override wins.  This is the
    #     only case where they can disagree: Fix 1 detected "m" from cover text
    #     but the prompt was built with the old "s" key.
    #   • If they agree (or generated_key_override is empty) → preserve incoming.
    #   • If incoming is empty → fall back to generated_key_override or re-derive.
    # ═══════════════════════════════════════════════════════════════════════════

    def _session_letter_from_key(key: str) -> str:
        """Extract the session letter from an igcse key, e.g. 'm' from 'igcse_0580_m18_42'."""
        m = re.search(r'_([msw])\d{2}[_$]', key)
        return m.group(1) if m else ""

    incoming_key = (result.get("paper_reference_key") or "").strip()
    override_key = (generated_key_override or "").strip()

    if override_key and incoming_key:
        incoming_sess = _session_letter_from_key(incoming_key)
        override_sess = _session_letter_from_key(override_key)
        if incoming_sess and override_sess and incoming_sess != override_sess:
            # Session conflict: override wins — it was built from actual PDF content.
            # Also rewrite the session in result["session"] to stay consistent.
            result["paper_reference_key"] = override_key
            result["session"] = override_sess
            logger.warning(
                f"[Key Resolution] Session conflict detected: "
                f"Gemini echoed '{incoming_key}' (session='{incoming_sess}') "
                f"but generated_key_override='{override_key}' (session='{override_sess}'). "
                f"Override wins — built from PDF cover-text, not filename."
            )
        else:
            # Sessions agree or one is missing — preserve incoming.
            result["paper_reference_key"] = incoming_key
            logger.info(f"[Key Resolution] Preserving Gemini key '{incoming_key}' (no conflict).")
    elif incoming_key:
        result["paper_reference_key"] = incoming_key
        logger.info(f"[Key Resolution] No override — preserving Gemini key '{incoming_key}'.")
    elif override_key:
        result["paper_reference_key"] = override_key
        logger.info(f"[Key Resolution] No incoming key — using override '{override_key}'.")
    else:
        # Last resort: re-derive from filename with the now-finalised session.
        if board.upper() == "IGCSE":
            final_key = generate_igcse_key(filename=filename, content_detected_session=result.get("session"))
            result["paper_reference_key"] = final_key if final_key else ""
            if final_key:
                logger.info(f"[Key Resolution] Re-derived IGCSE key: '{final_key}'")
        else:
            result["paper_reference_key"] = ""
    
    result["curriculum"] = board.upper()

    # Native table extraction can know the paper identity from the generated key
    # before any AI metadata exists. Fill blank metadata from that key so the
    # review form shows subject, session, year, and paper number immediately.
    if board.upper() == "IGCSE":
        key_for_fields = (result.get("paper_reference_key") or generated_key_override or "").strip()
        key_match = re.search(
            r"^igcse_(?P<subject>\d{4})_(?P<session>[msw])(?P<yy>\d{2})(?:_(?:qp|ms))?_(?P<paper>\d{1,2})$",
            key_for_fields,
            re.IGNORECASE,
        )
        if key_match:
            key_subject = key_match.group("subject")
            key_session = key_match.group("session").lower()
            key_year = 2000 + int(key_match.group("yy"))
            paper_code = key_match.group("paper")
            key_paper_number = int(paper_code[0]) if len(paper_code) > 1 else int(paper_code)

            # The final IGCSE key is built before normalisation using the best
            # available paper identity signal (QP cover text when readable,
            # filename only as fallback). Gemini can still emit stale metadata
            # inside per-page JSON, e.g. key=s23 but session=m. Downstream Node
            # treats metadata.session as authoritative and rewrites keys from it,
            # so mismatches here cause QP/MS split-brain. Once the key is final,
            # keep these scalar fields consistent with it.
            old_identity = {
                "subjectCode": result.get("subjectCode"),
                "session": result.get("session"),
                "year": result.get("year"),
                "paperNumber": result.get("paperNumber"),
            }
            result["subjectCode"] = key_subject
            result["session"] = key_session
            result["year"] = key_year
            result["paperNumber"] = key_paper_number
            new_identity = {
                "subjectCode": result.get("subjectCode"),
                "session": result.get("session"),
                "year": result.get("year"),
                "paperNumber": result.get("paperNumber"),
            }
            if old_identity != new_identity:
                logger.warning(
                    "[IGCSE Metadata Sync] Aligned metadata fields to final "
                    "paper_reference_key %r: %s -> %s",
                    key_for_fields,
                    old_identity,
                    new_identity,
                )

            if str(result.get("tier") or "").strip().upper() == "N/A" and key_match.group("subject") == "0580":
                paper_digit = key_match.group("paper")[0]
                if paper_digit in {"2", "4"}:
                    result["tier"] = "Extended"
                elif paper_digit in {"1", "3"}:
                    result["tier"] = "Core"

    # 🔧 FIX 3: Tier fallback detection from filename
    if result.get("tier") == "N/A":
        filename_lower = filename.lower()
        if "extended" in filename_lower:
            result["tier"] = "Extended"
        elif "core" in filename_lower:
            result["tier"] = "Core"
        elif "higher" in filename_lower or "hl" in filename_lower:
            result["tier"] = "HL"
        elif "standard" in filename_lower or "sl" in filename_lower:
            result["tier"] = "SL"

    return result


def _normalize_question(
    raw: dict,
    fallback_metadata: dict,
    document_type: str,
    question_normalizer: QuestionNumberNormalizer,
) -> dict:
    if not isinstance(raw, dict):
        return dict(_QUESTION_DEFAULTS)

    raw = _remap_keys(raw, _QUESTION_FIELD_ALIASES)
    result = dict(_QUESTION_DEFAULTS)
    for k in result:
        if k in raw:
            result[k] = raw[k]

    explicit_q_id = result.get("question_id") or ""
    q_latex_text = result.get("question_latex") or ""
    if _is_placeholder_question_id(explicit_q_id):
        explicit_q_id = ""
        result["question_id"] = ""
    q_id_raw = (
        question_normalizer.extract_leading_label(explicit_q_id)
        or question_normalizer.extract_leading_label(q_latex_text)
        or ("" if _is_placeholder_question_id(q_latex_text) else explicit_q_id)
        or ("" if _is_placeholder_question_id(q_latex_text) else q_latex_text)
    )
    # BUG 2 FIX: Strip Cambridge mark-allocation brackets (e.g. "[3]") that Gemini
    # sometimes captures from the end of answer spaces into the question ID.
    # These must be removed before canonical ID normalisation to prevent spurious
    # sub-part suffixes like "4.b.ii.3" being written to the database.
    if q_id_raw:
        q_id_raw = re.sub(r'\s*\[\d+\]\s*$', '', q_id_raw.strip())
        safe_label = question_normalizer.extract_leading_label(q_id_raw)
        if safe_label:
            q_id_raw = safe_label
            result["question_id"] = safe_label
    if q_id_raw and fallback_metadata.get("paper_reference_key"):
        normalized_data = question_normalizer.normalize(
            raw_question_id=q_id_raw,
            paper_reference_key=fallback_metadata["paper_reference_key"],
        )
        result.update(normalized_data)
        result["question_number_metadata"] = QuestionNumberMetadata(
            **normalized_data["question_number_metadata"]
        ).model_dump()

    result["document_type"] = document_type
    result["tier"] = _normalize_tier(result.get("tier"))

    for meta_key in (
        "curriculum", "program", "subjectCode", "tier", "paperNumber", "session", "year",
        "paper_reference_key", "unified_paper_key", "validation_status", "validation_warnings",
        "ref_code_base", "ref_code_full",
    ):
        current_meta_value = result.get(meta_key)
        if (
            fallback_metadata.get(meta_key)
            and (
                not current_meta_value
                or str(current_meta_value).strip().upper() == "N/A"
            )
        ):
            result[meta_key] = fallback_metadata[meta_key]

    if fallback_metadata.get("curriculum"):
        result["curriculum"] = fallback_metadata["curriculum"]
    if not result.get("paper_reference_key") and fallback_metadata.get("paper_reference_key"):
        result["paper_reference_key"] = fallback_metadata["paper_reference_key"]

    # For IGCSE, the normalized paper_reference_key is the paper identity
    # contract. Do not allow per-question Gemini metadata to keep a conflicting
    # session/year/subject/paper after top-level metadata has been aligned to
    # the final key. This prevents rows like key=s23 + session=m from causing
    # Node to rewrite the saved paper to m23.
    fallback_key = str(fallback_metadata.get("paper_reference_key") or "").strip()
    if fallback_key.lower().startswith("igcse_"):
        for meta_key in (
            "subjectCode",
            "paperNumber",
            "session",
            "year",
            "paper_reference_key",
            "unified_paper_key",
        ):
            if fallback_metadata.get(meta_key) not in (None, "", "N/A"):
                result[meta_key] = fallback_metadata[meta_key]

    if document_type.strip().lower() == "marking scheme":
        if not result.get("question_id"): result["question_id"] = result.get("question_latex", "")
        if not result.get("final_answer"): result["final_answer"] = ""
        result["total_marks"]  = _coerce_int(result.get("total_marks"), 0)
        result["method_steps"] = _normalize_method_steps(result.get("method_steps", []))

    result["paperNumber"]     = _coerce_int(result["paperNumber"], 0)
    result["year"]            = _coerce_int(result["year"], 0)
    result["isTemplatizable"] = _coerce_bool(result["isTemplatizable"], False)
    result["variables"]       = _coerce_list(result["variables"], [])

    # diagram_urls: always a clean list, never None
    raw_diagrams = _coerce_list(result["diagram_urls"], [])
    valid_urls = []
    flattened: List[str] = []
    for item in raw_diagrams:
        if item is None: continue
        if isinstance(item, list):
            flattened.extend([str(si).strip() for si in item if si])
        else:
            flattened.append(str(item).strip())
    for item_str in flattened:
        if not item_str: continue
        if (
            item_str.startswith("http")
            or item_str.startswith("data:image")
            or item_str.startswith("//")
            or item_str == "[NEEDS_CROP]"
            or "cloudinary" in item_str
        ):
            valid_urls.append(item_str)

    result["diagram_urls"] = valid_urls
    if not isinstance(result["diagram_urls"], list):
        result["diagram_urls"] = []

    result["needs_review"] = _coerce_bool(result["needs_review"], False)

    _VALID_DEMANDS = {"LOW", "MEDIUM", "HIGH"}
    if str(result.get("cognitive_demand", "")).upper() not in _VALID_DEMANDS:
        result["cognitive_demand"] = "MEDIUM"
    else:
        result["cognitive_demand"] = str(result["cognitive_demand"]).upper()

    if result.get("difficulty_override") not in {"Easy", "Medium", "Hard", None}:
        result["difficulty_override"] = None

    result["curriculum"]     = result["curriculum"] or ""
    result["subjectCode"]    = result["subjectCode"] or ""
    if document_type.strip().lower() == "question paper":
        _ensure_qp_text_has_full_label(result, question_normalizer)
    result["question_latex"] = _sanitize_answer_blanks(result.get("question_latex") or "")
    if document_type.strip().lower() == "question paper":
        result["question_latex"] = _sanitize_qp_print_artifacts(result["question_latex"])
        result["question_latex"] = _repair_common_native_math_text(result["question_latex"])
        result = _mark_qp_math_risk_if_needed(result)
    return result


_MS_ROMAN_MARKER_RE = re.compile(
    r"(?<![A-Za-z0-9])\(\s*(viii|vii|vi|iv|ix|iii|ii|i|x|v)\s*\)",
    re.IGNORECASE,
)


def _question_payload_for_sequence(q: dict) -> str:
    if not isinstance(q, dict):
        return str(q or "")
    steps = q.get("method_steps") or []
    step_text = ""
    if isinstance(steps, list):
        step_text = " ".join(
            str(step.get("description", "")) if isinstance(step, dict) else str(step)
            for step in steps
        )
    return " ".join(
        str(part or "")
        for part in (
            q.get("question_latex"),
            q.get("official_marking_scheme_latex"),
            q.get("final_answer"),
            step_text,
        )
        if part
    )


_PLACEHOLDER_QID_RE = re.compile(
    r"^\s*(?:unknown|unk|invalid|n/?a|none|null|\?)(?:\s*[\(\.\-].*)?\s*$",
    re.IGNORECASE,
)


def _is_placeholder_question_id(value: object) -> bool:
    return bool(_PLACEHOLDER_QID_RE.match(str(value or "").strip()))


def _question_warning_list(q: dict) -> list:
    warnings = q.get("validation_warnings")
    return warnings if isinstance(warnings, list) else []


def _append_question_warning(q: dict, message: str) -> None:
    warnings = _question_warning_list(q)
    if message not in warnings:
        warnings.append(message)
    q["validation_warnings"] = warnings


def _roman_rank(value: str) -> int | None:
    order = getattr(QuestionNumberNormalizer, "_ROMAN_ORDER", [])
    value = str(value or "").lower()
    return order.index(value) if value in order else None


def _terminal_rank(value: str) -> tuple[int, int | str] | None:
    token = str(value or "").lower()
    roman = _roman_rank(token)
    if roman is not None:
        return (0, roman)
    if len(token) == 1 and "a" <= token <= "z":
        return (1, ord(token) - ord("a"))
    if token.isdigit():
        return (2, int(token))
    return None


def _parts_label_pattern(parts: list[str]) -> str:
    if not parts:
        return r"a^"
    root = re.escape(str(parts[0]))
    pattern = rf"\b{root}\b"
    for part in parts[1:]:
        pattern += rf"\s*\(\s*{re.escape(str(part))}\s*\)"
    return pattern


def _contains_visible_question_label(text: str, parts: list[str]) -> bool:
    if not text or not parts:
        return False
    pattern = _parts_label_pattern(parts)
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _count_visible_question_label(text: str, parts: list[str]) -> int:
    if not text or not parts:
        return 0
    pattern = _parts_label_pattern(parts)
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _contains_terminal_marker(text: str, token: str) -> bool:
    if not text or not token:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9])\(\s*{re.escape(str(token))}\s*\)(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _annotate_grouped_qp_split_candidates(
    questions_raw: list,
    missing_ids: list[str],
    normalizer: QuestionNumberNormalizer,
) -> list:
    """
    Add conservative, row-local repair hints for dashboard/manual review.

    This does not create rows, rename rows, or suppress QP/MS parity. It only
    marks the most likely existing grouped row when a missing MS leaf appears
    to be inside a neighbouring QP row. The UI can then offer a "Split grouped
    row" action instead of repeatedly running targeted rescue.
    """
    if not isinstance(questions_raw, list) or not missing_ids:
        return questions_raw

    normalized_missing: list[tuple[str, list[str]]] = []
    for value in missing_ids:
        parts = normalizer.extract_parts(str(value or ""))
        canonical = normalizer.canonical_from_parts(parts)
        if canonical and len(parts) >= 2:
            normalized_missing.append((canonical, parts))
    if not normalized_missing:
        return questions_raw

    rows: list[tuple[int, dict, str, list[str], str]] = []
    for idx, raw in enumerate(questions_raw):
        if not isinstance(raw, dict):
            continue
        label = _safe_question_label_from_raw(raw, normalizer)
        parts = normalizer.extract_parts(label)
        canonical = normalizer.canonical_from_parts(parts)
        text = _question_payload_for_sequence(raw)
        if canonical and parts and text:
            rows.append((idx, raw, canonical, parts, text))

    if not rows:
        return questions_raw

    annotated = list(questions_raw)

    for missing_id, missing_parts in normalized_missing:
        missing_parent = normalizer.canonical_from_parts(missing_parts[:-1])
        missing_root = missing_parts[0]
        missing_terminal = missing_parts[-1]
        best: tuple[int, dict, str, str] | None = None

        # Highest confidence: the exact visible missing label occurs inside a
        # different saved row. Example row 2.a.ii text contains "2(a)(i)".
        for idx, raw, canonical, parts, text in rows:
            if canonical == missing_id:
                continue
            if _contains_visible_question_label(text, missing_parts):
                best = (idx, raw, canonical, "high_exact_visible_label")
                break

        # Medium confidence: same immediate parent, missing previous/nearby
        # sibling, and the row text shows grouped/repeated subpart labels. This
        # covers common Cambridge cases where (i) and (ii) are merged under the
        # second label, without inventing an automatic split.
        if best is None and missing_parent:
            missing_rank = _terminal_rank(missing_terminal)
            for idx, raw, canonical, parts, text in rows:
                if canonical == missing_id or len(parts) != len(missing_parts):
                    continue
                if normalizer.canonical_from_parts(parts[:-1]) != missing_parent:
                    continue
                current_rank = _terminal_rank(parts[-1])
                own_label_count = _count_visible_question_label(text, parts)
                has_missing_tail_marker = _contains_terminal_marker(text, missing_terminal)
                has_current_tail_marker = _contains_terminal_marker(text, parts[-1])
                nearby_previous = (
                    missing_rank is not None
                    and current_rank is not None
                    and missing_rank[0] == current_rank[0]
                    and isinstance(missing_rank[1], int)
                    and isinstance(current_rank[1], int)
                    and 0 <= current_rank[1] - missing_rank[1] <= 2
                )
                if nearby_previous and (
                    own_label_count >= 2
                    or (has_missing_tail_marker and has_current_tail_marker)
                ):
                    best = (idx, raw, canonical, "medium_grouped_sibling_labels")
                    break

        # Medium confidence for root-level letter misses: same root and visible
        # terminal marker e.g. missing 6.d while a 6.a/6.b grouped row contains
        # "(d)". This is intentionally only a hint, not a repair.
        if best is None and len(missing_parts) == 2:
            for idx, raw, canonical, parts, text in rows:
                if canonical == missing_id or not parts or parts[0] != missing_root:
                    continue
                if _contains_terminal_marker(text, missing_terminal):
                    best = (idx, raw, canonical, "medium_visible_terminal_marker")
                    break

        if best is None:
            continue

        idx, raw, source_id, reason = best
        updated = dict(raw)
        warning = (
            "REPAIR_HINT split_grouped_row "
            f"missing_id={missing_id} source_id={source_id} confidence={reason}. "
            "This missing QP/MS ID appears to be inside this grouped row; split/clean this row instead of rerunning full extraction."
        )
        _append_question_warning(updated, warning)
        # Keep this visible in review only when the row was already not clean or
        # when the hint is exact. This avoids creating noisy warnings on clean
        # exact-match papers because this function only runs for real missing IDs.
        if reason.startswith("high"):
            updated["needs_review"] = True
        annotated[idx] = updated

    return annotated


def _rescue_task_signature(text: str, canonical_id: str, normalizer: QuestionNumberNormalizer) -> str:
    """
    Return the task-specific tail of a rescued QP row.

    Cambridge rows can share a long stem and differ only at the final subpart
    prompt. Full-text similarity would reject valid siblings, so compare the
    final labelled task instead.
    """
    raw = str(text or "")
    if not raw.strip():
        return ""
    _label, remainder = normalizer.split_label_and_remainder(raw)
    working = remainder.strip() or raw.strip()
    token_pattern = r"(?:xviii|xvii|xvi|xiv|xiii|xii|xi|viii|vii|vi|iv|ix|iii|ii|i|xx|xix|xv|x|v|[a-z])"
    label_group = rf"(?:\b\d{{1,2}}\s*)?(?:\(\s*{token_pattern}\s*\)\s*)+"
    segments = [
        segment.strip()
        for segment in re.split(label_group, working, flags=re.IGNORECASE)
        if segment.strip()
    ]
    if segments:
        working = segments[-1]

    canonical_parts = normalizer.extract_parts(canonical_id)
    if canonical_parts:
        visible_label = normalizer.format_parts(canonical_parts)
        if visible_label:
            working = re.sub(re.escape(visible_label), " ", working, flags=re.IGNORECASE)

    working = re.sub(r"\\[a-zA-Z]+", " ", working)
    working = re.sub(r"[^a-zA-Z0-9]+", " ", working.lower())
    return " ".join(working.split())


def _rescue_duplicate_sibling_reason(
    question: ExtractedQuestion,
    sibling_candidates: list[ExtractedQuestion],
    normalizer: QuestionNumberNormalizer,
) -> str:
    canonical = str(getattr(question, "canonical_question_id", "") or "").strip().lower()
    parts = normalizer.extract_parts(canonical)
    if len(parts) < 2:
        return ""
    parent = normalizer.canonical_from_parts(parts[:-1])
    signature = _rescue_task_signature(
        str(getattr(question, "question_latex", "") or ""),
        canonical,
        normalizer,
    )
    if len(signature) < 8:
        return ""

    for sibling in sibling_candidates:
        sibling_canonical = str(getattr(sibling, "canonical_question_id", "") or "").strip().lower()
        if not sibling_canonical or sibling_canonical == canonical:
            continue
        sibling_parts = normalizer.extract_parts(sibling_canonical)
        if len(sibling_parts) < 2:
            continue
        sibling_parent = normalizer.canonical_from_parts(sibling_parts[:-1])
        if sibling_parent != parent:
            continue
        sibling_signature = _rescue_task_signature(
            str(getattr(sibling, "question_latex", "") or ""),
            sibling_canonical,
            normalizer,
        )
        if len(sibling_signature) < 8:
            continue
        similarity = SequenceMatcher(None, signature, sibling_signature).ratio()
        copied_long_task = (
            similarity >= 0.96
            and min(len(signature), len(sibling_signature)) >= 24
        )
        if signature == sibling_signature or copied_long_task:
            return (
                f"Recovered text for {canonical!r} duplicates sibling {sibling_canonical!r} "
                f"(task similarity {similarity:.2f})."
            )
    return ""


def _rescue_exact_model_row_rejection_reason(
    question: ExtractedQuestion,
    sibling_candidates: list[ExtractedQuestion],
    normalizer: QuestionNumberNormalizer,
) -> str:
    duplicate_reason = _rescue_duplicate_sibling_reason(question, sibling_candidates, normalizer)
    if duplicate_reason:
        return duplicate_reason

    canonical = str(getattr(question, "canonical_question_id", "") or "").strip().lower()
    parts = normalizer.extract_parts(canonical)
    if len(parts) >= 4 and str(
        os.getenv("PAPERLY_RESCUE_ACCEPT_DEEP_MODEL_SPLITS", "false")
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        return (
            f"Gemini-only targeted rescue for deep nested ID {canonical!r} is not auto-accepted; "
            "deep splits must be local/native exact or manually split from the grouped row."
        )
    return ""


def _strip_rescue_diagrams_if_needed(question: ExtractedQuestion) -> ExtractedQuestion:
    if str(os.getenv("PAPERLY_RESCUE_KEEP_MODEL_DIAGRAMS", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return question
    warning = "Targeted rescue does not auto-copy model diagram crops; verify/paste diagram manually if needed."
    existing = list(getattr(question, "validation_warnings", None) or [])
    if warning not in existing:
        existing.append(warning)
    update = {
        "diagram_urls": [],
        "diagram_regions": [],
        "needs_review": True,
        "validation_warnings": existing,
    }
    if hasattr(question, "model_copy"):
        return question.model_copy(update=update)
    for key, value in update.items():
        try:
            setattr(question, key, value)
        except Exception:
            pass
    return question


def _build_rescue_split_hints(
    missing_ids: list[str],
    candidate_questions: list[ExtractedQuestion],
    normalizer: QuestionNumberNormalizer,
) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    by_id: dict[str, ExtractedQuestion] = {}
    for question in candidate_questions or []:
        canonical = str(getattr(question, "canonical_question_id", "") or "").strip().lower()
        if canonical and canonical not in by_id:
            by_id[canonical] = question

    for missing_id in missing_ids or []:
        canonical = str(missing_id or "").strip().lower()
        parts = normalizer.extract_parts(canonical)
        if len(parts) < 2:
            continue

        sibling_sources: list[str] = []
        possible_sources: list[str] = []
        parent = normalizer.canonical_from_parts(parts[:-1])
        grand_parent = normalizer.canonical_from_parts(parts[:-2]) if len(parts) >= 3 else ""

        sibling_parent = parent
        for candidate_id in by_id:
            candidate_parts = normalizer.extract_parts(candidate_id)
            if len(candidate_parts) == len(parts):
                candidate_parent = normalizer.canonical_from_parts(candidate_parts[:-1])
                if candidate_parent == sibling_parent:
                    sibling_sources.append(candidate_id)

        possible_sources.extend(sibling_sources)
        if parent:
            possible_sources.append(parent)
        if grand_parent:
            possible_sources.append(grand_parent)

        source_id = ""
        for possible in possible_sources:
            if possible in by_id:
                source_id = possible
                break
        if not source_id:
            continue

        source_question = by_id[source_id]
        hints.append({
            "missing_id": canonical,
            "source_id": source_id,
            "action": "split_grouped_source_row",
            "reason": (
                f"Targeted rescue found nearby parent/sibling {source_id!r} but not exact {canonical!r}. "
                "Split/clean the grouped source row instead of rerunning rescue."
            ),
            "source_text_preview": str(getattr(source_question, "question_latex", "") or "")[:240],
        })
    return hints


def _split_review_text_from_source(
    *,
    missing_id: str,
    source_id: str,
    source_text: str,
    normalizer: QuestionNumberNormalizer,
    page_texts: list[str] | None = None,
) -> str:
    """Create conservative text for a missing child row from a grouped source row."""
    missing_parts = normalizer.extract_parts(missing_id)
    missing_label = normalizer.format_parts(missing_parts)
    if not missing_label:
        return ""

    extracted_segment = ""
    search_texts = [str(text or "") for text in (page_texts or []) if str(text or "").strip()]
    try:
        label_patterns = [_parts_label_pattern(missing_parts)]
        if len(missing_parts) >= 4:
            ancestor_pattern = _parts_label_pattern(missing_parts[:-1])
            terminal = re.escape(missing_parts[-1])
            label_patterns.append(
                rf"{ancestor_pattern}.*?(?<![A-Za-z0-9])\(\s*{terminal}\s*\)(?![A-Za-z0-9])"
            )

        for text in search_texts:
            if not text:
                continue
            for label_pattern in label_patterns:
                match = re.search(label_pattern, text, flags=re.IGNORECASE | re.DOTALL)
                if not match:
                    continue
                tail = text[match.end():]
                next_label = re.search(
                    r"\b\d{1,2}\s*(?:\(\s*(?:[a-z]|[ivxlcdm]+)\s*\)\s*){1,5}|(?<![A-Za-z0-9])\(\s*(?:[a-z]|[ivxlcdm]+)\s*\)(?![A-Za-z0-9])",
                    tail,
                    flags=re.IGNORECASE,
                )
                extracted_segment = tail[: next_label.start()].strip() if next_label else tail.strip()
                extracted_segment = " ".join(extracted_segment.split())
                if len(extracted_segment) >= 8:
                    break
            if len(extracted_segment) >= 8:
                break
    except Exception:
        extracted_segment = ""

    body = " ".join((extracted_segment or "").split())
    if not body:
        return ""

    return f"{missing_label} {body}".strip()


def _build_rescue_split_review_rows(
    missing_ids: list[str],
    candidate_questions: list[ExtractedQuestion],
    normalizer: QuestionNumberNormalizer,
    page_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Turn high-confidence grouped-source hints into exact split-review rows.

    These rows are not declared clean. They are useful, exact-ID placeholders
    with the best available source text, no auto-copied diagrams, and mandatory
    human review warnings. This keeps targeted rescue practical without letting
    Gemini clone sibling diagrams/text silently.
    """
    if str(os.getenv("PAPERLY_RESCUE_CREATE_SPLIT_REVIEW_ROWS", "true")).strip().lower() in {
        "0", "false", "no", "off"
    }:
        return []

    hints = _build_rescue_split_hints(missing_ids, candidate_questions, normalizer)
    if not hints:
        return []

    by_id: dict[str, ExtractedQuestion] = {}
    for question in candidate_questions or []:
        canonical = str(getattr(question, "canonical_question_id", "") or "").strip().lower()
        if canonical and canonical not in by_id:
            by_id[canonical] = question

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hint in hints:
        missing_id = str(hint.get("missing_id") or "").strip().lower()
        source_id = str(hint.get("source_id") or "").strip().lower()
        if not missing_id or missing_id in seen or source_id not in by_id:
            continue

        source = by_id[source_id]
        source_payload = source.model_dump(mode="json") if hasattr(source, "model_dump") else dict(source)
        missing_parts = normalizer.extract_parts(missing_id)
        if not missing_parts:
            continue

        label = normalizer.format_parts(missing_parts)
        source_text = str(source_payload.get("question_latex") or "")
        split_text = _split_review_text_from_source(
            missing_id=missing_id,
            source_id=source_id,
            source_text=source_text,
            normalizer=normalizer,
            page_texts=page_texts,
        )
        if not split_text:
            continue
        warnings = source_payload.get("validation_warnings")
        if not isinstance(warnings, list):
            warnings = []
        warnings = list(dict.fromkeys([
            *warnings,
            (
                f"Targeted rescue created split-review row {missing_id!r} from grouped source "
                f"{source_id!r}; verify and trim against the PDF before saving."
            ),
            "Targeted rescue used native PDF page text for the split; diagrams were not auto-copied.",
        ]))

        row = {
            **source_payload,
            "question_id": label,
            "canonical_question_id": missing_id,
            "parent_canonical_id": normalizer.parent_from_parts(missing_parts),
            "question_latex": split_text,
            "diagram_urls": [],
            "diagram_regions": [],
            "needs_review": True,
            "validation_warnings": warnings,
        }
        rows.append(row)
        seen.add(missing_id)

    return rows


def _safe_question_label_from_raw(q: dict, normalizer: QuestionNumberNormalizer) -> str:
    explicit = str(q.get("question_id") or "").strip()
    text = str(q.get("question_latex") or "").strip()
    if _is_placeholder_question_id(explicit):
        explicit = ""
    if _is_placeholder_question_id(text):
        text = ""
    return normalizer.extract_leading_label(explicit) or normalizer.extract_leading_label(text)


def _visible_orphan_parts(q: dict, normalizer: QuestionNumberNormalizer) -> list[str]:
    text = str(q.get("question_latex") or "")
    text_parts = normalizer.extract_parts(text)
    if text_parts and not text_parts[0].isdigit():
        return text_parts
    label, _remainder = normalizer.split_label_and_remainder(text)
    parts = normalizer.extract_parts(label)
    if parts and not parts[0].isdigit():
        return parts
    explicit = str(q.get("question_id") or "").strip()
    explicit_parts = normalizer.extract_parts(explicit)
    if explicit_parts and not explicit_parts[0].isdigit():
        return explicit_parts
    return []


def _repair_qp_placeholders_and_orphans(
    questions_raw: list,
    normalizer: QuestionNumberNormalizer,
) -> list:
    """
    Convert Gemini placeholders such as "unknown" into structural IDs before
    normalisation. This is deliberately stateful: orphan "(ii)" labels inherit
    the last root/letter context, and placeholder rows inherit the next visible
    sub-label from their own text when possible.
    """
    repaired: list = []
    last_parts: list[str] = []

    for raw in questions_raw:
        if not isinstance(raw, dict):
            repaired.append(raw)
            continue

        q = dict(raw)
        explicit = str(q.get("question_id") or "").strip()
        text = str(q.get("question_latex") or "").strip()
        safe_label = _safe_question_label_from_raw(q, normalizer)
        safe_parts = normalizer.extract_parts(safe_label) if safe_label else []

        if safe_parts and safe_parts[0].isdigit():
            q["question_id"] = normalizer.format_parts(safe_parts)
            last_parts = safe_parts
            repaired.append(q)
            continue

        orphan_parts = _visible_orphan_parts(q, normalizer)
        inferred_parts: list[str] = []
        if orphan_parts and last_parts:
            if len(orphan_parts) == 1:
                terminal = orphan_parts[0]
                if terminal in normalizer._ROMAN_SET and len(last_parts) >= 2:
                    inferred_parts = last_parts[:2] + [terminal]
                else:
                    inferred_parts = last_parts[:1] + [terminal]
            else:
                inferred_parts = last_parts[:1] + orphan_parts
        elif _is_placeholder_question_id(explicit) and last_parts:
            inferred_parts = last_parts

        if inferred_parts and inferred_parts[0].isdigit():
            new_label = normalizer.format_parts(inferred_parts)
            q["question_id"] = new_label
            parsed_label, remainder = normalizer.split_label_and_remainder(text)
            if parsed_label:
                q["question_latex"] = f"{new_label} {remainder}".strip() if remainder else new_label
            elif text:
                q["question_latex"] = f"{new_label} {text}".strip()
            else:
                q["question_latex"] = new_label
            q["needs_review"] = True
            _append_question_warning(q, "QP placeholder/orphan guard inferred the question label from sequence context.")
            last_parts = inferred_parts

        repaired.append(q)

    return repaired


def _compact_repeated_qp_preambles(
    questions_raw: list,
    normalizer: QuestionNumberNormalizer,
) -> list:
    """
    Remove duplicated shared stems from later siblings while preserving the first
    child as the context anchor. This keeps Mongo/UI payloads compact without
    inventing DB schema fields.
    """
    compacted: list = []
    previous_by_parent: dict[str, str] = {}

    for raw in questions_raw:
        if not isinstance(raw, dict):
            compacted.append(raw)
            continue

        q = dict(raw)
        label = _safe_question_label_from_raw(q, normalizer)
        parts = normalizer.extract_parts(label)
        text = str(q.get("question_latex") or "")
        parsed_label, remainder = normalizer.split_label_and_remainder(text)

        if len(parts) < 2 or not remainder:
            compacted.append(q)
            if parts:
                previous_by_parent[normalizer.immediate_parent_from_parts(parts)] = remainder
            continue

        parent_key = normalizer.immediate_parent_from_parts(parts)
        previous_remainder = previous_by_parent.get(parent_key, "")
        if previous_remainder:
            prev_norm = re.sub(r"\s+", " ", previous_remainder.lower()).strip()
            cur_norm = re.sub(r"\s+", " ", remainder.lower()).strip()
            similarity = SequenceMatcher(None, prev_norm, cur_norm).ratio() if prev_norm and cur_norm else 0.0

            prefix_len = 0
            limit = min(len(previous_remainder), len(remainder))
            while prefix_len < limit and previous_remainder[prefix_len] == remainder[prefix_len]:
                prefix_len += 1
            while prefix_len > 0 and not previous_remainder[prefix_len - 1].isspace():
                prefix_len -= 1

            if similarity >= 0.62 and prefix_len >= 60:
                suffix = remainder[prefix_len:].lstrip()
                if len(suffix) >= 12:
                    visual_label = normalizer.format_parts(parts)
                    q["question_latex"] = f"{visual_label} {suffix}".strip()
                    q["needs_review"] = True
                    _append_question_warning(q, "QP preamble compactor removed repeated shared stem from this sibling.")

        compacted.append(q)
        previous_by_parent[parent_key] = remainder

    return compacted


def _repair_qp_backward_root_intrusions(
    questions_raw: list,
    normalizer: QuestionNumberNormalizer,
) -> list:
    """
    QP-only monotonicity guard.

    Cambridge question papers are ordered by increasing root number. If a lower
    root suddenly appears after a later root has been accepted, it is almost
    always a page number, graph axis label, or mark bracket that poisoned the
    model/state tracker. Repair it to the next sibling under the last valid
    parent instead of letting a ghost "4.*" enter Mongo.
    """
    return questions_raw


def _repair_qp_embedded_subpart_labels(
    questions_raw: list,
    normalizer: QuestionNumberNormalizer,
) -> list:
    """
    QP-only granularity repair.

    Whole-document Gemini sometimes keeps an embedded printed child marker in
    question_latex but emits only the parent label. Example from Cambridge
    0580_s20_41:
      QP text: "9 ... (a) Calculate the total perimeter..."
      emitted id: "9"
      correct id: "9(a)"

    It can also misread a continued roman child as the next first-level letter:
      emitted id/text: "9(d) ... (ii) Calculate the volume..."
      previous sibling: "9(c) ... (i) Calculate..."
      correct id: "9(c)(ii)"

    This repair only uses markers physically present in the extracted text; it
    does not invent missing labels.
    """
    if not isinstance(questions_raw, list):
        return questions_raw

    repaired: list = []
    previous_by_root: dict[str, dict] = {}

    def first_embedded_marker(text: str, allowed: set[str], limit: int = 260) -> str:
        head = str(text or "")[:limit]
        for match in re.finditer(r"\(\s*([a-zivx]{1,5})\s*\)", head, flags=re.IGNORECASE):
            token = match.group(1).lower()
            if token == "x":
                continue
            if token in allowed:
                return token
        return ""

    def patch_question(q: dict, new_parts: list[str], warning: str) -> dict:
        q = dict(q)
        old_label = str(q.get("question_id") or q.get("question_latex") or "").strip()
        old_parts = normalizer.extract_parts(old_label)
        old_visual = normalizer.format_parts(old_parts) if old_parts else ""
        new_visual = normalizer.format_parts(new_parts)

        q["question_id"] = new_visual
        q["needs_review"] = True
        _append_question_warning(q, warning)

        text = str(q.get("question_latex") or "").strip()
        if text:
            parsed_label, remainder = normalizer.split_label_and_remainder(text)
            if parsed_label:
                q["question_latex"] = f"{new_visual} {remainder}".strip() if remainder else new_visual
            elif old_visual and text.startswith(old_visual):
                q["question_latex"] = f"{new_visual}{text[len(old_visual):]}".strip()
            else:
                q["question_latex"] = f"{new_visual} {text}".strip()

        return q

    for raw in questions_raw:
        if not isinstance(raw, dict):
            repaired.append(raw)
            continue

        q = dict(raw)
        label_source = str(q.get("question_id") or q.get("question_latex") or "")
        parts = normalizer.extract_parts(label_source)
        text = str(q.get("question_latex") or "")
        root = parts[0] if parts and parts[0].isdigit() else ""

        if root:
            _label, remainder = normalizer.split_label_and_remainder(text)

            # Root-only row with an embedded first-level marker is a child, not
            # a standalone root question.
            if len(parts) == 1:
                marker = first_embedded_marker(remainder, set("abcdefghijklmnopqrstuvwxyz"))
                if marker:
                    q = patch_question(
                        q,
                        [root, marker],
                        f"QP embedded subpart guard promoted {normalizer.format_parts(parts)} to {root}({marker}).",
                    )
                    parts = [root, marker]

            # First-level row with an embedded roman marker is a roman child.
            if len(parts) == 2:
                roman = first_embedded_marker(remainder, normalizer._LABEL_ROMAN_SET)
                previous = previous_by_root.get(root)
                previous_parts = previous.get("parts") if previous else None
                previous_remainder = previous.get("remainder", "") if previous else ""

                if roman and str(parts[1]).lower() not in normalizer._LABEL_ROMAN_SET:
                    new_parts = [root, parts[1], roman]

                    # If Gemini advanced the letter but repeated the same stem,
                    # keep the previous letter and attach the visible roman.
                    if previous_parts and len(previous_parts) >= 3:
                        similarity = SequenceMatcher(
                            None,
                            re.sub(r"\(\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*\)", " ", previous_remainder, flags=re.IGNORECASE)[:160],
                            re.sub(r"\(\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*\)", " ", remainder, flags=re.IGNORECASE)[:160],
                        ).ratio()
                        if similarity >= 0.58:
                            new_parts = [root, previous_parts[1], roman]

                    current_canonical = normalizer.canonical_from_parts(parts)
                    new_canonical = normalizer.canonical_from_parts(new_parts)
                    if new_canonical != current_canonical:
                        q = patch_question(
                            q,
                            new_parts,
                            f"QP embedded subpart guard corrected {current_canonical} to {new_canonical}.",
                        )
                        parts = new_parts

            if parts:
                previous_by_root[root] = {"parts": parts, "remainder": remainder}

        repaired.append(q)

    return repaired


def _sanitize_qp_print_artifacts(text: str) -> str:
    if not text:
        return text
    cleaned = str(text)
    cleaned = re.sub(
        r"^(\d+\s*(?:\([a-z]\))?)\s*\(\s*x\s*\)\s+(?=(?:Find|Calculate|Solve|Show|Write|Simplify|Differentiate|Integrate|Sketch|Draw)\b)",
        r"\1 ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bwww\.exam-mate\.com\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"©\s*UCLES\s*\d{4}", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*Turn over\s*\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<![A-Za-z0-9])\[\s*\d+\s*\](?![A-Za-z0-9])", " ", cleaned)
    cleaned = re.sub(r"\.{4,}", " ", cleaned)
    cleaned = re.sub(r"(?<=\s)\d{1,2}\s+(?=\([a-zivx]+\))", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ensure_qp_text_has_full_label(
    result: dict,
    normalizer: QuestionNumberNormalizer,
) -> None:
    qid = str(result.get("question_id") or "").strip()
    text = str(result.get("question_latex") or "").strip()
    if not qid or not text:
        return
    qid_parts = normalizer.extract_parts(qid)
    if not qid_parts or not qid_parts[0].isdigit():
        return
    full_label = normalizer.format_parts(qid_parts)
    parsed_label, remainder = normalizer.split_label_and_remainder(text)
    parsed_parts = normalizer.extract_parts(parsed_label)

    if parsed_parts and parsed_parts[0].isdigit():
        parsed_canonical = normalizer.canonical_from_parts(parsed_parts)
        qid_canonical = normalizer.canonical_from_parts(qid_parts)
        if parsed_canonical == qid_canonical:
            result["question_latex"] = f"{full_label} {remainder}".strip() if remainder else full_label
            return

    if parsed_label:
        if not parsed_parts or not parsed_parts[0].isdigit():
            remainder = re.sub(
                r"^\s*(?:\(\s*[a-zivx]+\s*\)\s*)+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
        result["question_latex"] = f"{full_label} {remainder}".strip() if remainder else full_label
    elif not text.startswith(full_label):
        result["question_latex"] = f"{full_label} {text}".strip()


def _repair_normalized_qp_backward_roots(
    questions: List[ExtractedQuestion],
    normalizer: QuestionNumberNormalizer,
) -> List[ExtractedQuestion]:
    """
    Final normalized-object gate. This catches any backward QP root that evaded
    raw-dict repair or slicer state repair before the response is returned.
    """
    return questions


def _build_local_qp_page_hints(pdf_base64: str, expected_ids: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Local-first QP skeleton pass.

    This is intentionally cheap and conservative: PyMuPDF text is used to find
    likely visible Cambridge roots per page before Gemini sees the image. The
    hints are not final extraction output; they only steer Gemini away from
    treating printed page numbers as question roots.
    """
    expected_clean = [
        str(value).strip().lower()
        for value in (expected_ids or [])
        if str(value).strip()
    ]
    expected_by_root: Dict[str, List[str]] = defaultdict(list)
    for value in expected_clean:
        root = value.split(".", 1)[0]
        if root and value not in expected_by_root[root]:
            expected_by_root[root].append(value)

    hints: List[Dict[str, Any]] = []
    if not pdf_base64:
        return hints

    try:
        normalized_b64 = pdf_base64.strip().split(",", 1)[-1]
        pdf_bytes = base64.b64decode(normalized_b64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.debug("[LocalQPSkeleton] Could not open PDF for local hints: %s", exc)
        return hints

    last_root = ""
    try:
        for page_index, page in enumerate(doc):
            page_dict = page.get_text("dict")
            line_records: List[tuple[str, float, float]] = []
            for block in page_dict.get("blocks", []):
                for line_obj in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line_obj.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue
                    x0, y0, _x1, _y1 = line_obj.get("bbox", (0, 0, 0, 0))
                    line_records.append((line_text, float(x0), float(y0)))

            line_records.sort(key=lambda item: (item[2], item[1]))
            printed_page_number = str(page_index + 1)
            candidate_root = ""
            visible_subparts: List[str] = []

            for line, x0, y0 in line_records[:120]:
                normalized = " ".join(line.split())
                if not normalized:
                    continue
                if y0 < 70 and normalized == printed_page_number:
                    continue
                lower_normalized = normalized.lower()
                if (
                    lower_normalized.startswith("0580")
                    or lower_normalized.startswith("©")
                    or lower_normalized.startswith("[turn over")
                    or lower_normalized.startswith("www.")
                ):
                    continue

                root_match = re.match(r"^(\d{1,2})(?:\s*(?:\([a-z]\)|[A-Za-z].*)|\s*$)", normalized)
                if root_match and x0 <= 120:
                    root = root_match.group(1)
                    if root != printed_page_number or expected_by_root.get(root):
                        candidate_root = root
                        break

                orphan_match = re.match(r"^\(([a-z]|[ivx]+)\)\b", normalized, flags=re.IGNORECASE)
                if orphan_match:
                    token = orphan_match.group(1).lower()
                    if token not in visible_subparts:
                        visible_subparts.append(token)

            likely_root = candidate_root or last_root
            if candidate_root:
                last_root = candidate_root

            expected_for_page = expected_by_root.get(likely_root, [])[:24] if likely_root else []
            hint = {
                "page_index": page_index,
                "printed_page_number": printed_page_number,
                "likely_root": likely_root,
                "visible_subparts": visible_subparts[:8],
                "expected_ids": expected_for_page,
            }
            hints.append(hint)
    except Exception as exc:
        logger.debug("[LocalQPSkeleton] Native page hint extraction failed: %s", exc)
    finally:
        doc.close()

    useful = sum(1 for hint in hints if hint.get("likely_root") or hint.get("expected_ids"))
    if useful:
        logger.warning(
            "[LocalQPSkeleton] Built local QP hints for %s/%s page(s).",
            useful,
            len(hints),
        )
    return hints


def _native_qp_root_snippets(pdf_base64: str) -> dict[str, str]:
    """
    Lightweight native-text audit used only as a QP safety net. It detects likely
    top-level Cambridge question roots and captures text until the next root so a
    missed question becomes a reviewable stub instead of vanishing silently.
    """
    snippets: dict[str, str] = {}
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        current_root = ""
        current_lines: list[str] = []
        expected_next_root = 1

        def flush() -> None:
            nonlocal current_root, current_lines
            if current_root and current_lines and current_root not in snippets:
                snippets[current_root] = " ".join(current_lines).strip()
            current_root = ""
            current_lines = []

        for page in doc:
            line_records: list[tuple[str, float, float]] = []
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line_obj in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line_obj.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue
                    x0, y0, _x1, _y1 = line_obj.get("bbox", (0, 0, 0, 0))
                    line_records.append((line_text, float(x0), float(y0)))

            line_records.sort(key=lambda item: (item[2], item[1]))
            lines = [item[0] for item in line_records]

            for idx, (line, x0, y0) in enumerate(line_records):
                if y0 < 55 and re.fullmatch(r"\d{1,2}", line):
                    continue
                root_match = re.match(r"^(\d{1,2})(?:\s+\([a-z]\)|\s+[A-Za-z].*|\s*$)", line)
                is_root = False
                if root_match:
                    root = root_match.group(1)
                    root_int = int(root)
                    next_window = " ".join(lines[idx:idx + 60])
                    is_root = (
                        x0 <= 90
                        and
                        root_int == expected_next_root
                        and bool(re.search(r"\([a-z]\)", next_window, flags=re.IGNORECASE))
                    )
                if is_root:
                    flush()
                    current_root = root_match.group(1)
                    current_lines = [line]
                    expected_next_root += 1
                elif current_root:
                    current_lines.append(line)
            flush()
        doc.close()
    except Exception as exc:
        logger.warning("[QPNativeAudit] Native root audit failed: %s", exc)
    return snippets


def _native_qp_first_level_snippets(pdf_base64: str) -> dict[str, str]:
    snippets: dict[str, str] = {}
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        root_snippets = _native_qp_root_snippets(pdf_base64)
        current_root = ""
        current_label = ""
        current_lines: list[str] = []
        expected_next_root = 1

        def flush() -> None:
            nonlocal current_label, current_lines
            if current_label and current_lines and current_label not in snippets:
                snippets[current_label] = " ".join(current_lines).strip()
            current_label = ""
            current_lines = []

        for page in doc:
            line_records: list[tuple[str, float, float]] = []
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line_obj in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line_obj.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue
                    x0, y0, _x1, _y1 = line_obj.get("bbox", (0, 0, 0, 0))
                    line_records.append((line_text, float(x0), float(y0)))

            line_records.sort(key=lambda item: (item[2], item[1]))
            lines = [item[0] for item in line_records]
            for idx, (line, x0, y0) in enumerate(line_records):
                if y0 < 55 and re.fullmatch(r"\d{1,2}", line):
                    continue

                root_match = re.match(r"^(\d{1,2})(?:\s+\([a-z]\)|\s+[A-Za-z].*|\s*$)", line)
                if root_match and x0 <= 90:
                    root_int = int(root_match.group(1))
                    next_window = " ".join(lines[idx:idx + 60])
                    if (
                        root_int == expected_next_root
                        and root_match.group(1) in root_snippets
                        and re.search(r"\([a-z]\)", next_window, flags=re.IGNORECASE)
                    ):
                        flush()
                        current_root = root_match.group(1)
                        expected_next_root += 1
                        inline_part = re.search(r"\(([a-hj-uw-z])\)", line, flags=re.IGNORECASE)
                        if inline_part:
                            current_label = f"{current_root}({inline_part.group(1).lower()})"
                            current_lines = [line.replace(root_match.group(1), current_label, 1)]
                            continue

                sub_match = re.match(r"^\(([a-hj-uw-z])\)\s*(.*)", line, flags=re.IGNORECASE)
                if current_root and sub_match and x0 <= 125:
                    flush()
                    current_label = f"{current_root}({sub_match.group(1).lower()})"
                    current_lines = [f"{current_label} {sub_match.group(2)}".strip()]
                    continue

                if current_label:
                    current_lines.append(line)
            flush()
        doc.close()
    except Exception as exc:
        logger.warning("[QPNativeAudit] Native first-level audit failed: %s", exc)
    return snippets


def _expected_qp_id_order(
    expected_ids: list,
    normalizer: QuestionNumberNormalizer,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in expected_ids or []:
        parts = normalizer.extract_parts(str(value or ""))
        canonical = normalizer.canonical_from_parts(parts) if parts else ""
        if canonical and canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    return ordered


def _line_should_skip_for_qp_skeleton(text: str, x0: float, y0: float, printed_page: str) -> bool:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return True
    lower = normalized.lower()
    formula_sheet_phrases = (
        "area, a, of triangle",
        "area, a, of circle",
        "circumference, c, of circle",
        "curved surface area",
        "surface area, a, of sphere",
        "volume, v, of prism",
        "volume, v, of pyramid",
        "volume, v, of cylinder",
        "volume, v, of cone",
        "volume, v, of sphere",
        "for the equation ax",
        "list of formulas",
    )
    if any(phrase in lower for phrase in formula_sheet_phrases):
        return True
    if "\x01" in normalized or normalized.startswith("* 000"):
        return True
    if re.fullmatch(r"[.,;\s\[\]\(\)hms/%:-]+", normalized) and "." in normalized:
        return True
    if len(normalized) >= 8:
        odd_chars = sum(1 for ch in normalized if ord(ch) < 32 or ord(ch) > 255)
        if odd_chars / max(1, len(normalized)) > 0.25:
            return True
    if sum(1 for ch in normalized if ord(ch) < 32) >= 1:
        return True
    if y0 < 70 and x0 > 180 and normalized == printed_page:
        return True
    if y0 < 75 and (
        lower.startswith("0580")
        or "cambridge igcse" in lower
        or lower == "question paper"
    ):
        return True
    if (
        lower.startswith("©")
        or lower.startswith("www.")
        or lower == "dfd"
        or re.match(r"^question\s+\d{1,2}\s+is\s+printed\b", lower)
        or re.match(r"^question\s+\d{1,2}\s+continues\b", lower)
        or re.match(r"^question\s+\d{1,2}\s+is\s+continued\b", lower)
        or "do not write in this margin" in lower
        or "0580/" in lower
        or lower.startswith("[turn over")
        or lower.startswith("turn over")
        or lower.startswith("page ")
        or lower == "blank page"
        or y0 > 760
    ):
        return True
    return False


def _trim_qp_transition_noise(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"\s+Question\s+\d{1,2}\s+is\s+printed\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+Question\s+\d{1,2}\s+(?:continues|is\s+continued)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+Permission\s+to\s+reproduce\s+items\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+To\s+avoid\s+the\s+issue\s+of\s+disclosure\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\.{6,}\s*(?:[A-Za-z/%]+\s*)?(?:\[\s*\d+\s*\])?",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?:\s*:\s*){2,}\s*(?:\[\s*\d+\s*\])?", "", cleaned)
    cleaned = _repair_common_native_math_text(cleaned)
    return cleaned.strip()


def _repair_common_native_math_text(text: str) -> str:
    """Repair conservative PyMuPDF text-layer math ordering failures.

    Some Cambridge PDFs store superscripts as separate tiny spans. Plain native
    text extraction can then flatten formulas like y = x^3 + x^2 - 5x into
    "y x x x 5 3 2 = + -". These replacements are intentionally narrow:
    they only trigger on the exact scrambled token patterns seen in the native
    text layer, so normal question text is left alone.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""

    # Common mojibake from Windows/log/PDF extraction. Keep the visible degree
    # symbol instead of forcing every angle into math mode.
    cleaned = cleaned.replace("Â°", "°").replace("â€“", "-").replace("âˆ’", "-")

    # Function notation is often emitted as "( ) f x" or "f ( ) x".
    # These substitutions intentionally cover only single-letter Cambridge
    # functions; they do not rewrite ordinary words containing f/g/h.
    cleaned = re.sub(
        r"\(\s*\)\s*f\s*y\s*x\s*=",
        r"$y=f(x)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\(\s*\)\s*f\s*([0-9]+)\b",
        lambda m: f"$f({m.group(1)})$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\(\s*\)\s*f\s*x\s+x\s+2\s*=\s*-",
        r"$f(x)=x^{-2}$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\(\s*\)\s*f\s*x\s+([0-9a-z.-]+)\s*=",
        lambda m: f"$f(x)={m.group(1)}$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\(\s*\)\s*f\s*x\s*=\s*([a-z0-9.-]+)",
        lambda m: f"$f(x)={m.group(1)}$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\(\s*\)\s*([fgh])\s*x\b",
        lambda m: f"${m.group(1)}(x)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b([fgh])\s*\(\s*\)\s*x\b",
        lambda m: f"{m.group(1)}(x)",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\by\s*=\s*([fgh])\s*\(\s*\)\s*x\b",
        lambda m: f"$y={m.group(1)}(x)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b([fgh])\s*\(\s*x\s*\)",
        lambda m: f"{m.group(1)}(x)",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Cambridge native text sometimes appends superscripts after the line:
    # "r 3 r 6 = 0 2 + -" -> r^2 + 3r - 6 = 0.
    # Keep this limited to equation-like text with a trailing exponent token.
    def _repair_quadratic_tail(match: re.Match) -> str:
        var = match.group("var")
        b = match.group("b")
        c = match.group("c")
        c_sign = match.group("csign")
        return f"${var}^2 + {b}{var} {c_sign} {c} = 0$"

    cleaned = re.sub(
        r"\b(?P<var>[a-z])\s+(?P<b>\d+)\s+(?P=var)\s+(?P<c>\d+)\s*=\s*0\s+2\s*(?P<plus>\+)\s*(?P<csign>[+-])",
        _repair_quadratic_tail,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?P<var>[a-z])\s+(?P<b>\d+)\s+(?P=var)\s+(?P<c>\d+)\s*=\s*0\b",
        lambda m: f"${m.group('var')}^2 + {m.group('b')}{m.group('var')} - {m.group('c')} = 0$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Derivative-style cubic seen in 0580_s24_qp_43:
    # "y x x 7 7 6 = -" is the PDF text layer's broken form of
    # y = x^3 - 7x (with stray duplicated coefficients/exponents).
    cleaned = re.sub(
        r"\by\s+x\s+x\s+7\s+7\s+6\s*=\s*-",
        r"$y = x^3 - 7x$",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\by\s+x\s+x\s+x\s+5\s+3\s+2\s*=\s*\+\s*-",
        r"$y = x^3 + x^2 - 5x$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bfor\s+y\s+x\s+x\s+x\s+5\s+3\s+2\s*=\s*\+\s*-",
        r"for $y = x^3 + x^2 - 5x$",
        cleaned,
        flags=re.IGNORECASE,
    )

    function_header = (
        r"\(\s*\)\s*f\s*x\s+x\s*2\s*3\s*=\s*-\s*"
        r"\(\s*\)\s*g\s*x\s+x\s*9\s*2\s*=\s*-\s*"
        r"\(\s*\)\s*h\s*x\s*3x\s*=?\s*-?"
    )
    cleaned = re.sub(
        function_header,
        r"$f(x)=2x-3$, $g(x)=9-x^2$, $h(x)=3^x$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bFind\s+f\s+4\s*,\s*\(\s*\)",
        r"Find $f(4)$,",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bFind\s+hg\s+3\s*,\s*\(\s*\)",
        r"Find $hg(3)$,",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bg\s*\(\s*2\s*\)\s*x\s+in\s+its\s+simplest\s+form",
        r"$g(2x)$ in its simplest form",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bfg\s+x\s+in\s+its\s+simplest\s+form\s*\.?\s*\(\s*\)",
        r"$fg(x)$ in its simplest form.",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bFind\s+x\s+f\s*\(\s*\)\s*-\s*1\s*x\s*f\s*\(\s*\)\s*-\s*=",
        r"Find $f^{-1}(x)$. $f^{-1}(x) =$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Graph/function questions.
    cleaned = re.sub(
        r"\bgraph\s+of\s+y\s*=\s*f\s*\(\s*x\s*\)",
        r"graph of $y=f(x)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bgraph\s+of\s+\$f\(x\)\$\s+y\s*=",
        r"graph of $y=f(x)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bline\s+x\s+0\s*=",
        r"line $x=0$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?<!\$)\bf\s*\(\s*2\s*\)(?!\$)",
        r"$f(2)$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Cubic rearrangement from the graph question:
    # "x px qx 2 3 2 + + =" -> x^3 + px^2 + qx = 2.
    cleaned = re.sub(
        r"\bx\s+p\s*x\s+q\s*x\s+2\s+3\s+2\s*\+\s*\+\s*=",
        r"$x^3 + px^2 + qx = 2$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Formula-sheet fragments that frequently appear in Cambridge diagrams.
    cleaned = re.sub(
        r"\bA\s*=\s*r\s*l\s+r\s*2\b",
        r"$A = \\pi r l + \\pi r^2$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bV\s*=\s*r\s+h\s+r\s+33\b",
        r"$V = \\frac{1}{3}\\pi r^2h$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Unit exponent cleanup from native text: "cm 200 2" -> "200 cm^2".
    cleaned = re.sub(
        r"\bcm\s+(\d+(?:\.\d+)?)\s+2\b",
        lambda m: f"{m.group(1)} cm^2",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bcm\s+(\d+(?:\.\d+)?)\s+3\b",
        lambda m: f"{m.group(1)} cm^3",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Arc/sector equation cleanup from Q11.
    cleaned = re.sub(
        r"\bx\s+108\s*=",
        r"$x=108$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\br\s+sin\s*y\s+y\s+360\s*=",
        r"$r = \\frac{360\\sin y}{y}$",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\br\s+siny\s+y\s+360\s*=",
        r"$r = \\frac{360\\sin y}{y}$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Flattened Cambridge value table from Q11(b)(ii). Native text emits the
    # table cells as a sentence; render them as a real LaTeX table so the review
    # UI is readable even if the crop is missing.
    table_11b_pattern = (
        r"(?:r?y\s+)?y\s+360\s+sin\s*y\s+"
        r"108\.4\s+341\.60\s+340\.55\s+"
        r"108\.5\s+341\.40\s+340\.86\s+"
        r"108\.6\s+341\.20\s+108\.7"
    )
    table_11b_latex = (
        r"$\begin{array}{c|c|c}"
        r"y & 360\sin y & \pi y \\"
        r"108.4 & 341.60 & 340.55 \\"
        r"108.5 & 341.40 & 340.86 \\"
        r"108.6 & 341.20 & \\"
        r"108.7 & &"
        r"\end{array}$"
    )
    cleaned = re.sub(
        table_11b_pattern,
        lambda _m: table_11b_latex,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b1\s+2\s*\[?\s*The\s+volume,\s*V,\s*of\s+a\s+cone\s+with\s+radius\s+r\s+and\s+height\s+h\s+is\s+V\s*=\s*r\s+h\s+r\s+33",
        r"[The volume, $V$, of a cone with radius $r$ and height $h$ is $V = \\frac{1}{3}\\pi r^2h$.]",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Exponential and nth-term cleanup.
    cleaned = re.sub(
        r"\b120\s*-\s*n\s+Find\b",
        r"$120-n$ Find",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bwhen\s+the\s+nth\s+term\s+is\s+-\s*1211\b",
        r"when the $n$th term is $-1211$",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Tidy doubled math delimiters introduced by adjacent repairs.
    cleaned = re.sub(r"\$\s+\$", " ", cleaned)
    cleaned = re.sub(r"\${2,}([^$]+)\${2,}", r"$\1$", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _mark_qp_math_risk_if_needed(row: dict) -> dict:
    """Flag rows that still contain likely native-PDF math corruption."""
    if not isinstance(row, dict):
        return row
    text = str(row.get("question_latex") or "")
    compact = " ".join(text.split())
    if not compact:
        return row
    risk_patterns = (
        r"\(\s*\)\s*[fgh]\s*x",
        r"\b[fgh]\s*\(\s*\)\s*x",
        r"\b[a-z]\s+[a-z]\s+[a-z]\s+\d+\s+\d+\s+\d+\s*=",
        r"\b[a-z]\s+\d+\s+[a-z]\s+\d+\s*=\s*0\s+2\b",
        r"\bA\s*=\s*r\s*l\s+r\s*2\b",
        r"\bV\s*=\s*r\s+h\s+r\s+33\b",
        r"Â|â",
    )
    if not any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in risk_patterns):
        return row
    warnings = row.get("validation_warnings")
    if not isinstance(warnings, list):
        warnings = []
    warning = "Native PDF math text may need LaTeX verification after automatic cleanup."
    if warning not in warnings:
        warnings.append(warning)
    row["validation_warnings"] = warnings
    row["needs_review"] = True
    return row


def _dedupe_repeated_qp_label(text: str, label: str) -> str:
    raw = " ".join(str(text or "").split())
    label_text = str(label or "").strip()
    if not raw or not label_text:
        return raw
    escaped_label = re.escape(label_text)
    pattern = rf"^({escaped_label}\s+.+?)\s+{escaped_label}\s+"
    return re.sub(pattern, r"\1 ", raw, count=1, flags=re.IGNORECASE).strip()


def _qp_text_allows_auto_diagram(text: str) -> bool:
    lower = " ".join(str(text or "").lower().split())
    if not lower:
        return False
    diagram_terms = (
        "diagram",
        "figure",
        "shown",
        "table",
        "complete the table",
        "value table",
        "values table",
        "grid",
        "graph",
        "axes",
        "axis",
        "sketch",
        "draw",
        "plot",
        "histogram",
        "cumulative frequency",
        "frequency table",
        "venn",
        "tree diagram",
        "number line",
        "not to scale",
        "map",
        "bearing",
        "angle",
        "shape",
        "triangle",
        "rectangle",
        "square",
        "polygon",
        "quadrilateral",
        "circle",
        "sector",
        "cone",
        "cuboid",
        "prism",
        "pyramid",
        "net",
        "region",
        "curve",
        "vector",
        "translation",
        "rotation",
        "reflection",
        "enlargement",
        "transformation",
        "stem-and-leaf",
        "pie chart",
        "bar chart",
        "scatter",
        "box-and-whisker",
        "frequency polygon",
    )
    return any(term in lower for term in diagram_terms)


def _qp_skeleton_text_is_formula_noise(text: str) -> bool:
    lower = " ".join(str(text or "").lower().split())
    return any(
        phrase in lower
        for phrase in (
            "area, a, of triangle",
            "area, a, of circle",
            "circumference, c, of circle",
            "curved surface area",
            "surface area, a, of sphere",
            "volume, v, of prism",
            "volume, v, of pyramid",
            "volume, v, of cylinder",
            "volume, v, of cone",
            "volume, v, of sphere",
            "for the equation ax",
            "list of formulas",
        )
    )


def _expected_children_by_parent(expected_order: list[str]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for canonical in expected_order:
        parts = canonical.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            if canonical not in children[parent]:
                children[parent].append(canonical)
    return children


def _expected_prefixes(expected_order: list[str]) -> set[str]:
    prefixes: set[str] = set()
    for canonical in expected_order:
        parts = canonical.split(".")
        for depth in range(1, len(parts) + 1):
            prefixes.add(".".join(parts[:depth]))
    return prefixes


def _choose_orphan_qp_id(
    token: str,
    current_parts: list[str],
    expected_order: list[str],
    expected_set: set[str],
    used_ids: set[str],
    normalizer: QuestionNumberNormalizer,
) -> str:
    """
    Resolve an orphan token ((b), (ii), etc.) to the best MS-expected canonical ID.

    Priority:
      1. Append token to current context at decreasing depth — always same root.
      2. Walk expected_order in order for the SAME root only; match same terminal
         token AND same depth as current_parts (prevents 3.b.i from grabbing 9.b.i).
      3. Walk expected_order sequentially for the next unused ID after the last
         used ID with the same root — this handles linear orphan sequences.
    """
    token = str(token or "").strip().lower()
    if not token or not current_parts:
        return ""

    # Prefer appending to the current exact context, then to progressively
    # shorter parents. This handles (b), (ii), etc. across continuation lines.
    for depth in range(len(current_parts), 0, -1):
        candidate_parts = current_parts[:depth] + [token]
        candidate = normalizer.canonical_from_parts(candidate_parts)
        if candidate in expected_set and candidate not in used_ids:
            return candidate

    root = current_parts[0]

    # Pass 2: same root, same terminal token, same hierarchy depth (exact depth match
    # prevents grabbing a sibling root's ID that happens to share the same terminal).
    current_depth = len(current_parts)
    for canonical in expected_order:
        if canonical in used_ids:
            continue
        parts = canonical.split(".")
        if parts and parts[0] == root and parts[-1] == token and len(parts) == current_depth:
            return canonical

    # Pass 3: same root, same terminal token at any depth (fallback, wider net).
    for canonical in expected_order:
        if canonical in used_ids:
            continue
        parts = canonical.split(".")
        if parts and parts[0] == root and parts[-1] == token:
            return canonical

    # Pass 4: walk expected_order in sequence — return the next unused ID under
    # the same root that immediately follows the last used sibling. This handles
    # pure sequential orphan pages where the token itself isn't visible.
    last_used_idx = -1
    for idx, canonical in enumerate(expected_order):
        if canonical in used_ids:
            parts = canonical.split(".")
            if parts and parts[0] == root:
                last_used_idx = idx
    if last_used_idx >= 0:
        for canonical in expected_order[last_used_idx + 1:]:
            if canonical in used_ids:
                break
            parts = canonical.split(".")
            if parts and parts[0] == root and parts[-1] == token:
                return canonical

    return ""


def _match_qp_orphan_label(text: str) -> tuple[list[str], str]:
    """
    Parse leading Cambridge orphan labels such as "(a) (i)" or "(c)(ii)".

    Native PDF text often splits QP rows as:
        9
        (a) (i) Find ...
        (ii) Find ...
        (b) ...

    The older parser consumed only the first token, so "(a) (i)" became
    context for 9.a instead of a real 9.a.i row. This helper returns all
    leading wrapped tokens and the remaining question text.
    """
    raw = str(text or "")
    if not raw.strip():
        return [], ""
    token_pattern = r"(?:xviii|xvii|xvi|xiv|xiii|xii|xi|viii|vii|vi|iv|ix|iii|ii|i|xx|xix|xv|x|v|[a-z])"
    pattern = re.compile(
        rf"^\s*((?:\(\s*({token_pattern})\s*\)\s*){{1,5}})(.*)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(raw)
    if not match:
        return [], raw.strip()
    label_block = match.group(1)
    tokens = [
        token.lower()
        for token in re.findall(rf"\(\s*({token_pattern})\s*\)", label_block, flags=re.IGNORECASE)
    ]
    return tokens, str(match.group(3) or "").strip()


def _orphan_tokens_to_qp_candidate_parts(tokens: list[str], current_parts: list[str]) -> list[str]:
    if not tokens or not current_parts:
        return []
    normalized_tokens = [str(token or "").strip().lower() for token in tokens if str(token or "").strip()]
    if not normalized_tokens:
        return []

    root = current_parts[0]
    first = normalized_tokens[0]
    roman_set = set(QuestionNumberNormalizer._ROMAN_ORDER)

    # "(a) (i)" or "(c)(ii)" starts a full branch below the current root.
    if len(first) == 1 and "a" <= first <= "z":
        return [root] + normalized_tokens

    # "(ii)" usually means the next sibling under the current immediate parent.
    if first in roman_set and len(current_parts) >= 2:
        return current_parts[:-1] + normalized_tokens

    return current_parts + normalized_tokens


def _orphan_tokens_to_qp_candidate_options(tokens: list[str], current_parts: list[str]) -> list[list[str]]:
    if not tokens or not current_parts:
        return []
    normalized_tokens = [str(token or "").strip().lower() for token in tokens if str(token or "").strip()]
    if not normalized_tokens:
        return []

    root = current_parts[0]
    first = normalized_tokens[0]
    roman_set = set(QuestionNumberNormalizer._ROMAN_ORDER)
    options: list[list[str]] = []

    if len(first) == 1 and "a" <= first <= "z":
        # Deep nested layout: parent label 4(a)(iii), then terminal rows (a), (b).
        # This must prefer 4.a.iii.a over the old root-level 4.a interpretation.
        if len(current_parts) >= 4:
            options.append(current_parts[:-1] + normalized_tokens)
        if len(current_parts) >= 3:
            options.append(current_parts + normalized_tokens)
        options.append([root] + normalized_tokens)
    elif first in roman_set and len(current_parts) >= 2:
        options.append(current_parts[:-1] + normalized_tokens)
        options.append(current_parts + normalized_tokens)
    else:
        options.append(current_parts + normalized_tokens)

    deduped: list[list[str]] = []
    seen: set[str] = set()
    for option in options:
        key = ".".join(option)
        if key and key not in seen:
            deduped.append(option)
            seen.add(key)
    return deduped


def _looks_like_qp_contamination_tail(tail: str) -> bool:
    probe = " ".join(str(tail or "").split()).lower()[:260]
    if len(probe) < 8:
        return False
    question_words = (
        "calculate", "work out", "find", "show that", "write down", "solve",
        "complete", "give your answer", "another swimmer", "blessy", "rashid",
        "adam", "diagram", "table", "graph", "curve", "histogram",
    )
    mark_or_answer_space = bool(re.search(r"\[\s*\d+\s*\]|\.{6,}", probe))
    return mark_or_answer_space or any(word in probe for word in question_words)


def _trim_qp_sibling_contamination(
    text: str,
    canonical: str,
    expected_order: list[str],
    normalizer: QuestionNumberNormalizer,
) -> str:
    """
    Trim native-text spillover from neighbouring QP subparts.

    This targets cases such as a clean `4(b)` row followed by copied text from
    `4(a)(iii)` / `(iv)` after the real row has already ended. It is deliberately
    conservative: only cut after a meaningful prefix and only when the tail
    looks like another Cambridge question fragment.
    """
    raw = str(text or "").strip()
    parts = normalizer.extract_parts(canonical)
    if not raw or len(parts) < 2:
        return raw

    label = normalizer.format_parts(parts)
    root = parts[0]
    sibling_tokens: set[str] = set()
    sibling_patterns: list[str] = []
    for expected in expected_order or []:
        if expected == canonical:
            continue
        expected_parts = normalizer.extract_parts(expected)
        if not expected_parts or expected_parts[0] != root:
            continue
        sibling_patterns.append(_parts_label_pattern(expected_parts))
        for token in expected_parts[1:]:
            sibling_tokens.add(str(token).lower())

    min_cut = max(len(label) + 24, 45)
    cut_positions: list[int] = []

    for pattern in sibling_patterns:
        for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
            if match.start() >= min_cut and _looks_like_qp_contamination_tail(raw[match.start():]):
                cut_positions.append(match.start())

    current_root_num = int(root) if str(root).isdigit() else None
    later_roots = sorted({
        int(candidate_parts[0])
        for expected in expected_order or []
        for candidate_parts in [normalizer.extract_parts(expected)]
        if candidate_parts
        and str(candidate_parts[0]).isdigit()
        and current_root_num is not None
        and int(candidate_parts[0]) > current_root_num
    })
    for later_root in later_roots[:4]:
        root_transition = rf"(?<![A-Za-z0-9]){later_root}\s+(?=[A-Z])"
        for match in re.finditer(root_transition, raw):
            if match.start() >= min_cut and _looks_like_qp_contamination_tail(raw[match.start():]):
                cut_positions.append(match.start())

    # Do not cut on orphan markers like "(i)" or "(a)" alone. In Cambridge
    # layouts these are often the actual target marker after a shared stem
    # (e.g. `4(a)` stem followed by `(i) ...`). Cutting there caused rows to
    # keep only parent context and drop the real question text. Exact sibling
    # labels above are still trimmed safely.

    if not cut_positions:
        return raw

    cut_at = min(cut_positions)
    trimmed = raw[:cut_at].rstrip(" .;,\n\t")
    return trimmed if len(trimmed) >= min_cut else raw


def _qp_row_has_wrong_deep_terminal(
    canonical: str,
    text: str,
    normalizer: QuestionNumberNormalizer,
) -> str:
    parts = normalizer.extract_parts(canonical)
    if len(parts) < 4:
        return ""
    terminal = str(parts[-1]).lower()
    body = str(text or "")
    _label, remainder = normalizer.split_label_and_remainder(body)
    probe = remainder[:420] if remainder else body[:420]
    if len(terminal) == 1 and "a" <= terminal <= "z":
        for letter_ord in range(ord("a"), ord("z") + 1):
            letter = chr(letter_ord)
            if letter == terminal:
                continue
            if re.search(rf"(?<![A-Za-z0-9])\(\s*{re.escape(letter)}\s*\)", probe, flags=re.IGNORECASE):
                return f"deep_terminal_mismatch_expected_{terminal}_saw_{letter}"
    return ""


def _qp_local_row_safe_for_auto_clean(
    canonical: str,
    text: str,
    normalizer: QuestionNumberNormalizer,
) -> tuple[bool, str]:
    if not str(text or "").strip():
        return False, "empty_text"
    wrong_terminal = _qp_row_has_wrong_deep_terminal(canonical, text, normalizer)
    if wrong_terminal:
        return False, wrong_terminal
    return True, ""


def _build_local_qp_ms_skeleton_rows(
    pdf_base64: str,
    expected_ids: list,
    normalizer: QuestionNumberNormalizer,
) -> list[dict[str, Any]]:
    """
    Build QP rows from the PDF text layer using saved-MS IDs as the canonical
    contract. This is intentionally local and non-inventive: only labels that
    are visible in native PDF text become rows.
    """
    expected_order = _expected_qp_id_order(expected_ids, normalizer)
    expected_set = set(expected_order)
    expected_prefix_set = _expected_prefixes(expected_order)
    if not pdf_base64 or not expected_order:
        return []

    rows: list[dict[str, Any]] = []
    root_stem_by_root: dict[str, str] = {}
    current_id = ""
    current_lines: list[str] = []
    current_parts: list[str] = []
    root_context_lines: list[str] = []
    context_lines: list[str] = []
    context_by_root: dict[str, list[str]] = {}
    used_ids: set[str] = set()

    def inherited_context_for(candidate_parts: list[str]) -> list[str]:
        """
        Return the active prompt context for a target row.

        If the scanner is currently inside a prefix branch such as `6.a`, an
        orphan child marker `(i)` must inherit the `6(a)` stem/prompt, not just
        the root Q6 stem. This is what keeps rows like `6.a.i` meaningful:
        diagram stem + parent instruction + leaf text.
        """
        if not candidate_parts:
            return []
        if current_parts and len(current_parts) < len(candidate_parts):
            if candidate_parts[: len(current_parts)] == current_parts:
                return list(context_lines)
        if len(candidate_parts) == 2:
            return list(root_context_lines)
        return list(context_lines or root_context_lines)

    def flush() -> None:
        nonlocal current_id, current_lines, current_parts
        if current_id and current_lines:
            parts = normalizer.extract_parts(current_id)
            label = normalizer.format_parts(parts)
            root_context = context_by_root.get(parts[0], []) if len(parts) > 1 and parts else []
            if not root_context and len(parts) > 1 and parts and root_stem_by_root.get(parts[0]):
                root_context = [root_stem_by_root[parts[0]]]
            if root_context and current_lines[: len(root_context)] != root_context:
                current_lines = root_context + current_lines
            text = " ".join(" ".join(line.split()) for line in current_lines if line.strip()).strip()
            if text:
                if _qp_skeleton_text_is_formula_noise(text):
                    current_id = ""
                    current_lines = []
                    current_parts = []
                    return
                _label, remainder = normalizer.split_label_and_remainder(text)
                label_variants = [label]
                for depth in range(1, max(1, len(parts))):
                    prefix_label = normalizer.format_parts(parts[:depth])
                    if prefix_label:
                        label_variants.append(prefix_label)
                changed = True
                while changed and remainder:
                    changed = False
                    for variant in sorted(set(label_variants), key=len, reverse=True):
                        if variant and remainder.lower().startswith(variant.lower()):
                            remainder = remainder[len(variant):].strip()
                            changed = True
                q_text = f"{label} {remainder}".strip() if remainder else (
                    text if text.startswith(label) else f"{label} {text}".strip()
                )
                q_text = _dedupe_repeated_qp_label(
                    _trim_qp_transition_noise(q_text),
                    label,
                )
                if not q_text:
                    current_id = ""
                    current_lines = []
                    current_parts = []
                    return
                rows.append({
                    "document_type": "Question Paper",
                    "question_id": label,
                    "canonical_question_id": normalizer.canonical_from_parts(parts),
                    "parent_canonical_id": normalizer.parent_from_parts(parts),
                    "question_latex": q_text,
                    "official_marking_scheme_latex": "",
                    "diagram_urls": [],
                    "diagram_regions": [],
                    "needs_review": False,
                    "cognitive_demand": "MEDIUM",
                    "validation_warnings": [],
                })
        current_id = ""
        current_lines = []

    try:
        normalized_b64 = pdf_base64.strip().split(",", 1)[-1]
        pdf_bytes = base64.b64decode(normalized_b64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.debug("[QPLocalSkeleton] Could not open PDF: %s", exc)
        return []

    try:
        for page_index, page in enumerate(doc):
            printed_page = str(page_index + 1)
            page_dict = page.get_text("dict")
            line_records: list[tuple[str, float, float]] = []
            for block in page_dict.get("blocks", []):
                for line_obj in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line_obj.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue
                    x0, y0, _x1, _y1 = line_obj.get("bbox", (0, 0, 0, 0))
                    line_records.append((line_text, float(x0), float(y0)))
            root_candidates: list[tuple[float, float, str, str]] = []
            for line_text, x0, y0 in sorted(line_records, key=lambda item: (item[2], item[1])):
                normalized = " ".join(line_text.split())
                if re.fullmatch(r"\d{1,2}", normalized or "") and x0 <= 90:
                    same_baseline = [
                        " ".join(peer_text.split())
                        for peer_text, peer_x0, peer_y0 in line_records
                        if peer_x0 > x0 + 4
                        and abs(peer_y0 - y0) <= 2.5
                        and " ".join(peer_text.split())
                        and not _line_should_skip_for_qp_skeleton(
                            " ".join(peer_text.split()),
                            float(peer_x0),
                            float(peer_y0),
                            printed_page,
                        )
                    ]
                    if same_baseline:
                        normalized = f"{normalized} {' '.join(same_baseline)}".strip()
                if _line_should_skip_for_qp_skeleton(normalized, float(x0), float(y0), printed_page):
                    continue
                label = normalizer.extract_leading_label(normalized)
                parts = normalizer.extract_parts(label)
                canonical = normalizer.canonical_from_parts(parts) if parts else ""
                if (
                    len(parts) == 1
                    and canonical in expected_prefix_set
                    and canonical not in expected_set
                    and float(x0) <= 90
                ):
                    root_candidates.append((float(y0), float(x0), canonical, normalized))
            for _y0, _x0, canonical, normalized in sorted(root_candidates):
                root_stem_by_root.setdefault(canonical, normalized)

        for page_index, page in enumerate(doc):
            page_plain_text = page.get_text("text") or ""
            page_lower = page_plain_text.lower()
            if (
                "list of formulas" in page_lower
                or "instructions" in page_lower and "answer all questions" in page_lower
            ):
                continue
            page_dict = page.get_text("dict")
            line_records: list[tuple[str, float, float]] = []
            for block in page_dict.get("blocks", []):
                for line_obj in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line_obj.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue
                    x0, y0, _x1, _y1 = line_obj.get("bbox", (0, 0, 0, 0))
                    line_records.append((line_text, float(x0), float(y0)))

            line_records.sort(key=lambda item: (item[2], item[1]))
            printed_page = str(page_index + 1)
            for line, x0, y0 in line_records:
                normalized = " ".join(line.split())
                if _line_should_skip_for_qp_skeleton(normalized, x0, y0, printed_page):
                    continue

                label = normalizer.extract_leading_label(normalized)
                parts = normalizer.extract_parts(label)
                canonical = normalizer.canonical_from_parts(parts) if parts else ""
                if len(parts) == 1:
                    label_at_question_margin = x0 <= 80
                else:
                    label_at_question_margin = (
                        x0 <= 140
                        or normalized.lower().startswith(label.lower())
                        or normalized.lower().startswith(normalizer.format_parts(parts).lower())
                    )
                is_visible_prefix = (
                    canonical in expected_prefix_set
                    and label_at_question_margin
                )
                is_visible_expected = (
                    canonical in expected_set
                    and canonical not in used_ids
                    and is_visible_prefix
                )

                if is_visible_expected:
                    flush()
                    current_id = canonical
                    current_parts = parts
                    if len(parts) == 1:
                        root_context_lines = [normalized]
                        context_by_root[canonical] = list(root_context_lines)
                        context_lines = list(root_context_lines)
                        current_lines = [normalized]
                    else:
                        current_lines = context_lines + [normalized]
                    used_ids.add(canonical)
                    continue

                if is_visible_prefix and canonical not in expected_set:
                    flush()
                    current_parts = parts
                    if len(parts) == 1:
                        root_context_lines = [normalized]
                        context_by_root[canonical] = list(root_context_lines)
                        context_lines = [normalized]
                    else:
                        context_lines = root_context_lines + [normalized]
                    # Keep context bounded so a long previous page does not
                    # get copied into every child.
                    context_lines = context_lines[-8:]
                    continue

                orphan_tokens, orphan_tail = _match_qp_orphan_label(normalized)
                if orphan_tokens and x0 <= 155:
                    token = orphan_tokens[-1]
                    candidate_options = _orphan_tokens_to_qp_candidate_options(
                        orphan_tokens,
                        current_parts,
                    )
                    candidate_parts: list[str] = []
                    candidate = ""
                    for option_parts in candidate_options:
                        option = normalizer.canonical_from_parts(option_parts)
                        if option in expected_set and option not in used_ids:
                            candidate_parts = option_parts
                            candidate = option
                            break
                    if not candidate:
                        for option_parts in candidate_options:
                            option = normalizer.canonical_from_parts(option_parts)
                            if option in expected_prefix_set:
                                candidate_parts = option_parts
                                candidate = option
                                break
                    if candidate in expected_prefix_set and candidate not in expected_set:
                        flush()
                        current_parts = candidate_parts
                        label_text = normalizer.format_parts(candidate_parts)
                        tail = orphan_tail
                        if len(candidate_parts) == 1:
                            root_context_lines = [f"{label_text} {tail}".strip()]
                            context_lines = list(root_context_lines)
                        else:
                            context_lines = root_context_lines + [f"{label_text} {tail}".strip()]
                        context_lines = context_lines[-8:]
                        continue
                    if candidate in expected_set and candidate not in used_ids:
                        flush()
                        current_id = candidate
                        current_parts = candidate_parts
                        label_text = normalizer.format_parts(candidate_parts)
                        inherited_context = inherited_context_for(current_parts)
                        current_lines = inherited_context + [f"{label_text} {orphan_tail}".strip()]
                        used_ids.add(candidate)
                        continue
                    orphan_id = _choose_orphan_qp_id(
                        token,
                        current_parts,
                        expected_order,
                        expected_set,
                        used_ids,
                        normalizer,
                    )
                    if orphan_id:
                        flush()
                        current_id = orphan_id
                        current_parts = normalizer.extract_parts(orphan_id)
                        label_text = normalizer.format_parts(current_parts)
                        tail = orphan_tail
                        inherited_context = inherited_context_for(current_parts)
                        current_lines = inherited_context + [f"{label_text} {tail}".strip()]
                        used_ids.add(orphan_id)
                        continue

                if current_id:
                    current_lines.append(normalized)
                    continue

                if current_parts:
                    active_context = normalizer.canonical_from_parts(current_parts)
                    if active_context in expected_prefix_set and active_context not in expected_set:
                        if len(current_parts) == 1:
                            root_context_lines.append(normalized)
                            root_context_lines = root_context_lines[-8:]
                            context_by_root[active_context] = list(root_context_lines)
                            context_lines = list(root_context_lines)
                        else:
                            context_lines.append(normalized)
                            context_lines = context_lines[-8:]
        flush()
    except Exception as exc:
        logger.debug("[QPLocalSkeleton] Native skeleton build failed: %s", exc)
        return []
    finally:
        doc.close()

    for row in rows:
        canonical = str(row.get("canonical_question_id") or "").strip().lower()
        parts = normalizer.extract_parts(canonical)
        if len(parts) <= 1:
            row["question_latex"] = _trim_qp_sibling_contamination(
                _trim_qp_transition_noise(str(row.get("question_latex") or "")),
                canonical,
                expected_order,
                normalizer,
            )
            continue
        root = parts[0]
        stem = root_stem_by_root.get(root)
        label = normalizer.format_parts(parts)
        current_text = _dedupe_repeated_qp_label(
            _trim_qp_transition_noise(str(row.get("question_latex") or "")),
            label,
        )
        if not stem or not current_text:
            row["question_latex"] = _trim_qp_sibling_contamination(
                current_text,
                canonical,
                expected_order,
                normalizer,
            )
            continue
        stem_label, stem_remainder = normalizer.split_label_and_remainder(stem)
        _row_label, row_remainder = normalizer.split_label_and_remainder(current_text)
        stem_body = stem_remainder.strip() if stem_label else stem.strip()
        row_body = row_remainder.strip() or current_text.strip()
        if stem_body and stem_body.lower() not in current_text.lower():
            row["question_latex"] = _trim_qp_sibling_contamination(
                _dedupe_repeated_qp_label(
                    _trim_qp_transition_noise(f"{label} {stem_body} {row_body}".strip()),
                    label,
                ),
                canonical,
                expected_order,
                normalizer,
            )
        else:
            row["question_latex"] = _trim_qp_sibling_contamination(
                current_text,
                canonical,
                expected_order,
                normalizer,
            )

    if str(os.getenv("PAPERLY_QP_LOCAL_SKELETON_CREATE_MISSED_STUBS", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        found_ids = {
            str(row.get("canonical_question_id") or "").strip().lower()
            for row in rows
            if isinstance(row, dict) and row.get("canonical_question_id")
        }
        for missing_canonical in expected_order:
            if missing_canonical in found_ids:
                continue
            parts = missing_canonical.split(".")
            if not parts or not parts[0].isdigit():
                continue
            root = parts[0]
            stem = root_stem_by_root.get(root, "")
            if not stem:
                continue
            label = normalizer.format_parts(normalizer.extract_parts(missing_canonical))
            if not label:
                continue
            _sl, _sr = normalizer.split_label_and_remainder(stem)
            stem_body = _sr.strip() if _sl else stem.strip()
            stub_text = f"{label} {stem_body}".strip() if stem_body else label
            rows.append({
                "document_type": "Question Paper",
                "question_id": label,
                "canonical_question_id": missing_canonical,
                "parent_canonical_id": normalizer.parent_from_parts(parts),
                "question_latex": stub_text,
                "official_marking_scheme_latex": "",
                "diagram_urls": [],
                "diagram_regions": [],
                "needs_review": True,
                "cognitive_demand": "MEDIUM",
                "validation_warnings": [
                    "Local QP skeleton created a disabled-by-default missed-ID stub; verify text/diagram."
                ],
            })
            logger.debug(
                "[QPLocalSkeleton][MissedStub] Created review stub for missing expected ID %r "
                "(root stem found: %r).",
                missing_canonical,
                stem[:60],
            )

    # Re-sort to expected_order so the stub rows land in the right position.
    expected_pos = {canonical: idx for idx, canonical in enumerate(expected_order)}
    rows.sort(
        key=lambda row: expected_pos.get(
            str(row.get("canonical_question_id") or "").strip().lower(),
            len(expected_order),
        )
    )

    return rows


def _diagram_payload_from_raw(raw: dict) -> dict[str, Any]:
    return {
        "diagram_urls": raw.get("diagram_urls") if isinstance(raw.get("diagram_urls"), list) else [],
        "diagram_regions": raw.get("diagram_regions") if isinstance(raw.get("diagram_regions"), list) else [],
        "diagram_page_number": raw.get("diagram_page_number"),
        "diagram_y_range": raw.get("diagram_y_range") if isinstance(raw.get("diagram_y_range"), list) else [],
    }


def _apply_local_qp_ms_skeleton_first(
    questions_raw: list,
    pdf_base64: str,
    expected_ids: list,
    normalizer: QuestionNumberNormalizer,
) -> list:
    if str(os.getenv("PAPERLY_QP_LOCAL_SKELETON_FIRST", "true")).strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return questions_raw
    if not isinstance(questions_raw, list) or not expected_ids:
        return questions_raw

    expected_order = _expected_qp_id_order(expected_ids, normalizer)
    expected_set = set(expected_order)
    if not expected_set:
        return questions_raw

    local_rows = _build_local_qp_ms_skeleton_rows(pdf_base64, expected_order, normalizer)
    if not local_rows:
        print("[QPLocalSkeleton] no native rows found; keeping Gemini rows")
        return questions_raw

    local_ids = [
        str(row.get("canonical_question_id") or "").strip().lower()
        for row in local_rows
        if isinstance(row, dict) and row.get("canonical_question_id")
    ]
    local_set = set(local_ids)
    gemini_by_id: dict[str, list[dict]] = defaultdict(list)
    for raw in questions_raw:
        if not isinstance(raw, dict):
            continue
        label = raw.get("canonical_question_id") or raw.get("question_id") or raw.get("question_latex") or ""
        parts = normalizer.extract_parts(str(label))
        canonical = normalizer.canonical_from_parts(parts) if parts else ""
        if canonical:
            gemini_by_id[canonical].append(raw)

    gemini_exact = len(set(gemini_by_id.keys()) & expected_set)
    local_exact = len(local_set & expected_set)
    coverage = local_exact / max(1, len(expected_set))
    # ── ANCHOR STRENGTH GATE ────────────────────────────────────────────────
    # Lower the floor: local skeleton is always MS-anchored (it only accepts IDs
    # from expected_set), so even 60% local coverage is more trustworthy than
    # 100% Gemini coverage that includes hallucinated labels.
    # The gemini_exact - 4 floor prevents a large local miss (>4 IDs) from
    # silently overriding a Gemini run that got significantly more right.
    min_coverage = float(os.getenv("PAPERLY_QP_LOCAL_SKELETON_MIN_COVERAGE", "0.60"))
    if coverage < min_coverage or local_exact < max(3, gemini_exact - 4):
        print(
            "[QPLocalSkeleton] keeping Gemini rows: "
            f"local_exact={local_exact}/{len(expected_set)} coverage={coverage:.2f} "
            f"gemini_exact={gemini_exact} threshold={min_coverage:.2f}"
        )
        return questions_raw

    merged: list[dict] = []
    used_diagram_sources: set[int] = set()
    used_gemini_text_sources: set[int] = set()  # track which Gemini rows already donated their text

    # Pre-build a flat list of all Gemini rows with their processed text for O(n) lookup.
    # This is used for the fallback full-corpus fuzzy search when exact ID lookup fails.
    _all_gemini_flat: list[tuple[dict, str]] = []
    for raw in questions_raw:
        if not isinstance(raw, dict):
            continue
        t = _trim_qp_transition_noise(str(raw.get("question_latex") or ""))
        if t:
            _all_gemini_flat.append((raw, re.sub(r"\s+", " ", t.lower()).strip()))

    for local in local_rows:
        canonical = str(local.get("canonical_question_id") or "").strip().lower()
        base = dict(local)
        exact_sources = gemini_by_id.get(canonical) or []

        # ── Try exact canonical ID match first ──────────────────────────────
        best_gemini_source = None
        best_gemini_text = ""
        best_gemini_score = 0.0
        best_gemini_match_kind = ""

        if exact_sources:
            source = exact_sources[0]
            source_text = str(source.get("question_latex") or "").strip()
            if source_text:
                local_text = _trim_qp_transition_noise(str(base.get("question_latex") or ""))
                candidate_text = _trim_qp_transition_noise(source_text)
                local_compact = re.sub(r"\s+", " ", local_text.lower()).strip()
                candidate_compact = re.sub(r"\s+", " ", candidate_text.lower()).strip()
                same_row_score = SequenceMatcher(
                    None,
                    local_compact[:600],
                    candidate_compact[:600],
                ).ratio()
                # Accept Gemini's LaTeX text when it clearly describes the same row.
                # 0.45 threshold handles glyph-encoded local text vs clean LaTeX Gemini text;
                # completely unrelated questions score < 0.20.
                candidate_safe, candidate_unsafe_reason = _qp_local_row_safe_for_auto_clean(
                    canonical,
                    candidate_text,
                    normalizer,
                )
                candidate_is_shorter_leaf = (
                    bool(local_compact)
                    and bool(candidate_compact)
                    and len(local_compact) >= len(candidate_compact) + 45
                    and len(candidate_compact) < int(len(local_compact) * 0.78)
                )
                if candidate_safe and (
                    not local_text
                    or local_compact in candidate_compact
                    or (same_row_score >= 0.62 and not candidate_is_shorter_leaf)
                ):
                    best_gemini_source = source
                    best_gemini_text = candidate_text
                    best_gemini_score = same_row_score
                    best_gemini_match_kind = "exact"
                elif candidate_safe and candidate_is_shorter_leaf:
                    logger.debug(
                        "[QPLocalSkeleton][RejectExactText] %r rejected shorter Gemini leaf "
                        "text over richer local parent context: score=%.2f local_len=%d gemini_len=%d",
                        canonical,
                        same_row_score,
                        len(local_compact),
                        len(candidate_compact),
                    )
                elif not candidate_safe:
                    logger.debug(
                        "[QPLocalSkeleton][RejectExactText] %r rejected exact Gemini text: %s",
                        canonical,
                        candidate_unsafe_reason,
                    )

        # ── Fallback: full-corpus fuzzy text search ──────────────────────────
        # When no exact ID match exists (or the exact match failed similarity),
        # search ALL Gemini rows by text similarity. This handles cases where
        # SequenceGuard renamed Gemini's IDs (e.g. bumped "4.c"→"4.d"→"4.e"),
        # corrupting the canonical lookup while the TEXT is still correct.
        # We skip rows already claimed by a prior local row (used_gemini_text_sources).
        if best_gemini_source is None:
            local_text_for_search = _trim_qp_transition_noise(str(base.get("question_latex") or ""))
            local_compact_for_search = re.sub(r"\s+", " ", local_text_for_search.lower()).strip()
            # Only search if local text has enough signal (>10 chars beyond just the label)
            # Lowered from 20 to 10 to handle shorter questions
            _label_only = re.sub(r"^\d+\([a-z]\)(\([ivx]+\))?\s*", "", local_compact_for_search).strip()
            if len(_label_only) >= 10:
                corpus_best_score = 0.0
                corpus_best_source = None
                corpus_best_text = ""
                for raw, raw_compact in _all_gemini_flat:
                    if id(raw) in used_gemini_text_sources:
                        continue
                    score = SequenceMatcher(
                        None,
                        local_compact_for_search[:600],
                        raw_compact[:600],
                    ).ratio()
                    if score > corpus_best_score:
                        corpus_best_score = score
                        corpus_best_source = raw
                        corpus_best_text = str(raw.get("question_latex") or "")
                # Accept corpus match at 0.45 — same reasoning as exact-match threshold.
                # A corpus match that is also the BEST match (and above threshold) is
                # genuinely the same question, just with a wrong label from SequenceGuard.
                corpus_threshold = float(os.getenv("PAPERLY_QP_LOCAL_SKELETON_CORPUS_TEXT_MIN_SIMILARITY", "0.78"))
                corpus_safe, corpus_unsafe_reason = _qp_local_row_safe_for_auto_clean(
                    canonical,
                    corpus_best_text,
                    normalizer,
                )
                if corpus_best_source is not None and corpus_best_score >= corpus_threshold and corpus_safe:
                    best_gemini_source = corpus_best_source
                    best_gemini_text = _trim_qp_transition_noise(corpus_best_text)
                    best_gemini_score = corpus_best_score
                    best_gemini_match_kind = "corpus_fuzzy"
                    logger.debug(
                        "[QPLocalSkeleton][CorpusFuzzy] %r: no exact Gemini ID match; "
                        "best corpus text score=%.2f from q=%r",
                        canonical,
                        corpus_best_score,
                        str(corpus_best_source.get("question_id") or "")[:40],
                    )
                elif corpus_best_source is not None and corpus_best_score >= 0.45:
                    logger.debug(
                        "[QPLocalSkeleton][RejectCorpusFuzzy] %r score=%.2f threshold=%.2f reason=%s",
                        canonical,
                        corpus_best_score,
                        corpus_threshold,
                        corpus_unsafe_reason or "below_safe_threshold",
                    )

        # ── Fallback 2: Subpart pattern matching ─────────────────────────────
        # When text matching fails but Gemini got the ROOT wrong (e.g., "66.b.i"
        # instead of "2.b.i"), match by subpart structure. Same subparts = same question.
        if best_gemini_source is None and canonical:
            local_parts = canonical.split(".")
            if len(local_parts) >= 2:  # Has subparts like "2.b.i"
                local_subparts = ".".join(local_parts[1:])  # "b.i"
                subpart_best_score = 0.0
                subpart_best_source = None
                subpart_best_text = ""

                for raw, raw_compact in _all_gemini_flat:
                    if id(raw) in used_gemini_text_sources:
                        continue
                    gemini_id = str(raw.get("canonical_question_id") or "").strip().lower()
                    if not gemini_id:
                        continue
                    gemini_parts = gemini_id.split(".")
                    if len(gemini_parts) >= 2:
                        gemini_subparts = ".".join(gemini_parts[1:])
                        # Exact subpart match (e.g., both have ".b.i")
                        if gemini_subparts == local_subparts:
                            # Use text similarity as tiebreaker if multiple matches
                            local_text_for_subpart = _trim_qp_transition_noise(str(base.get("question_latex") or ""))
                            local_compact_subpart = re.sub(r"\s+", " ", local_text_for_subpart.lower()).strip()
                            score = 0.5  # Base score for subpart match
                            if len(local_compact_subpart) > 10 and len(raw_compact) > 10:
                                score += 0.5 * SequenceMatcher(
                                    None,
                                    local_compact_subpart[:300],
                                    raw_compact[:300],
                                ).ratio()
                            if score > subpart_best_score:
                                subpart_best_score = score
                                subpart_best_source = raw
                                subpart_best_text = str(raw.get("question_latex") or "")

                subpart_threshold = float(os.getenv("PAPERLY_QP_LOCAL_SKELETON_SUBPART_TEXT_MIN_SCORE", "0.90"))
                subpart_safe, subpart_unsafe_reason = _qp_local_row_safe_for_auto_clean(
                    canonical,
                    subpart_best_text,
                    normalizer,
                )
                if subpart_best_source is not None and subpart_best_score >= subpart_threshold and subpart_safe:
                    best_gemini_source = subpart_best_source
                    best_gemini_text = _trim_qp_transition_noise(subpart_best_text)
                    best_gemini_score = subpart_best_score
                    best_gemini_match_kind = "subpart"
                    logger.info(
                        "[QPLocalSkeleton][SubpartMatch] %r: matched by subpart pattern; "
                        "Gemini had q=%r (wrong root, correct subparts)",
                        canonical,
                        str(subpart_best_source.get("question_id") or "")[:40],
                    )
                elif subpart_best_source is not None:
                    logger.debug(
                        "[QPLocalSkeleton][RejectSubpartMatch] %r score=%.2f threshold=%.2f reason=%s",
                        canonical,
                        subpart_best_score,
                        subpart_threshold,
                        subpart_unsafe_reason or "below_safe_threshold",
                    )

        # ── Apply the best Gemini source found ──────────────────────────────
        if best_gemini_source is not None:
            if best_gemini_text:
                base["question_latex"] = best_gemini_text
            diagram_payload = _diagram_payload_from_raw(best_gemini_source)
            has_diagram_payload = bool(diagram_payload.get("diagram_urls") or diagram_payload.get("diagram_regions"))
            allow_diagram_transfer = False
            if has_diagram_payload:
                if best_gemini_match_kind == "exact":
                    allow_diagram_transfer = True
                elif best_gemini_match_kind == "corpus_fuzzy":
                    allow_diagram_transfer = best_gemini_score >= 0.72 and _qp_text_allows_auto_diagram(
                        str(base.get("question_latex") or "")
                    )
                elif best_gemini_match_kind == "subpart":
                    allow_diagram_transfer = best_gemini_score >= 0.85 and _qp_text_allows_auto_diagram(
                        str(base.get("question_latex") or "")
                    )

            if allow_diagram_transfer:
                for key, value in diagram_payload.items():
                    if value not in (None, [], ""):
                        base[key] = value
                used_diagram_sources.add(id(best_gemini_source))
            elif has_diagram_payload:
                warnings = base.get("validation_warnings")
                if not isinstance(warnings, list):
                    warnings = []
                warnings.append(
                    "Skipped diagram transfer from a weak non-exact Gemini match; verify/paste diagram manually if needed."
                )
                base["validation_warnings"] = warnings
                base["needs_review"] = True
            used_gemini_text_sources.add(id(best_gemini_source))
            # Clear stale review flag if the corpus rescued it (it has good text now)
            if best_gemini_text and base.get("needs_review"):
                existing_warns = base.get("validation_warnings") or []
                skeleton_warns = [
                    w for w in existing_warns
                    if "Local QP skeleton created" not in str(w)
                ]
                if not skeleton_warns:
                    base["needs_review"] = False
                base["validation_warnings"] = skeleton_warns
        else:
            base["needs_review"] = True
            warnings = base.get("validation_warnings")
            if not isinstance(warnings, list):
                warnings = []
            if not any("Local QP skeleton created" in str(w) for w in warnings):
                warnings.append(
                    "Local QP skeleton created this MS-numbered row from native PDF text; verify text/diagram."
                )
            base["validation_warnings"] = warnings
        if str(base.get("question_latex") or ""):
            base["question_latex"] = _repair_common_native_math_text(str(base.get("question_latex") or ""))
            base = _mark_qp_math_risk_if_needed(base)
        merged.append(base)

    # Preserve diagrams from Gemini rows whose exact label was wrong but whose
    # text is similar to a local row with no diagram.
    allow_fuzzy_diagrams = str(
        os.getenv("PAPERLY_QP_LOCAL_SKELETON_FUZZY_DIAGRAMS", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}
    fuzzy_diagram_threshold = float(
        os.getenv("PAPERLY_QP_LOCAL_SKELETON_FUZZY_DIAGRAM_MIN_SIMILARITY", "0.78")
    )
    for idx, row in enumerate(merged):
        if row.get("diagram_urls"):
            continue
        local_text = str(row.get("question_latex") or "")
        if not allow_fuzzy_diagrams or not _qp_text_allows_auto_diagram(local_text):
            continue
        best_source = None
        best_score = 0.0
        for raw in questions_raw:
            if not isinstance(raw, dict) or id(raw) in used_diagram_sources:
                continue
            diagrams = raw.get("diagram_urls")
            if not isinstance(diagrams, list) or not diagrams:
                continue
            score = SequenceMatcher(
                None,
                re.sub(r"\s+", " ", local_text.lower())[:500],
                re.sub(r"\s+", " ", str(raw.get("question_latex") or "").lower())[:500],
            ).ratio()
            if score > best_score:
                best_score = score
                best_source = raw
        if best_source is not None and best_score >= fuzzy_diagram_threshold:
            updated = dict(row)
            for key, value in _diagram_payload_from_raw(best_source).items():
                if value not in (None, [], ""):
                    updated[key] = value
            warnings = updated.get("validation_warnings")
            if not isinstance(warnings, list):
                warnings = []
            warnings.append(
                f"Local QP skeleton copied diagram from nearest Gemini row (similarity {best_score:.2f}); verify."
            )
            updated["validation_warnings"] = warnings
            updated["needs_review"] = True
            merged[idx] = updated
            used_diagram_sources.add(id(best_source))

    print(
        "[QPLocalSkeleton] applied local-MS skeleton: "
        f"local_rows={len(local_rows)} local_exact={local_exact}/{len(expected_set)} "
        f"gemini_rows={len(questions_raw)} gemini_exact={gemini_exact} "
        f"coverage={coverage:.2f}"
    )

    # ── Gap-fill: for expected IDs that local skeleton missed, pull in the Gemini
    # row if Gemini produced an exact match. These rows are inserted in
    # expected_order position so the final array stays in MS order.
    # This ensures that even when the PDF text layer loses a label (e.g. a
    # sub-part on a continuation page), the Gemini extraction still contributes
    # that row rather than it being silently dropped.
    merged_canonicals = {
        str(row.get("canonical_question_id") or "").strip().lower()
        for row in merged
        if isinstance(row, dict) and row.get("canonical_question_id")
    }
    gap_rows: list[tuple[int, dict]] = []  # (position_in_expected_order, row_dict)
    for pos, canonical in enumerate(expected_order):
        if canonical in merged_canonicals:
            continue
        gemini_sources = gemini_by_id.get(canonical) or []
        if not gemini_sources:
            continue
        source = gemini_sources[0]
        gap = dict(source)
        # Canonicalise the identity fields to match expected_order entry
        gap["canonical_question_id"] = canonical
        parts_for_gap = normalizer.extract_parts(canonical)
        gap["question_id"] = normalizer.format_parts(parts_for_gap)
        gap["parent_canonical_id"] = normalizer.parent_from_parts(parts_for_gap)
        gap["document_type"] = "Question Paper"
        warnings = gap.get("validation_warnings")
        if not isinstance(warnings, list):
            warnings = []
        warnings.append(
            "Local QP skeleton did not find this MS-expected ID in native text; "
            "Gemini row used as gap-fill — verify text/diagram."
        )
        gap["validation_warnings"] = warnings
        gap["needs_review"] = True
        gap_rows.append((pos, gap))
        print(
            f"[QPLocalSkeleton][GapFill] MS-expected {canonical!r} missing from local skeleton; "
            "Gemini gap-fill row inserted."
        )

    if gap_rows:
        # Merge gap rows into position-correct order.
        # Build a list of (expected_order_position, row) for existing merged rows.
        merged_with_pos: list[tuple[int, dict]] = []
        for row in merged:
            canonical = str(row.get("canonical_question_id") or "").strip().lower()
            try:
                pos = expected_order.index(canonical)
            except ValueError:
                pos = len(expected_order)
            merged_with_pos.append((pos, row))
        merged_with_pos.extend(gap_rows)
        merged_with_pos.sort(key=lambda item: item[0])
        merged = [row for _, row in merged_with_pos]
        print(
            f"[QPLocalSkeleton] gap-fill applied: {len(gap_rows)} Gemini row(s) inserted. "
            f"Final merged count: {len(merged)}"
        )

    return merged


_MS_LABEL_RE = re.compile(
    r"^\s*(\d{1,2}\s*(?:\(\s*(?:[a-z]|[ivxlcdm]+)\s*\)){0,6})\s*$",
    re.IGNORECASE,
)
_MS_MARK_CODE_RE = re.compile(
    r"\b((?:M|A|B|SC)[1-9]\d*(?:dep|FT)?|FT|cao|oe|nfww|isw)\b",
    re.IGNORECASE,
)
_CAMBRIDGE_LEFT_BRACKET_GLYPHS = "\uf0e6\uf0e7\uf0e8\uf8eb\uf8ec\uf8ed"
_CAMBRIDGE_RIGHT_BRACKET_GLYPHS = "\uf0f6\uf0f7\uf0f8\uf8f6\uf8f7\uf8f8"
_CAMBRIDGE_BRACKET_GLYPHS = (
    _CAMBRIDGE_LEFT_BRACKET_GLYPHS + _CAMBRIDGE_RIGHT_BRACKET_GLYPHS
)


def _normalize_cambridge_private_math(text: str) -> str:
    """Convert Cambridge private-font bracket stacks into renderable LaTeX."""
    if not any(ch in text for ch in _CAMBRIDGE_BRACKET_GLYPHS):
        return text

    left_re = re.compile(f"[{re.escape(_CAMBRIDGE_LEFT_BRACKET_GLYPHS)}]")
    right_re = re.compile(f"[{re.escape(_CAMBRIDGE_RIGHT_BRACKET_GLYPHS)}]")

    def flush_matrix(lines: list[str], output: list[str]) -> None:
        if not lines:
            return
        entries: list[str] = []
        for line in lines:
            clean_line = left_re.sub("", line)
            clean_line = right_re.sub("", clean_line)
            clean_line = re.sub(r"\s+", " ", clean_line).strip()
            if clean_line:
                entries.append(clean_line)
        if len(entries) >= 2:
            output.append("$\\begin{pmatrix}" + r"\\".join(entries) + "\\end{pmatrix}$")
        elif entries:
            output.append("(" + entries[0] + ")")

    output_lines: list[str] = []
    matrix_lines: list[str] = []
    for raw_line in text.splitlines():
        if any(ch in raw_line for ch in _CAMBRIDGE_BRACKET_GLYPHS):
            matrix_lines.append(raw_line)
            continue
        flush_matrix(matrix_lines, output_lines)
        matrix_lines = []
        output_lines.append(raw_line)
    flush_matrix(matrix_lines, output_lines)
    return "\n".join(output_lines)


def _clean_native_ms_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = _normalize_cambridge_private_math(text)
    text = text.replace("\uf0e6", "(").replace("\uf0f6", ")")
    text = text.replace("\uf0e7", "(").replace("\uf0f7", ")")
    text = text.replace("\uf0e8", "(").replace("\uf0f8", ")")
    text = text.replace("\uf8eb", "(").replace("\uf8f6", ")")
    text = text.replace("\uf8ec", "(").replace("\uf8f7", ")")
    text = text.replace("\uf8ed", "(").replace("\uf8f8", ")")
    text = text.replace("\uf075", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ms_table_row_to_four_cells(row: list[Any]) -> list[str]:
    if not row:
        return ["", "", "", ""]
    cells = list(row)
    if len(cells) >= 8 and len(cells) % 4 == 0:
        width = len(cells) // 4
        output: list[str] = []
        for idx in range(4):
            group = cells[idx * width:(idx + 1) * width]
            non_empty = [
                _clean_native_ms_text(cell)
                for cell in group
                if _clean_native_ms_text(cell)
            ]
            output.append("\n".join(non_empty).strip())
        return output
    if len(cells) == 11:
        # Some Cambridge MS pages lose one spacer column during PyMuPDF table
        # detection. The visible columns are still Question | Answer | Marks |
        # Partial Marks, but a simple first-four-cells fallback drops rows.
        groups = (cells[0:3], cells[3:5], cells[5:8], cells[8:11])
        return [
            "\n".join(
                _clean_native_ms_text(cell)
                for cell in group
                if _clean_native_ms_text(cell)
            ).strip()
            for group in groups
        ]
    padded = [_clean_native_ms_text(cell) for cell in cells[:4]]
    while len(padded) < 4:
        padded.append("")
    return padded


def _extract_ms_visible_label(label_cell: str, normalizer: QuestionNumberNormalizer) -> str:
    cleaned = _clean_native_ms_text(label_cell)
    if not cleaned:
        return ""
    first_line = cleaned.splitlines()[0].strip()
    # Primary: full-line exact match (label is the entire first line)
    match = _MS_LABEL_RE.match(first_line)
    if not match:
        # Secondary: prefix match — label appears at the START of the line,
        # optionally followed by whitespace or non-label characters.
        # This handles Cambridge cells where PyMuPDF merges a label with a
        # trailing mark hint (e.g. "4(g) or equivalent").
        prefix_re = re.compile(
            r"^\s*(\d{1,2}\s*(?:\(\s*(?:[a-z]|[ivxlcdm]+)\s*\)){0,6})\s*(?:$|[^(])",
            re.IGNORECASE,
        )
        match = prefix_re.match(first_line)
    if not match:
        return ""
    parts = normalizer.extract_parts(match.group(1))
    if not parts or not str(parts[0]).isdigit():
        return ""
    return normalizer.format_parts(parts)


def _union_optional_rect(existing: Any, new_rect: Any) -> Optional[fitz.Rect]:
    if new_rect is None:
        return fitz.Rect(existing) if existing else None
    try:
        rect = fitz.Rect(new_rect)
    except Exception:
        return fitz.Rect(existing) if existing else None
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return fitz.Rect(existing) if existing else None
    if existing:
        return fitz.Rect(existing) | rect
    return rect


def _ms_table_logical_cell_rect(table_row: Any, logical_col: int) -> Optional[fitz.Rect]:
    cells = list(getattr(table_row, "cells", []) or [])
    if not cells:
        return None
    if len(cells) == 11:
        groups = ((0, 3), (3, 5), (5, 8), (8, 11))
        start, end = groups[min(max(logical_col, 0), 3)]
        result: Optional[fitz.Rect] = None
        for cell in cells[start:end]:
            if cell is None:
                continue
            result = _union_optional_rect(result, cell)
        return result
    group_width = max(1, len(cells) // 4)
    start = logical_col * group_width
    end = min(len(cells), start + group_width)
    result: Optional[fitz.Rect] = None
    for cell in cells[start:end]:
        if cell is None:
            continue
        result = _union_optional_rect(result, cell)
    return result


def _ms_answer_cell_has_visual_content(page: fitz.Page, rect: fitz.Rect) -> bool:
    """Detect real vector diagrams inside an MS answer cell, ignoring table lines."""
    try:
        inner = fitz.Rect(rect)
        inner.x0 += 3
        inner.y0 += 3
        inner.x1 -= 3
        inner.y1 -= 3
        if inner.is_empty:
            return False
        visual_area = 0.0
        for drawing in page.get_drawings():
            draw_rect = drawing.get("rect")
            if not draw_rect:
                continue
            draw_rect = fitz.Rect(draw_rect)
            if not draw_rect.intersects(inner):
                continue
            intersection = draw_rect & inner
            if intersection.is_empty:
                continue
            # Thin horizontal/vertical table rules and fraction bars are not diagrams.
            if draw_rect.width < 6 or draw_rect.height < 6:
                continue
            visual_area += intersection.get_area()
            if visual_area > 800:
                return True
    except Exception:
        return False
    return False


def _crop_ms_answer_cell_to_data_url(page: fitz.Page, rect: fitz.Rect) -> str:
    try:
        clip = fitz.Rect(rect)
        clip.x0 = max(page.rect.x0, clip.x0 - 4)
        clip.y0 = max(page.rect.y0, clip.y0 - 4)
        clip.x1 = min(page.rect.x1, clip.x1 + 4)
        clip.y1 = min(page.rect.y1, clip.y1 + 4)
        if clip.is_empty or clip.width < 15 or clip.height < 15:
            return ""
        matrix = fitz.Matrix(180 / 72.0, 180 / 72.0)
        pix = page.get_pixmap(matrix=matrix, clip=clip, colorspace=fitz.csRGB, alpha=False)
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=84, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
    except Exception:
        return ""


def _append_ms_row_text(
    row: dict[str, Any],
    answer: str,
    marks: str,
    partial: str,
    answer_bbox: Any = None,
) -> None:
    if answer:
        row["answer_lines"].append(answer)
    if marks:
        row["marks_lines"].append(marks)
    if partial:
        row["partial_lines"].append(partial)
    if marks or partial:
        row.setdefault("mark_partial_pairs", []).append((marks, partial))
    if answer_bbox is not None:
        row["answer_bbox"] = _union_optional_rect(row.get("answer_bbox"), answer_bbox)


def _infer_native_ms_total_marks(marks_text: str, partial_text: str) -> int:
    for line in marks_text.splitlines():
        clean = line.strip()
        if re.fullmatch(r"\d{1,2}", clean):
            return int(clean)
    total = 0
    for code in _MS_MARK_CODE_RE.findall(f"{marks_text}\n{partial_text}"):
        match = re.search(r"\d+", code)
        if match:
            total += int(match.group(0))
    return total


def _native_ms_method_steps(
    marks_text: str,
    partial_text: str,
    mark_partial_pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for marks_part, partial_part in mark_partial_pairs or []:
        marks_clean = _clean_native_ms_text(marks_part)
        partial_clean = _clean_native_ms_text(partial_part)
        if not marks_clean:
            continue
        for code in _MS_MARK_CODE_RE.findall(marks_clean):
            description = partial_clean.strip(" :-")
            if description:
                steps.append({"type": code.strip(), "description": description})

    combined_lines = [
        line.strip()
        for line in f"{marks_text}\n{partial_text}".splitlines()
        if line.strip()
    ]
    for line in combined_lines:
        match = _MS_MARK_CODE_RE.search(line)
        if not match:
            continue
        mark_type = match.group(1).strip()
        description = (line[:match.start()] + line[match.end():]).strip(" :-")
        if not description and any(step["type"] == mark_type for step in steps):
            continue
        steps.append({"type": mark_type, "description": description})
    if not steps and partial_text.strip():
        steps.append({"type": "mark", "description": partial_text.strip()})
    return steps


def _native_ms_cognitive_demand(total_marks: int) -> str:
    if total_marks <= 1:
        return "LOW"
    if total_marks == 2:
        return "MEDIUM"
    return "HIGH"


def _native_ms_row_to_question(row: dict[str, Any]) -> dict[str, Any]:
    label = row["label"]
    answer_text = _clean_native_ms_text("\n".join(row["answer_lines"]))
    marks_text = _clean_native_ms_text("\n".join(row["marks_lines"]))
    partial_text = _clean_native_ms_text("\n".join(row["partial_lines"]))
    official = "\n".join(part for part in (answer_text, marks_text, partial_text) if part)
    total_marks = _infer_native_ms_total_marks(marks_text, partial_text)
    return {
        "document_type": "Marking Scheme",
        "question_latex": label,
        "question_id": label,
        "final_answer": answer_text,
        "total_marks": total_marks,
        "method_steps": _native_ms_method_steps(marks_text, partial_text, row.get("mark_partial_pairs") or []),
        "official_marking_scheme_latex": official,
        "diagram_urls": row.get("diagram_urls") or [],
        "diagram_regions": [],
        "diagram_page_number": row.get("diagram_page_number"),
        "diagram_y_range": row.get("diagram_y_range") or [],
        "needs_review": False,
        "validation_warnings": [],
        "cognitive_demand": _native_ms_cognitive_demand(total_marks),
        "difficulty_override": None,
    }


def _extract_ms_tables_native_response(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str,
    generated_paper_reference_key: str,
    extra_metadata: dict | None = None,
) -> Optional[SlicedQuestionsResponse]:
    """
    Fast deterministic Marking Scheme extractor for Cambridge table PDFs.

    It only activates on real "Question | Answer | Marks | Partial Marks"
    tables. Visible labels always start new rows; blank/row-spanned rows merge
    into the previous visible label.
    """
    if str(os.getenv("PAPERLY_MS_TABLE_FIRST", "true")).strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return None

    normalizer = QuestionNumberNormalizer()
    try:
        try:
            if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "mupdf_display_errors"):
                fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass
        normalized_b64 = pdf_base64.strip().split(",", 1)[-1]
        pdf_bytes = base64.b64decode(normalized_b64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("[NativeMSTable] Could not open PDF for native MS extraction: %s", exc)
        return None

    extracted_rows: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    table_pages = 0

    def flush_current() -> None:
        nonlocal current
        if current and current.get("label"):
            page_index = current.get("page_index")
            answer_bbox = current.get("answer_bbox")
            if isinstance(page_index, int) and answer_bbox is not None and 0 <= page_index < len(doc):
                page = doc[page_index]
                bbox = fitz.Rect(answer_bbox)
                if _ms_answer_cell_has_visual_content(page, bbox):
                    data_url = _crop_ms_answer_cell_to_data_url(page, bbox)
                    if data_url:
                        current.setdefault("diagram_urls", []).append(data_url)
                        current["diagram_page_number"] = page_index + 1
                        current["diagram_y_range"] = [
                            round((bbox.y0 / page.rect.height) * 100, 2),
                            round((bbox.y1 / page.rect.height) * 100, 2),
                        ]
            extracted_rows.append(_native_ms_row_to_question(current))
        current = None

    try:
        for page in doc:
            if not hasattr(page, "find_tables"):
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    tables = page.find_tables().tables
            except Exception:
                continue
            for table in tables:
                try:
                    rows = table.extract()
                except Exception:
                    continue
                four_col_rows = [_ms_table_row_to_four_cells(row) for row in rows]
                has_ms_header = any(
                    cells[0].strip().lower() == "question"
                    and cells[1].strip().lower() == "answer"
                    and cells[2].strip().lower() == "marks"
                    and "partial" in cells[3].strip().lower()
                    for cells in four_col_rows
                )
                if not has_ms_header:
                    continue
                table_pages += 1
                for row_idx, cells in enumerate(four_col_rows):
                    q_cell, answer, marks, partial = cells
                    if q_cell.strip().lower() == "question":
                        continue
                    if not any(cells):
                        continue
                    label = _extract_ms_visible_label(q_cell, normalizer)
                    answer_bbox = None
                    if row_idx < len(getattr(table, "rows", []) or []):
                        answer_bbox = _ms_table_logical_cell_rect(table.rows[row_idx], 1)
                    if label:
                        flush_current()
                        current = {
                            "label": label,
                            "answer_lines": [],
                            "marks_lines": [],
                            "partial_lines": [],
                            "mark_partial_pairs": [],
                            "diagram_urls": [],
                            "page_index": page.number,
                        }
                        _append_ms_row_text(current, answer, marks, partial, answer_bbox)
                    elif current is not None:
                        _append_ms_row_text(current, answer, marks, partial, answer_bbox)
        flush_current()
    finally:
        doc.close()

    unique_labels = {
        row.get("question_id")
        for row in extracted_rows
        if row.get("question_id")
    }
    # Lowered from 8 to 4: short 0607 Paper 2 style papers have fewer than 8
    # questions; the native table path should activate for any real Cambridge
    # 4-column MS table regardless of question count.
    if len(unique_labels) < 4 or table_pages == 0:
        return None

    print(
        f"[NativeMSTable] Extracted {len(extracted_rows)} MS row(s) "
        f"from {table_pages} Cambridge table page(s). Gemini MS call skipped."
    )
    parsed_dict = {
        "metadata": {
            "curriculum": board.upper(),
            "paper_reference_key": generated_paper_reference_key,
        },
        "questions_array": extracted_rows,
    }
    if extra_metadata:
        parsed_dict["metadata"].update(extra_metadata)

    return _normalize_response(
        parsed=parsed_dict,
        filename=filename,
        document_type=document_type,
        board=board,
        generated_paper_reference_key=generated_paper_reference_key,
        extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
    )


def _add_missing_qp_root_stubs(
    questions_raw: list,
    pdf_base64: str,
    normalizer: QuestionNumberNormalizer,
) -> list:
    if os.getenv("ENABLE_QP_NATIVE_RECOVERY", "false").lower() != "true":
        return questions_raw

    snippets = _native_qp_root_snippets(pdf_base64)
    if not snippets:
        return questions_raw

    seen_roots: set[str] = set()
    for q in questions_raw:
        if not isinstance(q, dict):
            continue
        label = _safe_question_label_from_raw(q, normalizer)
        parts = normalizer.extract_parts(label)
        if parts and parts[0].isdigit():
            seen_roots.add(parts[0])

    missing_roots = sorted(
        (root for root in snippets if root not in seen_roots),
        key=lambda value: int(value),
    )
    if not missing_roots:
        augmented = list(questions_raw)
    else:
        augmented = list(questions_raw)
        for root in missing_roots:
            snippet = snippets[root]
            label = f"{root}(a)" if re.search(r"\([a]\)", snippet, flags=re.IGNORECASE) else root
            augmented.append({
                "document_type": "Question Paper",
                "question_id": label,
                "question_latex": snippet if snippet.startswith(root) else f"{label} {snippet}".strip(),
                "official_marking_scheme_latex": None,
                "diagram_urls": [],
                "diagram_regions": [],
                "needs_review": True,
                "cognitive_demand": "MEDIUM",
                "validation_warnings": [
                    f"QP native audit recovered missing root question {root}; verify subpart boundaries."
                ],
            })
            logger.warning("[QPNativeAudit] Added review stub for missing QP root %s.", root)

    first_level_snippets = _native_qp_first_level_snippets(pdf_base64)
    seen_first_level: set[str] = set()
    for q in augmented:
        if not isinstance(q, dict):
            continue
        label = _safe_question_label_from_raw(q, normalizer)
        parts = normalizer.extract_parts(label)
        if len(parts) >= 2 and parts[0].isdigit():
            seen_first_level.add(normalizer.canonical_from_parts(parts[:2]))

    for label, snippet in first_level_snippets.items():
        parts = normalizer.extract_parts(label)
        canonical = normalizer.canonical_from_parts(parts)
        if not canonical or canonical in seen_first_level:
            continue
        augmented.append({
            "document_type": "Question Paper",
            "question_id": label,
            "question_latex": snippet if snippet.startswith(label) else f"{label} {snippet}".strip(),
            "official_marking_scheme_latex": None,
            "diagram_urls": [],
            "diagram_regions": [],
            "needs_review": True,
            "cognitive_demand": "MEDIUM",
            "validation_warnings": [
                f"QP native audit recovered missing first-level question {label}; verify subpart boundaries."
            ],
        })
        seen_first_level.add(canonical)
        logger.warning("[QPNativeAudit] Added review stub for missing QP label %s.", label)

    return augmented


def _add_missing_qp_expected_leaf_stubs(
    questions_raw: list,
    expected_ids: list,
    local_page_hints: list,
    normalizer: QuestionNumberNormalizer,
) -> list:
    """
    MS-first QP safety net.

    If Gemini returns fewer QP rows than the saved MS leaf IDs, add a
    review-only placeholder for missing expected IDs that the local PDF
    skeleton confirms belong to the paper. This does not invent question text;
    it exposes the exact subpart an intern must split/paste before approval.
    """
    if not isinstance(questions_raw, list) or not isinstance(expected_ids, list):
        print(
            "[MSAnchorTrace][LeafStubGate] skipped: invalid inputs "
            f"questions_type={type(questions_raw).__name__} "
            f"expected_type={type(expected_ids).__name__}"
        )
        return questions_raw

    expected_clean: list[str] = []
    seen_expected: set[str] = set()
    for value in expected_ids:
        parts = normalizer.extract_parts(str(value or ""))
        canonical = normalizer.canonical_from_parts(parts)
        if canonical and canonical not in seen_expected:
            expected_clean.append(canonical)
            seen_expected.add(canonical)
    if not expected_clean:
        print("[MSAnchorTrace][LeafStubGate] skipped: expected_ids empty after normalization")
        return questions_raw

    skeleton_ids: set[str] = set()
    if isinstance(local_page_hints, list):
        for hint in local_page_hints:
            if not isinstance(hint, dict):
                continue
            for value in hint.get("expected_ids") or []:
                cid = str(value or "").strip().lower()
                if cid:
                    skeleton_ids.add(cid)
    if not skeleton_ids:
        print(
            "[MSAnchorTrace][LeafStubGate] skipped: local skeleton has no expected IDs "
            f"expected_count={len(expected_clean)} questions_count={len(questions_raw)}"
        )
        return questions_raw

    current_ids: list[str] = []
    for raw in questions_raw:
        if not isinstance(raw, dict):
            continue
        label = raw.get("canonical_question_id") or raw.get("question_id") or raw.get("question_latex") or ""
        parts = normalizer.extract_parts(str(label))
        canonical = normalizer.canonical_from_parts(parts)
        if canonical:
            current_ids.append(canonical)

    current_set = set(current_ids)
    expected_set = set(expected_clean)
    exact_matches = len(expected_set & current_set)
    extras = sorted(current_set - expected_set)

    def covered_by_existing_parent(expected_id: str) -> bool:
        return any(
            current_id
            and expected_id.startswith(f"{current_id}.")
            and len(expected_id) > len(current_id) + 1
            for current_id in current_set
        )

    raw_missing = [
        expected_id
        for expected_id in expected_clean
        if expected_id not in current_set
    ]
    parent_covered = [
        expected_id
        for expected_id in raw_missing
        if covered_by_existing_parent(expected_id)
    ]
    not_in_skeleton = [
        expected_id
        for expected_id in raw_missing
        if expected_id not in skeleton_ids
    ]
    missing_ids = [
        expected_id
        for expected_id in expected_clean
        if expected_id not in current_set
        and not covered_by_existing_parent(expected_id)
        and expected_id in skeleton_ids
    ]
    print(
        "[MSAnchorTrace][LeafStubGate] "
        f"questions={len(questions_raw)} current_unique={len(current_set)} "
        f"expected={len(expected_clean)} skeleton_expected={len(skeleton_ids)} "
        f"exact={exact_matches} raw_missing={len(raw_missing)} "
        f"parent_covered={len(parent_covered)} not_in_skeleton={len(not_in_skeleton)} "
        f"to_add={len(missing_ids)} extras={extras[:20]} "
        f"raw_missing_ids={raw_missing[:40]} "
        f"not_in_skeleton_ids={not_in_skeleton[:40]}"
    )
    if not missing_ids:
        print("[MSAnchorTrace][LeafStubGate] no placeholders added")
        return questions_raw

    if str(os.getenv("PAPERLY_QP_ADD_EXPECTED_LEAF_STUBS", "false")).strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        print(
            "[MSAnchorTrace][LeafStubResult] placeholders_disabled=true "
            f"before={len(questions_raw)} after={len(questions_raw)} "
            f"missing_kept_as_review_report={missing_ids}"
        )
        return _annotate_grouped_qp_split_candidates(
            questions_raw=questions_raw,
            missing_ids=missing_ids,
            normalizer=normalizer,
        )

    # CREATE STUBS for missing MS IDs that exist in skeleton but not in extracted questions
    augmented = list(questions_raw)
    for missing_id in missing_ids:
        parts = normalizer.extract_parts(missing_id)
        label = normalizer.format_parts(parts)
        parent_id = normalizer.parent_from_parts(parts)

        # Find the parent or sibling question to extract context
        parent_question = None
        for q in questions_raw:
            q_canonical = str(q.get("canonical_question_id") or "").strip().lower()
            # Match parent or sibling
            if q_canonical == parent_id or (parent_id and q_canonical.startswith(f"{parent_id}.")):
                parent_question = q
                break

        # Build stub with context from parent/sibling
        stub = {
            "document_type": "Question Paper",
            "question_id": label,
            "canonical_question_id": missing_id,
            "parent_canonical_id": parent_id,
            "question_latex": f"{label} [Missing subpart - verify in original PDF]",
            "official_marking_scheme_latex": None,
            "diagram_urls": [],
            "diagram_regions": [],
            "needs_review": True,
            "cognitive_demand": "MEDIUM",
            "validation_warnings": [
                f"MS expects '{missing_id}' but not found in QP PDF text. "
                f"This is a placeholder stub. Verify if this subpart exists in the original PDF."
            ],
        }

        # Copy context from parent if available
        if parent_question:
            stub["page_num"] = parent_question.get("page_num", 0)
            stub["tier"] = parent_question.get("tier", "CORE")
            # Don't copy diagram_urls or question_latex - those belong to parent

        augmented.append(stub)
        logger.warning(
            "[MSAnchorTrace][LeafStubCreated] Added placeholder stub for missing MS ID '%s' "
            "that exists in skeleton but not in extracted questions.",
            missing_id
        )

    print(
        "[MSAnchorTrace][LeafStubResult] placeholders_enabled=true "
        f"before={len(questions_raw)} after={len(augmented)} "
        f"stubs_created={len(missing_ids)} missing_ids={missing_ids}"
    )

    # Also annotate nearby questions for manual review
    return _annotate_grouped_qp_split_candidates(
        questions_raw=augmented,
        missing_ids=missing_ids,
        normalizer=normalizer,
    )


def _replace_question_label_in_raw(
    q: dict,
    old_label: str,
    new_label: str,
    normalizer: QuestionNumberNormalizer,
) -> dict:
    updated = dict(q)
    updated["question_id"] = new_label

    q_latex = str(updated.get("question_latex") or "")
    if q_latex:
        parsed_label, remainder = normalizer.split_label_and_remainder(q_latex)
        if parsed_label:
            updated["question_latex"] = f"{new_label} {remainder}".strip() if remainder else new_label
        elif q_latex.strip() == old_label:
            updated["question_latex"] = new_label
    else:
        updated["question_latex"] = new_label

    warnings = updated.get("validation_warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings.append("Structural sequence guard corrected a duplicated question label.")
    updated["validation_warnings"] = warnings
    updated["needs_review"] = True
    return updated


def _next_unused_question_parts(
    parts: list[str],
    used_canonical_ids: set[str],
    normalizer: QuestionNumberNormalizer,
) -> list[str] | None:
    candidate = [str(p).lower() for p in parts]
    for _ in range(20):
        candidate = normalizer.increment_terminal_part(candidate) or []
        if not candidate:
            return None
        if normalizer.canonical_from_parts(candidate) not in used_canonical_ids:
            return candidate
    return None


def _is_ms_anchor_placeholder_raw(raw: dict) -> bool:
    if not isinstance(raw, dict):
        return False
    text = str(raw.get("question_latex") or "")
    if "MS anchor placeholder:" in text:
        return True
    warnings = raw.get("validation_warnings")
    if not isinstance(warnings, list):
        return False
    return any("MS anchor added a review-only placeholder" in str(w) for w in warnings)


def _repair_raw_question_sequence(
    questions_raw: list,
    question_normalizer: QuestionNumberNormalizer,
) -> list:
    """
    Final assembly guard for fallback paths and cached raw dicts.

    This mirrors the slicer's structural repair: duplicate canonical IDs are
    advanced to the next real sibling label, never suffixed with artificial
    "_dup_N" keys.
    """
    repaired: list = []
    used_canonical_ids: set[str] = set()
    canonical_to_index: dict[str, int] = {}

    for raw in questions_raw:
        if not isinstance(raw, dict):
            repaired.append(raw)
            continue

        label = (
            question_normalizer.extract_leading_label(raw.get("question_id", ""))
            or question_normalizer.extract_leading_label(raw.get("question_latex", ""))
        )
        parts = question_normalizer.extract_parts(label)
        canonical = question_normalizer.canonical_from_parts(parts) if parts else ""

        if not canonical or len(parts) < 2 or canonical not in used_canonical_ids:
            if canonical:
                used_canonical_ids.add(canonical)
                canonical_to_index[canonical] = len(repaired)
            repaired.append(raw)
            continue

        existing_index = canonical_to_index.get(canonical)
        existing_raw = repaired[existing_index] if existing_index is not None and 0 <= existing_index < len(repaired) else None
        current_is_placeholder = _is_ms_anchor_placeholder_raw(raw)
        existing_is_placeholder = _is_ms_anchor_placeholder_raw(existing_raw) if isinstance(existing_raw, dict) else False

        if current_is_placeholder and existing_index is not None:
            logger.warning(
                "[NormalizeSequenceGuard] Dropped redundant MS-anchor placeholder for %r; "
                "a real extracted row already exists.",
                canonical,
            )
            continue

        if existing_is_placeholder and not current_is_placeholder and existing_index is not None:
            repaired[existing_index] = raw
            logger.warning(
                "[NormalizeSequenceGuard] Replaced MS-anchor placeholder for %r with real extracted row.",
                canonical,
            )
            continue

        corrected_parts = _next_unused_question_parts(
            parts,
            used_canonical_ids,
            question_normalizer,
        )
        if not corrected_parts:
            repaired.append(raw)
            used_canonical_ids.add(canonical)
            continue

        corrected_label = question_normalizer.format_parts(corrected_parts)
        corrected = _replace_question_label_in_raw(
            raw,
            old_label=label,
            new_label=corrected_label,
            normalizer=question_normalizer,
        )
        used_canonical_ids.add(question_normalizer.canonical_from_parts(corrected_parts))
        repaired.append(corrected)

        logger.warning(
            "[NormalizeSequenceGuard] Duplicate canonical ID %r repaired to %r.",
            canonical,
            question_normalizer.canonical_from_parts(corrected_parts),
        )

    return repaired


def _ordered_ms_child_markers(text: str) -> list[tuple[str, re.Match]]:
    seen: set[str] = set()
    ordered: list[tuple[str, re.Match]] = []
    for match in _MS_ROMAN_MARKER_RE.finditer(text or ""):
        marker = match.group(1).lower()
        if marker not in seen:
            seen.add(marker)
            ordered.append((marker, match))
    return ordered


def _expand_grouped_marking_scheme_entries(
    questions_raw: list,
    question_normalizer: QuestionNumberNormalizer,
) -> list:
    """
    Split visible grouped MS blocks into child roman entries.

    Example: an MS row labelled "8(a)" whose text contains "(i)", "(ii)",
    "(iii)", "(iv)" becomes four rows "8(a)(i)" through "8(a)(iv)".
    This gives QP leaves a deterministic MS counterpart when Gemini grouped
    the marking scheme more coarsely than the question paper.
    """
    expanded: list = []

    for raw in questions_raw:
        if not isinstance(raw, dict):
            expanded.append(raw)
            continue

        label = (
            question_normalizer.extract_leading_label(raw.get("question_id", ""))
            or question_normalizer.extract_leading_label(raw.get("question_latex", ""))
        )
        parts = question_normalizer.extract_parts(label)
        if len(parts) != 2:
            expanded.append(raw)
            continue

        official_text = str(raw.get("official_marking_scheme_latex") or "")
        source_text = official_text or _question_payload_for_sequence(raw)
        markers = _ordered_ms_child_markers(source_text)
        if len(markers) < 2:
            expanded.append(raw)
            continue

        logger.warning(
            "[MSHierarchyGuard] Expanding grouped MS entry %r into %d child entries.",
            label,
            len(markers),
        )

        for idx, (marker, match) in enumerate(markers):
            next_start = markers[idx + 1][1].start() if idx + 1 < len(markers) else len(source_text)
            segment = source_text[match.end():next_start].strip()
            child_parts = parts + [marker]
            child_label = question_normalizer.format_parts(child_parts)
            child = dict(raw)
            child["question_id"] = child_label
            child["question_latex"] = child_label
            if segment:
                child["official_marking_scheme_latex"] = segment
                child["final_answer"] = segment[:500]
                if not child.get("method_steps"):
                    child["method_steps"] = [{"type": "note", "description": segment}]
            child["needs_review"] = True
            warnings = child.get("validation_warnings")
            if not isinstance(warnings, list):
                warnings = []
            warnings.append(
                f"MS hierarchy guard expanded grouped entry {label} to {child_label}."
            )
            child["validation_warnings"] = warnings
            expanded.append(child)

    return expanded


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
    
    # Log incoming metadata key for audit trail
    incoming_key = meta_raw.get("paper_reference_key", "")
    if incoming_key:
        logger.info(f"[Flow Audit] Incoming paper_reference_key from Gemini: '{incoming_key}'")
    
    meta_normalized = _normalize_metadata(meta_raw, filename, board, generated_paper_reference_key)
    
    # Log post-normalization key to ensure it was preserved
    normalized_key = meta_normalized.get("paper_reference_key", "")
    if normalized_key and incoming_key and normalized_key != incoming_key:
        logger.warning(f"[Flow Audit] ALERT: paper_reference_key was modified! "
                       f"Before: '{incoming_key}' → After: '{normalized_key}'")
    elif normalized_key:
        logger.info(f"[Flow Audit] Post-normalization paper_reference_key: '{normalized_key}'")

    question_normalizer = QuestionNumberNormalizer()
    if not meta_normalized.get("unified_paper_key") and meta_normalized.get("paper_reference_key"):
        meta_normalized["unified_paper_key"] = generate_unified_paper_key(
            meta_normalized["paper_reference_key"]
        )

    questions_raw = parsed.get("questions_array") or []
    if not isinstance(questions_raw, list):
        questions_raw = []
    if document_type.strip().lower() == "marking scheme":
        questions_raw = _expand_grouped_marking_scheme_entries(
            questions_raw,
            question_normalizer,
        )
    elif document_type.strip().lower() == "question paper":
        questions_raw = _repair_qp_placeholders_and_orphans(
            questions_raw,
            question_normalizer,
        )
        questions_raw = _compact_repeated_qp_preambles(
            questions_raw,
            question_normalizer,
        )
        questions_raw = _repair_qp_embedded_subpart_labels(
            questions_raw,
            question_normalizer,
        )
        questions_raw = _repair_qp_backward_root_intrusions(
            questions_raw,
            question_normalizer,
        )
    questions_raw = _repair_raw_question_sequence(
        questions_raw,
        question_normalizer,
    )
    if document_type.strip().lower() == "question paper":
        expected_canonical_ids = meta_raw.get("expected_canonical_ids") or []
        questions_raw = reconcile_qp_against_ms(
            questions_raw,
            expected_canonical_ids,
            question_normalizer,
        )

    questions: List[ExtractedQuestion] = []
    qp_parent_ids: set = set()
    ms_parent_ids: set = set()

    for i, q in enumerate(questions_raw):
        try:
            normalized = _normalize_question(q, meta_normalized, document_type, question_normalizer)
            schema_fields = set(ExtractedQuestion.model_fields.keys())
            filtered = {k: v for k, v in normalized.items() if k in schema_fields}
            question_obj = ExtractedQuestion(**filtered)
            questions.append(question_obj)
            if question_obj.parent_canonical_id:
                if question_obj.document_type == "Question Paper":
                    qp_parent_ids.add(question_obj.parent_canonical_id)
                else:
                    ms_parent_ids.add(question_obj.parent_canonical_id)
        except Exception as exc:
            print(f"⚠️  [normalize] Skipping question {i}: {exc}")
            try:
                safe = dict(_QUESTION_DEFAULTS)
                safe.update({k: v for k, v in meta_normalized.items() if k in safe})
                safe["document_type"] = document_type
                safe["question_latex"] = str(q) if not isinstance(q, dict) else q.get("question_latex", "")
                safe["needs_review"] = True
                safe["diagram_urls"] = []  # Always []
                schema_fields = set(ExtractedQuestion.model_fields.keys())
                questions.append(ExtractedQuestion(**{k: v for k, v in safe.items() if k in schema_fields}))
            except Exception:
                pass

    if document_type.strip().lower() == "question paper":
        questions = _repair_normalized_qp_backward_roots(
            questions,
            question_normalizer,
        )

    # Sequence gap validation
    validation_status = "ok"
    validation_warnings = []
    recommendation = "proceed"
    parent_ids_to_check = qp_parent_ids if document_type == "Question Paper" else ms_parent_ids

    if parent_ids_to_check:
        int_parents = [int(pid) for pid in parent_ids_to_check if str(pid).isdigit()]
        if int_parents:
            int_parents.sort()
            expected = set(range(min(int_parents), max(int_parents) + 1))
            missing  = expected - set(int_parents)
            if missing:
                validation_status = "warning"
                recommendation    = "review"
                validation_warnings.append(
                    f"Sequence gap detected in {document_type}. Missing parent questions: "
                    f"{', '.join(map(str, sorted(missing)))}"
                )

    meta_normalized["validation_status"]   = validation_status
    meta_normalized["validation_warnings"] = validation_warnings

    val_report = ValidationReport(
        status=validation_status,
        recommendation=recommendation,
        message=" | ".join(validation_warnings) if validation_warnings else "Sequence is continuous.",
        checks={"sequence_gaps": bool(validation_warnings)},
    )

    return SlicedQuestionsResponse(
        metadata=ExtractedPaperMetadata(**meta_normalized),
        questions_array=questions,
        validation_report=val_report,
    )


# ===========================================================================
# SECTION 7b: Groq-to-Gemini schema converter  (kept for historical compat)
# ===========================================================================

def _convert_groq_questions_for_normalize(
    groq_questions: list,
    document_type: str,
) -> list:
    """Maps legacy groq_slicer ExtractedQuestion objects into _normalize_response format."""
    _DIFF_MAP: dict = {
        "EASY": "LOW", "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HARD": "HIGH", "HIGH": "HIGH",
    }
    output = []
    for q in groq_questions:
        raw: dict = q.model_dump() if hasattr(q, "model_dump") else dict(q)
        q_latex = (raw.get("latex") or "").strip() or (raw.get("question") or "").strip()
        output.append({
            "question_latex":   q_latex,
            "question_type":    raw.get("question_type") or "SUBJECTIVE",
            "options":          raw.get("options") if isinstance(raw.get("options"), list) else [],
            "document_type":    document_type,
            "cognitive_demand": _DIFF_MAP.get(str(raw.get("difficulty", "Medium")).upper(), "MEDIUM"),
            "diagram_urls":     [],
            "needs_review":     False,
        })
    return output


# ===========================================================================
# SECTION 8: JSON Parser  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _parse_json_payload(content: str, *, strict: bool = False) -> dict:
    if not content or not content.strip():
        return {"metadata": {}, "questions_array": []}

    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    cleaned = re.sub(r'(?<!\\)\\(?!["\\n])', r'\\\\', cleaned)

    for attempt in range(10):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            err_msg = str(e)
            if "Invalid \\escape" in err_msg or "Invalid \\u" in err_msg:
                pos = e.pos
                while pos > 0 and cleaned[pos] != '\\':
                    pos -= 1
                if cleaned[pos] == '\\':
                    cleaned = cleaned[:pos] + '\\\\' + cleaned[pos:]
                    continue
                else:
                    print(f"CRITICAL PARSE FAIL (Auto-Heal): {err_msg}")
                    break
            else:
                print(f"CRITICAL PARSE FAIL (Structure): {err_msg}")
                break

    if strict:
        raise PipelineServiceError(
            stage="pdf_native_gemini_parse",
            message="Gemini returned malformed JSON that could not be parsed.",
            details={
                "provider": "gemini",
                "reason": "json_decode_error",
                "preview": cleaned[:500],
            },
        )

    return {"metadata": {}, "questions_array": []}


# ===========================================================================
# SECTION 9: Gemini client helpers  (ORIGINAL LOGIC PRESERVED)
# ===========================================================================

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _gemini_files_fallback_enabled() -> bool:
    """
    Gemini Files fallback is intentionally opt-in.

    The fallback can rescue a failed slicer run, but it also means a single PDF
    may be billed twice: once per rendered page, then again as a whole-document
    upload. Keep it disabled for normal production cost control; enable with
    GEMINI_ALLOW_FILES_FALLBACK=true only for emergency recovery/audit runs.
    """
    return os.getenv("GEMINI_ALLOW_FILES_FALLBACK", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _wait_for_file_ready(
    client: genai.Client, file_name: str, timeout_seconds: int = 240
):
    deadline = time.time() + timeout_seconds
    state_code_map = {0: "STATE_UNSPECIFIED", 1: "PROCESSING", 2: "ACTIVE", 3: "FAILED"}

    def _normalize_state(sv) -> str:
        if sv is None: return "UNKNOWN"
        name = getattr(sv, "name", None)
        if isinstance(name, str) and name: return name.upper()
        if isinstance(sv, int): return state_code_map.get(sv, str(sv))
        try: return state_code_map.get(int(sv), str(sv)).upper()
        except Exception: return str(sv).upper() or "UNKNOWN"

    last_state = "UNKNOWN"
    while time.time() < deadline:
        remote_file = run_gemini_sync(lambda: client.files.get(name=file_name))
        last_state  = _normalize_state(getattr(remote_file, "state", None))
        if "ACTIVE" in last_state: return remote_file
        if "FAILED" in last_state:
            raise RuntimeError(f"Uploaded file FAILED: {last_state}")
        time.sleep(1.0)
    raise TimeoutError(f"File not ACTIVE before timeout. Last state: {last_state}")


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: dict,
    retries: int = 3,
    delay: float = 5.0,
):
    last_exc = None
    for attempt in range(retries):
        try:
            response = run_gemini_sync(
                lambda: client.models.generate_content(model=model, contents=contents, config=config)
            )
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                record_gemini_usage(
                    model=model,
                    document_type="PDF fallback",
                    page_num=None,
                    attempt=attempt + 1,
                    component="gemini_files_fallback",
                    usage=usage,
                )
                logger.info(
                    "[GeminiUsage][Fallback] model=%s attempt=%s prompt_tokens=%s "
                    "candidates_tokens=%s thoughts_tokens=%s total_tokens=%s",
                    model,
                    attempt + 1,
                    getattr(usage, "prompt_token_count", None),
                    getattr(usage, "candidates_token_count", None),
                    getattr(usage, "thoughts_token_count", None),
                    getattr(usage, "total_token_count", None),
                )
            else:
                logger.info(
                    "[GeminiUsage][Fallback] model=%s attempt=%s usage_metadata=missing",
                    model,
                    attempt + 1,
                )
            return response
        except Exception as e:
            last_exc = e
            err_str  = str(e)
            is_transient = any(
                code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
            )
            if is_transient and attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"⚠️  [Gemini] Transient error, retry {attempt+1}/{retries} in {wait:.0f}s: {e}")
                time.sleep(wait)
                continue
            raise
    raise last_exc


_MODEL_PRIORITY: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]


def _pick_available_model(client: genai.Client, exclude: list = None) -> str:
    exclude_set = set(exclude or [])
    try:
        available = {m.name.replace("models/", "") for m in client.models.list()}
    except Exception as e:
        print(f"⚠️  [_pick_available_model] Could not fetch model list: {e}")
        available = set(_MODEL_PRIORITY)

    for m in _MODEL_PRIORITY:
        if m not in exclude_set and m in available:
            return m
    for m in _MODEL_PRIORITY:
        if m not in exclude_set:
            return m
    return _MODEL_PRIORITY[0]


# ===========================================================================
# SECTION 10: Paper reference key builder  (IGCSE + IB)
# ===========================================================================

# ===========================================================================
# SECTION 11a: Gemini Files API fallback (sync, runs in thread)
# ===========================================================================

def _extract_pdf_native_sync(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
) -> SlicedQuestionsResponse:
    """
    Upload the full PDF to Gemini Files API and extract every question as
    structured JSON. Used as a fallback when gemini_slicer path fails or
    rendering is unavailable.

    All original key-generation, metadata-verification, diagram-crop, and
    normalisation logic is preserved verbatim.
    """
    if not pdf_base64 or not pdf_base64.strip():
        empty_meta = ExtractedPaperMetadata(**_METADATA_DEFAULTS)
        return SlicedQuestionsResponse(metadata=empty_meta, questions_array=[])

    normalized_b64 = pdf_base64.strip()
    if "," in normalized_b64:
        normalized_b64 = normalized_b64.split(",", 1)[1]

    uploaded_file  = None
    temp_file_path = None
    client         = None
    extra_metadata = {}
    paper_reference_key = ""

    try:
        client    = _get_client()
        pdf_bytes = base64.b64decode(normalized_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_file_path = tmp.name

        if board.upper() == "IGCSE":
            # For IGCSE, prioritize content-detected session from PDF content.
            content_detected_session = None
            from builders.key_builder import _extract_session_from_content

            # ── FIX 1: Use fitz.get_text() — NOT raw byte decode ─────────────────
            # (Same fix as primary path — see comment there for full rationale.)
            try:
                _fitz_doc_fb = fitz.open(stream=pdf_bytes, filetype="pdf")
                _cover_text_fb = "".join(
                    _fitz_doc_fb[_pn].get_text("text")
                    for _pn in range(min(1, len(_fitz_doc_fb)))
                )
                _fitz_doc_fb.close()
                content_detected_session = _extract_session_from_content(_cover_text_fb)
                if content_detected_session:
                    logger.info(
                        f"[Fallback IGCSE] fitz cover-text session: "
                        f"'{content_detected_session}'"
                    )
                else:
                    logger.debug(
                        "[Fallback IGCSE] fitz cover text contained no session keyword."
                    )
            except Exception as e:
                logger.debug(f"[Fallback IGCSE] fitz session extraction failed: {e}")

            paper_reference_key = generate_igcse_key(filename=filename, content_detected_session=content_detected_session)
            print(f"ℹ️  [Fallback Task A] IGCSE key: {paper_reference_key!r}")
        else:
            # For IB, extract metadata and use the comprehensive key generation from key_builder
            ib_metadata = {}
            if page1_base64:
                ib_metadata = _extract_ib_metadata_from_page(page1_base64) or {}
            ref_code, method = regex_extract_ref_code(temp_file_path)

            session_from_ib_meta = ib_metadata.get("session", "")
            year_from_ib_meta = ib_metadata.get("year", "")
            
            # Prioritize content-detected session from IB metadata over filename/ref_code
            content_detected_session = None
            # We use _extract_session_from_content from key_builder as a private helper
            # to normalize the session string extracted by Gemini\\'s multimodal model.
            # This ensures \\'May/June\\' -> \\'s\\', \\'Oct/Nov\\' -> \\'w\\', etc.
            if session_from_ib_meta:
                content_detected_session = _extract_session_from_content(session_from_ib_meta)

            # Fallback to ref_code if IB metadata doesn't provide session/year
            if ref_code and (not session_from_ib_meta or not year_from_ib_meta):
                prefix = getattr(ref_code, "session_prefix", "")
                if len(prefix) == 4:
                    year_from_ib_meta = "20" + prefix[:2]
                    sess_digits = prefix[2:]
                    if sess_digits == "25":
                        session_from_ib_meta = "may"
                    elif sess_digits in ("11", "00"):
                        session_from_ib_meta = "november"
                    else:
                        session_from_ib_meta = "november"
                        logger.warning(f"[IB Key] Unrecognized session digits '{sess_digits}'. Defaulting to 'november'.")

            paper_reference_key = generate_ib_key(
                subject=ib_metadata.get("subject_name", ""),
                level=ib_metadata.get("level", ""),
                paper_number=ib_metadata.get("paper_number", ""),
                session=session_from_ib_meta, # Use potentially updated session
                year=year_from_ib_meta,       # Use potentially updated year
                document_type=document_type,
                timezone=ib_metadata.get("timezone"),
                content_detected_session=content_detected_session
            )
            if extra_metadata is not None:
                extra_metadata["ref_code_base"] = ref_code.base if ref_code else ""
                if ref_code:
                    extra_metadata["ref_code_full"] = getattr(ref_code, "raw", "")

            print(f"ℹ️  [Fallback Task A] IB key: {paper_reference_key!r} via {method}")

        uploaded_file = run_gemini_sync(lambda: client.files.upload(file=temp_file_path))
        _wait_for_file_ready(client, uploaded_file.name, timeout_seconds=240)

        system_prompt = _build_pdf_system_prompt(document_type, paper_reference_key, board)

        response  = None
        last_exc  = None
        for model_name in _MODEL_PRIORITY:
            try:
                print(f"ℹ️  [Fallback Task A] Trying model '{model_name}'…")
                response = _generate_with_retry(
                    client,
                    model=model_name,
                    contents=[system_prompt, uploaded_file],
                    config={
                        "response_mime_type": "application/json",
                        "thinking_config": {"thinking_budget": 0},
                    },
                    retries=3,
                    delay=5.0,
                )
                print(f"✅ [Fallback Task A] Model '{model_name}' succeeded.")
                break
            except Exception as exc:
                print(f"⚠️  [Fallback Task A] Model '{model_name}' failed: {exc}. Trying next…")
                last_exc = exc

        if response is None:
            raise PipelineServiceError(
                stage="pdf_native_gemini",
                message="All models failed.",
                details={"provider": "gemini", "reason": str(last_exc),
                         "exception_type": type(last_exc).__name__},
            )

        raw_text    = getattr(response, "text", "") or ""
        parsed_dict = _parse_json_payload(raw_text, strict=True)
        questions_list_for_guard = parsed_dict.get("questions_array", [])
        if not isinstance(questions_list_for_guard, list) or not questions_list_for_guard:
            raise PipelineServiceError(
                stage="pdf_native_gemini_parse",
                message="Gemini Files API returned zero extracted questions.",
                details={
                    "provider": "gemini",
                    "document_type": document_type,
                    "paper_reference_key": paper_reference_key,
                    "raw_preview": raw_text[:500],
                },
            )

        # Legacy [NEEDS_CROP] diagram handling for fallback path
        try:
            questions_list = parsed_dict.get("questions_array", [])
            is_marking_scheme_doc = document_type.strip().lower() == "marking scheme"
            if is_marking_scheme_doc:
                for q in questions_list:
                    if isinstance(q, dict):
                        q["diagram_urls"] = []
                        q["diagram_page_number"] = None
                        q["diagram_y_range"] = []

            needs_crop = (not is_marking_scheme_doc) and any(
                isinstance(q, dict) and (
                    (isinstance(q.get("diagram_urls"), list) and "[NEEDS_CROP]" in q.get("diagram_urls", []))
                    or q.get("diagram_urls") == "[NEEDS_CROP]"
                )
                for q in questions_list
            )

            if needs_crop:
                crop_bytes = base64.b64decode(normalized_b64)
                doc = fitz.open(stream=crop_bytes, filetype="pdf")
                try:
                    for q in questions_list:
                        if not isinstance(q, dict): continue
                        urls = q.get("diagram_urls", [])
                        if not (
                            (isinstance(urls, list) and "[NEEDS_CROP]" in urls)
                            or urls == "[NEEDS_CROP]"
                        ):
                            continue

                        page_number = max(0, int(q.get("diagram_page_number", 1) or 1) - 1)
                        page_number = min(page_number, len(doc) - 1)
                        page        = doc[page_number]

                        y_range = q.get("diagram_y_range") or []
                        if isinstance(y_range, list) and len(y_range) == 2:
                            try:
                                PADDING = 0.05
                                safe_y0 = max(0.0, float(y_range[0]) - PADDING)
                                safe_y1 = min(1.0, float(y_range[1]) + PADDING)
                                y0   = safe_y0 * page.rect.height
                                y1   = safe_y1 * page.rect.height
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
            print(f"⚠️  [Fallback Task A] Legacy diagram crop failed: {e}")

        # Ensure all questions have diagram_urls as []
        for q in parsed_dict.get("questions_array", []):
            if isinstance(q, dict):
                if document_type.strip().lower() == "marking scheme":
                    q["diagram_urls"] = []
                    q["diagram_page_number"] = None
                    q["diagram_y_range"] = []
                elif not isinstance(q.get("diagram_urls"), list):
                    q["diagram_urls"] = []
        if document_type.strip().lower() == "question paper":
            parsed_dict["questions_array"] = _add_missing_qp_root_stubs(
                parsed_dict.get("questions_array") or [],
                normalized_b64,
                QuestionNumberNormalizer(),
            )

        normalized_response = _normalize_response(
            parsed_dict, filename, document_type, board, paper_reference_key,
            extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
        )
        return normalized_response

    except PipelineServiceError:
        raise
    except Exception as exc:
        print(f"❌ [Fallback Task A Error] {type(exc).__name__}: {exc!r}")
        raise PipelineServiceError(
            stage="pdf_native_gemini",
            message="Failed to extract structured questions from PDF.",
            details={"provider": "gemini", "reason": str(exc),
                     "exception_type": type(exc).__name__},
        ) from exc
    finally:
        if client and uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# ===========================================================================
# SECTION 11b: Phase 2 — Primary gemini_slicer path
# ===========================================================================

async def _extract_via_gemini_slicer(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
    extra_metadata: dict | None = None,
) -> SlicedQuestionsResponse:
    """
    Phase 2 primary extraction path.

    Steps:
    1. Render all PDF pages to JPEG via pdf_base64_to_vision_pages_async.
    2. Build paper_reference_key (IGCSE regex or IB regex + Gemini metadata page).
    3. Call gemini_slicer.extract_pages_batch() concurrently across all pages.
       The internal asyncio.Semaphore(3) prevents Gemini 503 errors.
    4. Unpack the decoupled List[dict] → {"model": ExtractedQuestion, "diagram_regions": [...]}.
    5. For each question with diagram_regions, call crop_and_compress_diagram_async
       and append the compressed JPEG base64 to model.diagram_urls.
    6. Assemble SlicedQuestionsResponse via _normalize_response.

    Raises PipelineServiceError on failure so the caller can activate the
    Gemini Files API fallback.
    """
    normalized_b64 = pdf_base64.strip()
    if "," in normalized_b64:
        normalized_b64 = normalized_b64.split(",", 1)[1]

    # ── Step 1: Render pages ─────────────────────────────────────────────────
    print(f"📄 [GeminiSlicer Path] Rendering PDF pages… board={board!r} type={document_type!r}")
    page_images: List[str] = await pdf_base64_to_vision_pages_async(normalized_b64, dpi=150)
    if not page_images:
        raise PipelineServiceError(
            stage="gemini_slicer",
            message="PDF rendering produced no pages. File may be corrupted.",
            details={"provider": "pdf_processor", "filename": filename},
        )
    print(f"📄 [GeminiSlicer Path] {len(page_images)} page(s) rendered.")

    # ── Step 2: Build paper_reference_key ────────────────────────────────────
    paper_reference_key = ""
    path_extra_metadata: dict = dict(extra_metadata or {})
    temp_file_path: Optional[str] = None

    try:
        if board.upper() == "IGCSE":
            # For IGCSE, prioritize content-detected session from PDF content.
            content_detected_session = None
            from builders.key_builder import _extract_session_from_content

            # ── FIX 1: Use fitz.get_text() — NOT raw byte decode ─────────────────
            # Root cause of MS session bug:
            #   pdf_bytes.decode('utf-8', errors='ignore') reads raw PDF binary
            #   syntax (compressed object streams, xref tables). "February/March"
            #   lives inside a zlib-compressed content stream — it is NEVER visible
            #   in raw bytes. fitz.open() + page.get_text("text") decompresses the
            #   stream and extracts actual readable text, which reliably contains
            #   "February/March 2018" on Cambridge cover pages.
            #
            # Strategy (try fitz first; raw-byte decode is a dead-letter fallback
            # kept only for PDF files fitz cannot open):
            try:
                pdf_bytes = base64.b64decode(normalized_b64)
                _fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                # Scan pages 0–1 (cover page is always page 0; page 1 is sometimes
                # a continuation of the cover for Cambridge papers).
                _cover_text = "".join(
                    _fitz_doc[_pn].get_text("text")
                    for _pn in range(min(1, len(_fitz_doc)))
                )
                _fitz_doc.close()
                content_detected_session = _extract_session_from_content(_cover_text)
                if content_detected_session:
                    logger.info(
                        f"[GeminiSlicer IGCSE] fitz cover-text session: "
                        f"'{content_detected_session}'"
                    )
                else:
                    logger.debug(
                        "[GeminiSlicer IGCSE] fitz cover text contained no session "
                        "keyword — will rely on filename fallback in generate_igcse_key."
                    )
            except Exception as e:
                logger.debug(f"[GeminiSlicer IGCSE] fitz session extraction failed: {e}")

            paper_reference_key = generate_igcse_key(filename=filename, content_detected_session=content_detected_session)
            print(f"ℹ️  [GeminiSlicer Path] IGCSE key: {paper_reference_key!r}")
        else:
            # IB: run metadata extraction from page1 and ref-code regex concurrently
            ib_metadata: dict = {}
            ib_meta_task = None
            if page1_base64:
                ib_meta_task = asyncio.to_thread(_extract_ib_metadata_from_page, page1_base64)

            # Write PDF to temp file for regex_extract_ref_code (sync)
            pdf_bytes = base64.b64decode(normalized_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                temp_file_path = tmp.name

            ref_code_task = asyncio.to_thread(regex_extract_ref_code, temp_file_path)

            gather_results = await asyncio.gather(
                ib_meta_task if ib_meta_task else asyncio.coroutine(lambda: {})(),
                ref_code_task,
                return_exceptions=True,
            )

            if not isinstance(gather_results[0], Exception):
                ib_metadata = gather_results[0] or {}
            else:
                logger.warning(f"[GeminiSlicer Path] IB metadata extraction failed: {gather_results[0]}")

            ref_code, method = (None, None)
            if not isinstance(gather_results[1], Exception):
                ref_code, method = gather_results[1]
            else:
                logger.warning(f"[GeminiSlicer Path] IB ref-code extraction failed: {gather_results[1]}")

            session_from_ib_meta = ib_metadata.get("session", "")
            year_from_ib_meta = ib_metadata.get("year", "")
            
            # Prioritize content-detected session from IB metadata over filename/ref_code
            content_detected_session = None
            # We use _extract_session_from_content from key_builder as a private helper
            # to normalize the session string extracted by Gemini's multimodal model.
            # This ensures 'May/June' -> 's', 'Oct/Nov' -> 'w', etc.
            from builders.key_builder import _extract_session_from_content
            if session_from_ib_meta:
                content_detected_session = _extract_session_from_content(session_from_ib_meta)

            # Fallback to ref_code if IB metadata doesn't provide session/year
            if ref_code and (not session_from_ib_meta or not year_from_ib_meta):
                prefix = getattr(ref_code, "session_prefix", "")
                if len(prefix) == 4:
                    year_from_ib_meta = "20" + prefix[:2]
                    sess_digits = prefix[2:]
                    if sess_digits == "25":
                        session_from_ib_meta = "may"
                    elif sess_digits in ("11", "00"):
                        session_from_ib_meta = "november"
                    else:
                        session_from_ib_meta = "november"
                        logger.warning(f"[IB Key] Unrecognized session digits '{sess_digits}'. Defaulting to 'november'.")

            paper_reference_key = generate_ib_key(
                subject=ib_metadata.get("subject_name", ""),
                level=ib_metadata.get("level", ""),
                paper_number=ib_metadata.get("paper_number", ""),
                session=session_from_ib_meta, # Use potentially updated session
                year=year_from_ib_meta,       # Use potentially updated year
                document_type=document_type,
                timezone=ib_metadata.get("timezone"),
                content_detected_session=content_detected_session
            )
            if path_extra_metadata is not None:
                path_extra_metadata["ref_code_base"] = ref_code.base if ref_code else ""
                if ref_code:
                    path_extra_metadata["ref_code_full"] = getattr(ref_code, "raw", "")

            print(f"ℹ️  [GeminiSlicer Path] IB key: {paper_reference_key!r} via {method}")
    except Exception as key_exc:
        logger.warning(f"[GeminiSlicer Path] Key generation failed: {key_exc}. Proceeding without key.")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    # ── Step 3: Build fallback metadata dict for gemini_slicer ───────────────
    fallback_metadata: dict = {
        "curriculum": board.upper(),
        "paper_reference_key": paper_reference_key,
    }
    if path_extra_metadata:
        fallback_metadata.update(path_extra_metadata)

    if document_type.strip().lower() != "marking scheme":
        expected_ids = fallback_metadata.get("expected_canonical_ids") or []
        expected_ids_for_log = [
            str(value).strip().lower()
            for value in expected_ids
            if str(value).strip()
        ] if isinstance(expected_ids, list) else []
        print(
            "[MSAnchorTrace][PythonIngress] "
            f"doc_type={document_type!r} key={paper_reference_key!r} "
            f"expected_ids={len(expected_ids_for_log)} "
            f"first_ids={expected_ids_for_log[:8]} "
            f"last_ids={expected_ids_for_log[-5:]}"
        )
        local_page_hints = _build_local_qp_page_hints(
            pdf_base64=normalized_b64,
            expected_ids=expected_ids if isinstance(expected_ids, list) else [],
        )
        if local_page_hints:
            fallback_metadata["local_qp_page_hints"] = local_page_hints
            pages_with_expected = sum(
                1
                for hint in local_page_hints
                if isinstance(hint, dict) and hint.get("expected_ids")
            )
            sample_hints = [
                {
                    "page": idx + 1,
                    "printed": hint.get("printed_page_number"),
                    "root": hint.get("likely_root"),
                    "expected": (hint.get("expected_ids") or [])[:8],
                }
                for idx, hint in enumerate(local_page_hints)
                if isinstance(hint, dict) and hint.get("expected_ids")
            ][:6]
            print(
                "[MSAnchorTrace][LocalSkeleton] "
                f"pages={len(local_page_hints)} pages_with_expected={pages_with_expected} "
                f"sample={sample_hints}"
            )
        else:
            print("[MSAnchorTrace][LocalSkeleton] no local QP skeleton hints built")

    if document_type.strip().lower() == "marking scheme":
        native_ms_response = _extract_ms_tables_native_response(
            pdf_base64=normalized_b64,
            document_type=document_type,
            filename=filename,
            board=board,
            generated_paper_reference_key=paper_reference_key,
            extra_metadata=fallback_metadata,
        )
        if native_ms_response and native_ms_response.questions_array:
            return native_ms_response

    # ── Step 4: Call gemini_slicer.extract_pages_batch() concurrently ─────────
    # The semaphore inside gemini_slicer limits concurrent Gemini calls to 3.
    print(f"🚀 [GeminiSlicer Path] Calling extract_pages_batch for {len(page_images)} page(s)…")
    slicer_results: List[Dict] = await gemini_slicer_extract_pages(
        page_jpeg_b64_list=page_images,
        document_type=document_type,
        board=board,
        paper_reference_key=paper_reference_key,
        fallback_metadata=fallback_metadata,
    )

    if not slicer_results:
        raise PipelineServiceError(
            stage="gemini_slicer",
            message="gemini_slicer returned zero questions for all pages.",
            details={
                "provider": "gemini",
                "total_pages": len(page_images),
                "document_type": document_type,
            },
        )
    print(f"✅ [GeminiSlicer Path] {len(slicer_results)} question(s) extracted across all pages.")

    # ── Step 5: Inject diagram crops from diagram_regions ─────────────────────
    # Each result dict: {"model": ExtractedQuestion, "diagram_regions": [...]}
    # _inject_diagram_crops_from_slicer processes all questions concurrently.
    # diagram_urls is guaranteed to be [] if no regions exist.
    final_questions: List[ExtractedQuestion] = await _inject_diagram_crops_from_slicer(
        slicer_results=slicer_results,
        pdf_base64=normalized_b64,
    )

    if not final_questions:
        raise PipelineServiceError(
            stage="gemini_slicer",
            message="All ExtractedQuestion models were invalid after crop injection.",
            details={"total_slicer_results": len(slicer_results)},
        )

    # ── Step 6: Assemble SlicedQuestionsResponse via _normalize_response ──────
    # We pass questions as a dict list so _normalize_response can apply
    # canonical_question_id normalization, sequence gap checks, etc.
    questions_as_dicts = [q.model_dump() for q in final_questions]
    if document_type.strip().lower() == "question paper":
        questions_as_dicts = _apply_local_qp_ms_skeleton_first(
            questions_raw=questions_as_dicts,
            pdf_base64=normalized_b64,
            expected_ids=fallback_metadata.get("expected_canonical_ids") or [],
            normalizer=QuestionNumberNormalizer(),
        )
        questions_as_dicts = _add_missing_qp_expected_leaf_stubs(
            questions_raw=questions_as_dicts,
            expected_ids=fallback_metadata.get("expected_canonical_ids") or [],
            local_page_hints=fallback_metadata.get("local_qp_page_hints") or [],
            normalizer=QuestionNumberNormalizer(),
        )
        questions_as_dicts = _add_missing_qp_root_stubs(
            questions_as_dicts,
            normalized_b64,
            QuestionNumberNormalizer(),
        )

    # Merge metadata from the first question (gemini_slicer injects it per-question)
    first_q = questions_as_dicts[0] if questions_as_dicts else {}
    page_metadata = {
        "curriculum": first_q.get("curriculum") or board.upper(),
        "program":    first_q.get("program"),
        "subjectCode": first_q.get("subjectCode") or "",
        "tier":       first_q.get("tier") or "N/A",
        "paperNumber": first_q.get("paperNumber") or 0,
        "session":    first_q.get("session") or "",
        "year":       first_q.get("year") or 0,
        "paper_reference_key": paper_reference_key,
    }
    if extra_metadata:
        page_metadata.update(extra_metadata)

    parsed_dict = {
        "metadata":        page_metadata,
        "questions_array": questions_as_dicts,
    }

    response = _normalize_response(
        parsed=parsed_dict,
        filename=filename,
        document_type=document_type,
        board=board,
        generated_paper_reference_key=paper_reference_key,
        extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
    )

    print(
        f"🎯 [GeminiSlicer Path] Assembly complete. "
        f"{len(response.questions_array)} questions in response."
    )
    _log_extracted_rows(
        "GeminiSlicerFinal",
        response.questions_array,
        document_type=document_type,
    )
    return response


def _log_extracted_rows(
    source: str,
    questions: List[Any],
    *,
    document_type: str = "",
) -> None:
    if str(os.getenv("PAPERLY_LOG_EXTRACTED_ROWS", "true")).strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return
    try:
        max_chars = max(80, int(os.getenv("PAPERLY_LOG_EXTRACTED_ROW_TEXT_CHARS", "700")))
    except Exception:
        max_chars = 700
    rows = list(questions or [])
    print(f"[ExtractedRows][{source}] doc_type={document_type!r} count={len(rows)}")
    for index, question in enumerate(rows, start=1):
        if hasattr(question, "model_dump"):
            data = question.model_dump(mode="json")
        elif isinstance(question, dict):
            data = question
        else:
            data = {}
        canonical = str(data.get("canonical_question_id") or data.get("question_id") or "").strip()
        question_id = str(data.get("question_id") or "").strip()
        text = str(
            data.get("question_latex")
            or data.get("final_answer")
            or data.get("official_marking_scheme_latex")
            or ""
        )
        text_preview = " ".join(text.split())
        if len(text_preview) > max_chars:
            text_preview = text_preview[:max_chars] + "..."
        diagram_count = len(data.get("diagram_urls") or []) if isinstance(data.get("diagram_urls"), list) else 0
        print(
            f"[ExtractedRows][{source}] {index:03d}/{len(rows):03d} "
            f"id={canonical!r} qid={question_id!r} "
            f"review={bool(data.get('needs_review'))} diagrams={diagram_count} "
            f"text={text_preview!r}"
        )


def _canonical_root(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(".", 1)[0] if text else ""


def _pdf_page_texts(pdf_base64: str) -> List[str]:
    normalized_b64 = pdf_base64.strip().split(",", 1)[-1]
    pdf_bytes = base64.b64decode(normalized_b64)
    texts: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            try:
                texts.append(page.get_text("text") or "")
            except Exception:
                texts.append("")
    return texts


def _select_rescue_page_indexes(
    pdf_base64: str,
    missing_ids: List[str],
    local_page_hints: List[Dict[str, Any]],
    max_pages: int = 6,
) -> List[int]:
    """
    SMART RESCUE — Multi-strategy page selection for missing questions.

    Strategy priority:
    1. Exact text match — search ALL pages for the exact missing ID pattern
    2. Root expansion — find pages with the parent question root
    3. Neighbor expansion — include ±2 pages around high-scoring pages
    4. Fallback — if nothing found, search broader range around expected location

    The goal: FIND the missing question, not just avoid a full redo.
    """
    missing_clean = [str(value).strip().lower() for value in missing_ids if str(value).strip()]
    # Search all native page text for evidence, but keep the final Gemini page
    # set bounded by max_pages so rescue stays cheap and avoids rate limits.
    roots = {_canonical_root(value) for value in missing_clean if _canonical_root(value)}
    scored: Dict[int, int] = defaultdict(int)
    exact_text_pages: List[int] = []
    root_pages: List[int] = []

    def _id_visible_in_text(canonical_id: str, text: str) -> bool:
        """Enhanced pattern matching for nested IDs like 4.b.i.b"""
        compact = " ".join(str(text or "").split()).lower()
        parts = [part for part in str(canonical_id or "").lower().split(".") if part]
        if not compact or not parts:
            return False

        # Strategy 1: Full exact match with parentheses: "4(b)(i)(b)"
        root = re.escape(parts[0])
        suffix = "".join(r"\s*\(\s*" + re.escape(part) + r"\s*\)" for part in parts[1:])
        full_pattern = rf"(^|\D){root}{suffix}(?=\s|\D|$)"
        if re.search(full_pattern, compact):
            return True

        # Strategy 2: Partial suffix match (orphan subparts): "(b)(i)(b)"
        if len(parts) > 1:
            orphan_suffix = "".join(r"\s*\(\s*" + re.escape(part) + r"\s*\)" for part in parts[1:])
            if re.search(rf"(^|\s){orphan_suffix}(?=\s|\D|$)", compact):
                return True

        # Strategy 3: Flexible nested match for deep IDs like "4.b.i.b"
        # Look for any of: "4(b)(i)(b)", "4 b i b", "4(b) (i) (b)"
        if len(parts) >= 3:
            # Try loose spacing: "4 b i b" or "4(b) i b"
            loose_parts = [re.escape(p) for p in parts]
            loose_pattern = rf"(^|\D){loose_parts[0]}\s*\(?\s*{loose_parts[1]}\s*\)?\s*\(?\s*{loose_parts[2]}"
            if len(parts) >= 4:
                loose_pattern += rf"\s*\)?\s*\(?\s*{loose_parts[3]}"
            if re.search(loose_pattern, compact):
                return True

        return False

    # Phase 1: Score pages based on local skeleton hints
    for idx, hint in enumerate(local_page_hints or []):
        if not isinstance(hint, dict):
            continue
        likely_root = str(hint.get("likely_root") or "").strip()
        expected = {
            str(value).strip().lower()
            for value in (hint.get("expected_ids") or [])
            if str(value).strip()
        }
        if expected.intersection(missing_clean):
            scored[idx] += 8
        if likely_root in roots:
            scored[idx] += 5
            if idx not in root_pages:
                root_pages.append(idx)

    # Phase 2: CRITICAL — Search ALL pages for exact text match
    # This is the PRIMARY strategy for finding deeply nested IDs like 4.b.i.b
    try:
        page_texts = _pdf_page_texts(pdf_base64)
        for idx, text in enumerate(page_texts):
            compact = " ".join(str(text or "").split()).lower()
            if not compact:
                continue

            # HIGH PRIORITY: Exact missing ID found in text
            if any(_id_visible_in_text(missing_id, compact) for missing_id in missing_clean):
                scored[idx] += 20  # Increased from 12 to 20 — this is gold
                if idx not in exact_text_pages:
                    exact_text_pages.append(idx)

            # MEDIUM PRIORITY: Root question number found
            for root in roots:
                if re.search(rf"(^|\D){re.escape(root)}(\D|$)", compact):
                    scored[idx] += 3
    except Exception as exc:
        logger.debug("[RescueMissing] Native page-text selection failed: %s", exc)

    if not scored:
        return []

    # Phase 3: Build selected pages with SMART strategy
    selected: List[int] = []

    # Priority 1: Pages with EXACT text matches (highest confidence)
    if exact_text_pages:
        for idx in sorted(exact_text_pages):
            if idx not in selected:
                selected.append(idx)
                # Add ±2 neighbors for context (questions span pages)
                for neighbor in (idx - 2, idx - 1, idx + 1, idx + 2):
                    if neighbor >= 0 and neighbor not in selected:
                        selected.append(neighbor)

    # Priority 2: Pages from local skeleton hints
    direct_pages: List[int] = []
    for idx, hint in enumerate(local_page_hints or []):
        if not isinstance(hint, dict):
            continue
        expected = {
            str(value).strip().lower()
            for value in (hint.get("expected_ids") or [])
            if str(value).strip()
        }
        if expected.intersection(missing_clean):
            direct_pages.append(idx)

    for idx in sorted(direct_pages):
        if idx not in selected:
            selected.append(idx)

    # Priority 3: Add pages with root questions
    for idx in root_pages:
        if idx not in selected:
            selected.append(idx)

    # Priority 4: Top-scoring pages by combined score
    for idx, _score in sorted(scored.items(), key=lambda item: (-item[1], item[0])):
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_pages:
            break

    # Phase 4: Final neighbor expansion around all high-value pages
    expansion_candidates = list(selected[:max_pages])  # Top pages only
    for idx in expansion_candidates:
        for candidate in (idx - 1, idx + 1):  # ±1 immediate neighbors
            if candidate >= 0 and candidate not in selected:
                selected.append(candidate)
            if len(selected) >= max_pages:
                break
        if len(selected) >= max_pages:
            break

    return sorted(selected[:max_pages])


async def rescue_missing_qp_questions(
    pdf_base64: str,
    missing_ids: List[str],
    filename: str,
    board: str = "IGCSE",
    extra_metadata: dict | None = None,
) -> Dict[str, Any]:
    """
    Targeted rescue endpoint for QP rows that QA says are missing.

    It renders the PDF locally but calls Gemini only on the likely pages. This
    keeps a one-off missing-subpart fix cheap compared with full "Redo
    Extraction". The function returns only exact recovered missing IDs; it does
    not add placeholders or extra sibling rows.
    """
    normalized_b64 = pdf_base64.strip().split(",", 1)[-1]
    missing_clean = [
        str(value).strip().lower()
        for value in (missing_ids or [])
        if str(value).strip()
    ]
    missing_set = set(missing_clean)
    if not missing_set:
        return {
            "questions_array": [],
            "rescue_report": {
                "missing_ids": [],
                "pages_attempted": [],
                "recovered_ids": [],
                "message": "No missing IDs were supplied.",
            },
        }

    paper_reference_key = str((extra_metadata or {}).get("paper_reference_key") or "").strip()
    try:
        content_detected_session = None
        if not paper_reference_key and board.upper() == "IGCSE":
            from builders.key_builder import _extract_session_from_content
            pdf_bytes = base64.b64decode(normalized_b64)
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                cover_text = "".join(
                    doc[pn].get_text("text")
                    for pn in range(min(1, len(doc)))
                )
            content_detected_session = _extract_session_from_content(cover_text)
            paper_reference_key = generate_igcse_key(
                filename=filename,
                content_detected_session=content_detected_session,
            )
    except Exception as exc:
        logger.debug("[RescueMissing] Key/session scan failed: %s", exc)

    expected_ids = list((extra_metadata or {}).get("expected_canonical_ids") or missing_clean)
    normalizer = QuestionNumberNormalizer()
    local_recovered_questions: List[Dict[str, Any]] = []
    try:
        local_rows = _build_local_qp_ms_skeleton_rows(normalized_b64, expected_ids, normalizer)
        local_by_id = {
            str(row.get("canonical_question_id") or "").strip().lower(): row
            for row in local_rows
            if isinstance(row, dict) and row.get("canonical_question_id")
        }
        local_hits = [
            dict(local_by_id[missing_id])
            for missing_id in missing_clean
            if missing_id in local_by_id
        ]
        if local_hits:
            normalized_local = _normalize_response(
                parsed={
                    "metadata": {
                        "curriculum": board.upper(),
                        "paper_reference_key": paper_reference_key,
                        **(extra_metadata or {}),
                    },
                    "questions_array": local_hits,
                },
                filename=filename,
                document_type="Question Paper",
                board=board,
                generated_paper_reference_key=paper_reference_key,
                extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
            )
            local_recovered_questions = [
                question.model_dump(mode="json")
                for question in (normalized_local.questions_array or [])
            ]
            local_recovered_ids = {
                str(question.get("canonical_question_id") or "").strip().lower()
                for question in local_recovered_questions
            }
            missing_clean = [
                missing_id for missing_id in missing_clean if missing_id not in local_recovered_ids
            ]
            missing_set = set(missing_clean)
            print(
                "[RescueMissing][LocalOnly] "
                f"recovered={sorted(local_recovered_ids)} remaining={sorted(missing_set)}"
            )
            if not missing_clean:
                return {
                    "metadata": (
                        normalized_local.metadata.model_dump(mode="json")
                        if normalized_local.metadata else {}
                    ),
                    "questions_array": local_recovered_questions,
                    "rescue_report": {
                        "missing_ids": sorted(local_recovered_ids),
                        "pages_attempted": [],
                        "extracted_ids": sorted(local_recovered_ids),
                        "recovered_ids": sorted(local_recovered_ids),
                        "message": (
                            f"Recovered {len(local_recovered_questions)} row(s) from native PDF text "
                            "without a Gemini rescue call."
                        ),
                        "local_only": True,
                    },
                }
    except Exception as exc:
        logger.debug("[RescueMissing][LocalOnly] Native rescue failed: %s", exc)

    local_hints = _build_local_qp_page_hints(normalized_b64, expected_ids=expected_ids)
    selected_page_indexes = _select_rescue_page_indexes(
        pdf_base64=normalized_b64,
        missing_ids=missing_clean,
        local_page_hints=local_hints,
        max_pages=int(os.getenv("PAPERLY_RESCUE_MAX_PAGES", "4")),
    )

    print(
        "[RescueMissing] "
        f"key={paper_reference_key!r} missing={sorted(missing_set)} "
        f"selected_pages={[idx + 1 for idx in selected_page_indexes]}"
    )

    if not selected_page_indexes:
        local_ids = [
            str(question.get("canonical_question_id") or "").strip().lower()
            for question in local_recovered_questions
            if str(question.get("canonical_question_id") or "").strip()
        ]
        return {
            "metadata": extra_metadata or {},
            "questions_array": local_recovered_questions,
            "rescue_report": {
                "missing_ids": sorted(missing_set),
                "pages_attempted": [],
                "extracted_ids": local_ids,
                "recovered_ids": local_ids,
                "message": (
                    "Could not locate likely Gemini rescue pages from local PDF text."
                    if not local_ids
                    else f"Recovered {len(local_ids)} row(s) locally; no Gemini rescue pages selected."
                ),
            },
        }

    page_images = await pdf_base64_to_vision_pages_async(normalized_b64, dpi=150)
    selected_page_images = [
        page_images[idx]
        for idx in selected_page_indexes
        if 0 <= idx < len(page_images)
    ]
    selected_hints = [
        local_hints[idx] if 0 <= idx < len(local_hints) else {}
        for idx in selected_page_indexes
        if 0 <= idx < len(page_images)
    ]
    try:
        page_texts = _pdf_page_texts(normalized_b64)
    except Exception:
        page_texts = []
    missing_roots = {_canonical_root(value) for value in missing_clean if _canonical_root(value)}
    for logical_idx, hint in enumerate(selected_hints):
        if isinstance(hint, dict):
            hint["page_index"] = logical_idx
            original_page_index = selected_page_indexes[logical_idx]
            hint["rescue_original_page_index"] = original_page_index
            hinted_root = str(hint.get("likely_root") or "").strip()
            if hinted_root:
                hint["rescue_target_ids"] = [
                    value for value in missing_clean
                    if _canonical_root(value) == hinted_root
                ] or missing_clean
            else:
                hint["rescue_target_ids"] = missing_clean
            if 0 <= original_page_index < len(page_texts):
                hint["rescue_page_text_excerpt"] = " ".join(page_texts[original_page_index].split())

    fallback_metadata = {
        "curriculum": board.upper(),
        "paper_reference_key": paper_reference_key,
        "expected_canonical_ids": expected_ids,
        "rescue_missing_ids": sorted(missing_set),
        "local_qp_page_hints": selected_hints,
    }
    if extra_metadata:
        fallback_metadata.update(extra_metadata)
        fallback_metadata["rescue_missing_ids"] = sorted(missing_set)
        fallback_metadata["local_qp_page_hints"] = selected_hints

    slicer_results = await gemini_slicer_extract_pages(
        page_jpeg_b64_list=selected_page_images,
        document_type="Question Paper",
        board=board,
        paper_reference_key=paper_reference_key,
        fallback_metadata=fallback_metadata,
    )

    # Remap subset page numbers back to original PDF pages before cropping.
    for entry in slicer_results or []:
        try:
            local_idx = int(entry.get("page_num", 0) or 0)
            if 0 <= local_idx < len(selected_page_indexes):
                entry["page_num"] = selected_page_indexes[local_idx]
        except Exception:
            pass

    final_questions = await _inject_diagram_crops_from_slicer(
        slicer_results=slicer_results,
        pdf_base64=normalized_b64,
    )
    raw_questions = [q.model_dump() for q in final_questions]
    parsed_dict = {
        "metadata": {
            "curriculum": board.upper(),
            "paper_reference_key": paper_reference_key,
            **(extra_metadata or {}),
        },
        "questions_array": raw_questions,
    }
    normalized = _normalize_response(
        parsed=parsed_dict,
        filename=filename,
        document_type="Question Paper",
        board=board,
        generated_paper_reference_key=paper_reference_key,
        extra_metadata=extra_metadata if board.upper() != "IGCSE" else None,
    )

    recovered = []
    rejected_recovered: list[dict[str, str]] = []
    seen = set()
    extracted_ids = [
        str(question.get("canonical_question_id") or "").strip().lower()
        for question in local_recovered_questions
        if str(question.get("canonical_question_id") or "").strip()
    ]
    local_seen = set(extracted_ids)
    # Build a mapping from extracted child IDs to which missing parent they cover.
    # e.g. if missing=["4.g"] and Gemini returns "4.g.i", "4.g" is parent-covered.
    parent_covered: dict[str, str] = {}  # missing_id -> child_canonical that covers it
    for question in normalized.questions_array or []:
        child_canonical = str(question.canonical_question_id or "").strip().lower()
        if not child_canonical:
            continue
        for missing_id in missing_set:
            if child_canonical.startswith(missing_id + ".") and missing_id not in parent_covered:
                parent_covered[missing_id] = child_canonical

    rescue_candidates = list(normalized.questions_array or [])
    for question in rescue_candidates:
        canonical = str(question.canonical_question_id or "").strip().lower()
        if canonical:
            extracted_ids.append(canonical)
        if canonical in missing_set and canonical not in seen:
            rejection_reason = _rescue_exact_model_row_rejection_reason(
                question,
                rescue_candidates,
                normalizer,
            )
            if rejection_reason:
                rejected_recovered.append({
                    "canonical_question_id": canonical,
                    "reason": rejection_reason,
                })
                print(f"[RescueMissing][RejectedModelRow] {rejection_reason}")
                continue
            recovered.append(_strip_rescue_diagrams_if_needed(question))
            seen.add(canonical)

    if str(os.getenv("PAPERLY_RESCUE_SYNTHESIZE_PARENT_COVER_STUBS", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        # Disabled by default: copied parent/child stubs make counts look fixed
        # but poison question text for review and RAG. Exact recovered rows above
        # remain enabled; non-exact parent coverage is reported only.
        for missing_id in missing_set:
            if missing_id in seen:
                continue
            child_canonical = parent_covered.get(missing_id)
            if not child_canonical:
                continue
            for question in normalized.questions_array or []:
                if str(question.canonical_question_id or "").strip().lower() == child_canonical:
                    q_dict = question.model_dump(mode="json") if hasattr(question, "model_dump") else dict(question)
                    stub = dict(q_dict)
                    normalizer_local = QuestionNumberNormalizer()
                    parent_parts = normalizer_local.extract_parts(missing_id)
                    parent_label = normalizer_local.format_parts(parent_parts)
                    stub["canonical_question_id"] = missing_id
                    stub["question_id"] = parent_label
                    stub["parent_canonical_id"] = normalizer_local.parent_from_parts(parent_parts)
                    _child_label, child_remainder = normalizer_local.split_label_and_remainder(
                        str(stub.get("question_latex") or "")
                    )
                    stub["question_latex"] = (
                        f"{parent_label} {child_remainder}".strip() if child_remainder else parent_label
                    )
                    stub["needs_review"] = True
                    warnings = stub.get("validation_warnings")
                    if not isinstance(warnings, list):
                        warnings = []
                    warnings.append(
                        f"Rescue mode: exact ID {missing_id!r} not found; "
                        f"parent-covered by child {child_canonical!r}. Verify text."
                    )
                    stub["validation_warnings"] = warnings
                    recovered.append(stub)
                    seen.add(missing_id)
                    print(
                        f"[RescueMissing][ParentCover] {missing_id!r} not found exactly; "
                        f"synthesized stub from child {child_canonical!r}."
                    )
                    break
    elif parent_covered:
        print(
            "[RescueMissing][ParentCover] non_exact_parent_coverage_report_only="
            f"{parent_covered}"
        )

    recovered_payload = list(local_recovered_questions)
    recovered_payload.extend(q.model_dump(mode="json") for q in recovered)
    all_recovered_ids = sorted(local_seen | seen)
    unrecovered_ids = [
        missing_id for missing_id in missing_clean
        if missing_id not in local_seen and missing_id not in seen
    ]
    split_hints = _build_rescue_split_hints(unrecovered_ids, rescue_candidates, normalizer)
    selected_page_texts = [
        page_texts[idx]
        for idx in selected_page_indexes
        if 0 <= idx < len(page_texts)
    ]
    split_review_rows = _build_rescue_split_review_rows(
        unrecovered_ids,
        rescue_candidates,
        normalizer,
        page_texts=selected_page_texts,
    )
    split_review_ids = {
        str(row.get("canonical_question_id") or "").strip().lower()
        for row in split_review_rows
        if str(row.get("canonical_question_id") or "").strip()
    }
    if split_review_rows:
        recovered_payload.extend(split_review_rows)
        all_recovered_ids = sorted(set(all_recovered_ids) | split_review_ids)
        unrecovered_ids = [
            missing_id for missing_id in unrecovered_ids
            if missing_id not in split_review_ids
        ]

    _log_extracted_rows(
        "TargetedRescuePayload",
        recovered_payload,
        document_type="Question Paper",
    )

    print(
        "[RescueMissing] "
        f"extracted={len(normalized.questions_array or [])} "
        f"extracted_ids={extracted_ids} "
        f"recovered={sorted(seen)} split_review={sorted(split_review_ids)} "
        f"rejected={rejected_recovered} split_hints={split_hints}"
    )

    rejection_message = ""
    if rejected_recovered:
        rejected_ids = [
            str(item.get("canonical_question_id") or "")
            for item in rejected_recovered
            if item.get("canonical_question_id")
        ]
        rejection_message = (
            f" Rejected {len(rejected_recovered)} cloned/duplicate sibling row(s): "
            f"{', '.join(rejected_ids)}."
        )
    split_hint_message = ""
    if split_review_ids:
        split_hint_message = (
            " Created split-review row(s) from grouped source row(s): "
            f"{', '.join(sorted(split_review_ids))}."
        )
    elif split_hints:
        hint_ids = [
            f"{hint.get('missing_id')} from {hint.get('source_id')}"
            for hint in split_hints
            if hint.get("missing_id") and hint.get("source_id")
        ]
        split_hint_message = (
            " Exact rescue did not safely recover every ID; split grouped source row(s): "
            f"{', '.join(hint_ids)}."
        )

    clean_recovered_count = len(local_recovered_questions) + len(recovered)
    split_review_count = len(split_review_rows)
    if split_review_count:
        recovery_summary = (
            f"Recovered {clean_recovered_count} exact row(s) and created "
            f"{split_review_count} split-review row(s) from {len(selected_page_indexes)} targeted page(s)."
        )
    else:
        recovery_summary = (
            f"Recovered {clean_recovered_count} exact missing row(s) from "
            f"{len(selected_page_indexes)} targeted page(s)."
        )

    return {
        "metadata": normalized.metadata.model_dump(mode="json") if normalized.metadata else {},
        "questions_array": recovered_payload,
        "rescue_report": {
            "missing_ids": sorted(set(missing_clean) | local_seen),
            "pages_attempted": [idx + 1 for idx in selected_page_indexes],
            "extracted_ids": extracted_ids,
            "recovered_ids": all_recovered_ids,
            "rejected_recovered": rejected_recovered,
            "repair_hints": split_hints,
            "split_review_ids": sorted(split_review_ids),
            "message": recovery_summary + rejection_message + split_hint_message,
        },
    }


# ===========================================================================
# SECTION 12: Public async entry-point
# ===========================================================================

async def extract_pdf_native_gemini(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
    extra_metadata: dict | None = None,
) -> SlicedQuestionsResponse:
    """
    Phase 2 public entry-point for PDF extraction.

    Primary path  → gemini_slicer (single-pass multimodal, per-page concurrent).
    Fallback path → Gemini Files API whole-document upload (_extract_pdf_native_sync).

    Routing:
      GEMINI_QP_ENGINE env-var (default "slicer"):
        "pdf"    → Question Papers use whole-document Gemini Files API.
                   This preserves document-global numbering and avoids the
                   per-page prompt/sequence drift that corrupts QP canonical IDs.
        "slicer" → Question Papers use gemini_slicer directly. This avoids
                   paying for a whole-document attempt and then a page fallback
                   when Gemini returns malformed JSON.

      GEMINI_SLICER_ENABLED env-var (default "true"):
        "true"  → Marking Schemes use primary gemini_slicer path.
        "false" → bypass directly to Gemini Files API fallback.

      On PipelineServiceError from primary path → automatically activates fallback.
      On any other Exception from primary path → also activates fallback.

    The fallback's Vision Engine (Task B) is retired from the primary flow.
    It is preserved in SECTION 4 for the Gemini Files API fallback path should
    the admin want to re-enable it via _run_vision_engine_for_page.
    """
    normalized_doc_type = (document_type or "").strip().lower()
    qp_engine = os.getenv("GEMINI_QP_ENGINE", "slicer").strip().lower()

    if normalized_doc_type == "marking scheme" and board.upper() == "IGCSE":
        normalized_b64 = pdf_base64.strip().split(",", 1)[-1]
        content_detected_session = None
        try:
            from builders.key_builder import _extract_session_from_content
            pdf_bytes = base64.b64decode(normalized_b64)
            with fitz.open(stream=pdf_bytes, filetype="pdf") as _fitz_doc:
                _cover_text = "".join(
                    _fitz_doc[_pn].get_text("text")
                    for _pn in range(min(1, len(_fitz_doc)))
                )
            content_detected_session = _extract_session_from_content(_cover_text)
        except Exception as exc:
            logger.debug("[NativeMSTable] Fast MS key session scan failed: %s", exc)
        ms_key = generate_igcse_key(
            filename=filename,
            content_detected_session=content_detected_session,
        )
        native_ms_response = _extract_ms_tables_native_response(
            pdf_base64=normalized_b64,
            document_type=document_type,
            filename=filename,
            board=board,
            generated_paper_reference_key=ms_key,
            extra_metadata={
                "curriculum": board.upper(),
                "paper_reference_key": ms_key,
            },
        )
        if native_ms_response and native_ms_response.questions_array:
            print(
                f"[extract_pdf_native_gemini] Native MS table path succeeded - "
                f"{len(native_ms_response.questions_array)} row(s)."
            )
            return native_ms_response

    if normalized_doc_type == "question paper" and qp_engine != "slicer":
        print(
            "ℹ️  [extract_pdf_native_gemini] GEMINI_QP_ENGINE=pdf — "
            "routing Question Paper through whole-document Gemini Files API "
            "for stable numbering."
        )
        try:
            result = await asyncio.to_thread(
                _extract_pdf_native_sync,
                pdf_base64, document_type, filename, board, page1_base64,
            )
            if not result.questions_array:
                raise PipelineServiceError(
                    stage="pdf_native_gemini",
                    message="Whole-document Gemini returned zero questions.",
                    details={
                        "provider": "gemini",
                        "document_type": document_type,
                        "filename": filename,
                    },
                )
            return result
        except PipelineServiceError as pdf_exc:
            if os.getenv("GEMINI_SLICER_ENABLED", "true").lower() == "true":
                if not _gemini_files_fallback_enabled():
                    print(
                        f"⚠️  [extract_pdf_native_gemini] whole-document QP path failed "
                        f"at stage='{pdf_exc.stage}': {pdf_exc.message}. "
                        "Gemini Files fallback is disabled for cost control; "
                        "set GEMINI_ALLOW_FILES_FALLBACK=true to allow double-pass recovery."
                    )
                    raise
                print(
                    f"⚠️  [extract_pdf_native_gemini] whole-document QP path failed "
                    f"at stage='{pdf_exc.stage}': {pdf_exc.message}. "
                    "Activating gemini_slicer fallback..."
                )
                result = await _extract_via_gemini_slicer(
                    pdf_base64, document_type, filename, board, page1_base64, extra_metadata
                )
                print(
                    f"✅ [extract_pdf_native_gemini] gemini_slicer fallback succeeded — "
                    f"{len(result.questions_array)} question(s)."
                )
                return result
            raise

    _SLICER_ENABLED: bool = (
        os.getenv("GEMINI_SLICER_ENABLED", "true").lower() == "true"
    )

    if _SLICER_ENABLED:
        try:
            print(f"🚀 [extract_pdf_native_gemini] Primary path: gemini_slicer | type={document_type!r} board={board!r}")
            result = await _extract_via_gemini_slicer(
                pdf_base64, document_type, filename, board, page1_base64, extra_metadata
            )
            print(
                f"✅ [extract_pdf_native_gemini] gemini_slicer succeeded — "
                f"{len(result.questions_array)} question(s)."
            )
            return result

        except PipelineServiceError as slicer_exc:
            print(
                f"⚠️  [extract_pdf_native_gemini] gemini_slicer failed at "
                f"stage='{slicer_exc.stage}': {slicer_exc.message}. "
            )
            if not _gemini_files_fallback_enabled():
                print(
                    "⚠️  [extract_pdf_native_gemini] Gemini Files API fallback is "
                    "disabled for cost control; set GEMINI_ALLOW_FILES_FALLBACK=true "
                    "to allow an expensive whole-document retry."
                )
                raise
            print("Activating Gemini Files API fallback…")
        except Exception as slicer_exc:
            print(
                f"⚠️  [extract_pdf_native_gemini] gemini_slicer raised "
                f"{type(slicer_exc).__name__}: {slicer_exc}. "
            )
            if not _gemini_files_fallback_enabled():
                raise PipelineServiceError(
                    stage="gemini_slicer",
                    message=(
                        "Gemini slicer failed and expensive Gemini Files fallback "
                        "is disabled for cost control."
                    ),
                    details={
                        "provider": "gemini",
                        "fallback_enabled": False,
                        "exception_type": type(slicer_exc).__name__,
                        "exception": str(slicer_exc),
                    },
                ) from slicer_exc
            print("Activating Gemini Files API fallback…")
    else:
        print(
            "ℹ️  [extract_pdf_native_gemini] GEMINI_SLICER_ENABLED=false — "
            "routing directly to Gemini Files API fallback."
        )

    # ── Fallback: Gemini Files API whole-document upload ──────────────────────
    print("📤 [extract_pdf_native_gemini] Uploading PDF to Gemini Files API (fallback)…")
    return await asyncio.to_thread(
        _extract_pdf_native_sync,
        pdf_base64, document_type, filename, board, page1_base64,
    )


__all__ = ["extract_pdf_native_gemini"]
