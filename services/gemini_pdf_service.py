import asyncio
import base64
import json
import os
import re
import tempfile
import time

import fitz
from google import genai

from schemas.ingestion_schema import ExtractedQuestion, ExtractedPaperMetadata, SlicedQuestionsResponse, QuestionNumberMetadata, ValidationReport
from services.pipeline_errors import PipelineServiceError
from extractors.ref_code_extractor import regex_extract_ref_code
from builders.key_builder import build_paper_reference_key
from utils.question_normalizer import QuestionNumberNormalizer


# ---------------------------------------------------------------------------
# paper_reference_key generator - IGCSE
# ---------------------------------------------------------------------------

def _generate_igcse_paper_reference_key(filename: str) -> str:
    """
    Derive a canonical slug from the IGCSE filename.
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
    uploaded_file = None
    temp_file_path = None
    client = None
    
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
        response = client.models.generate_content(
            model=model,
            contents=[system_prompt, uploaded_file],
            config={"response_mime_type": "application/json"},
        )
        raw_text = getattr(response, "text", "") or ""
        parsed = _parse_json_payload(raw_text)
        return parsed

    except Exception as e:
        print(f"❌ [IB Metadata Extraction Error] {type(e).__name__}: {e!r}")
        return None
        
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


# ---------------------------------------------------------------------------
# Prompt builders (UPDATED WITH ANTI-CROP OVERRIDE)
# ---------------------------------------------------------------------------

def _build_pdf_system_prompt(document_type: str, paper_reference_key: str = "", board: str = "IGCSE") -> str:
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

    prk_instruction = (
        f'\n- "paper_reference_key": set to "{paper_reference_key}" in BOTH metadata and every question object.'
        if paper_reference_key else '\n- "paper_reference_key": set to "" if you cannot determine it.'
    )

    # ── Curriculum-aware difficulty rules ────────────────────────────────────
    board_upper = board.upper()

    if board_upper == "IB":
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND (IB — AO-BASED):
Command term is PRIMARY. Mark count is FALLBACK only when no command term is visible.

  LOW  (AO1 – Recall):
       State, Write down, List, Label, Draw, Plot, Define, Identify, Name.
       Mark fallback: 1–2 marks.

  MEDIUM (AO2 – Application):
       Find, Calculate, Show, "Show that", Determine, Solve, Construct,
       Sketch, Verify, Justify, Apply, Complete, Describe, "Write an expression for".
       Mark fallback: 3–5 marks.

  HIGH (AO3/AO4 – Analysis & Evaluation):
       Derive, Prove, Explain, Analyse, Interpret, Comment, Discuss, Evaluate,
       Suggest, Deduce, "Hence", "Hence or otherwise", "Find the exact value of"
       (multi-step), "To what extent".
       Mark fallback: 6+ marks.

Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()

    elif board_upper in ("IGCSE", "CAMBRIDGE"):
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND (IGCSE UNIVERSAL — OFFICIAL COMMAND WORDS):
Mark count is PRIMARY. Command word is secondary confirmation.
Use the OFFICIAL Cambridge command word definitions and math patterns below.

  LOW  (1 mark  OR  these command words regardless of marks):
       State      — express in clear terms
       Write down — answer without significant working
       Give       — answer from recall or given source
       Write      — answer in a specific form
       Plot       — mark points on a graph
       Name, Identify, List, Label, Recall

  MEDIUM (2 marks  OR  these command words):
       Work out   — calculate with or without a calculator
       Calculate  — work out from given facts or information
       Describe   — state characteristics and main features
       Sketch     — freehand drawing showing key features
       Determine  — establish with certainty (single-step)
       Construct, Complete, Measure, Outline, Suggest
       CORE MATH  — Solve, Expand, Factorise, Simplify

  HIGH (3+ marks  OR  any of these command patterns regardless of marks):
       Show (that) — structured evidence leading to a given result
       Explain     — set out reasons / say why or how
       Comment     — give an informed opinion
       Compare     — identify similarities and/or differences
       Revise      — change to reflect further information
       "Make [variable] the subject of"
       "Find [expression] in terms of [variable]"
       "Draw a [histogram / graph / cumulative frequency curve]"
       "Find the average [speed / rate / density]"
       "Hence show", "Hence or otherwise"
       Derive, Justify, Prove, Analyse

Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()

    elif board_upper in ("A-LEVEL", "ALEVEL"):
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND (A-LEVEL — MARK-DRIVEN):
  LOW    = 1–2 marks.
  MEDIUM = 3–5 marks.
  HIGH   = 6+ marks OR: Prove, Derive, "Show that", Evaluate, Discuss.
Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()

    else:
        print(f"⚠️  [Difficulty] Unknown board '{board}', using IGCSE rules as fallback.")
        difficulty_rule = """
DIFFICULTY & COGNITIVE DEMAND:
  LOW = 1 mark. MEDIUM = 2–3 marks. HIGH = 4+ marks.
Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.
""".strip()

    # ── MS prompt ─────────────────────────────────────────────────────────────
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
3. STRICT DIAGRAM DETECTION ("diagram_urls"): ONLY output ["[NEEDS_CROP]"] if you physically see an ACTUAL visual element in the marking scheme.
   ✅ ALLOWED: Official MS graphs, geometry diagrams, number lines, coordinate axes, drawn figures.
   ❌ FORBIDDEN: NEVER trigger for blank answer lines, worked-text-only answers, or just because a QP had a diagram.
4. "diagram_y_range": If a diagram is detected, the bounding box MUST strictly wrap it. EXPAND by adding 0.05 whitespace ABOVE and BELOW the extreme pixels.
5. {difficulty_rule}
{prk_instruction}
{LATEX_RULES}
""".strip()

    # ── QP prompt ─────────────────────────────────────────────────────────────
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
1. HIERARCHICAL NUMBERING (MANDATORY): Prepend the parent integer to EVERY sub-question. Example: If you see "(a)", you MUST output "4(a)".
2. INFERRED NUMBERING & SEQUENCE (CRITICAL): If a question does not have a visible number, you MUST infer its identity from the sequence.
3. STRICT DIAGRAM DETECTION ("diagram_urls"): ONLY output ["[NEEDS_CROP]"] if you physically see an ACTUAL visual element. 
   ✅ ALLOWED: Geometry figures, trigonometry triangles, graphs, coordinate grids, statistical charts, data tables, 2D/3D shapes.
   ❌ FORBIDDEN: NEVER trigger for blank spaces, ruled lines, empty working areas, or just because the text says "Draw a histogram/graph".
4. DIAGRAM OWNERSHIP: If a diagram appears before sub-parts (a), (b), attach it ONLY to the first sub-part (e.g., "4(a)").
5. "diagram_y_range": The bounding box MUST strictly wrap the visual element. DO NOT include massive empty spaces. Intentionally EXPAND the crop box by adding 0.05 extra whitespace ABOVE and BELOW the extreme pixels.
6. Duplicate ALL metadata fields inside EVERY question object.
7. {difficulty_rule}
{prk_instruction}
{LATEX_RULES}
""".strip()


def _sanitize_answer_blanks(text: str) -> str:
    """
    Remove answer-blank artifacts that Gemini sometimes outputs despite prompt instructions.
    Covers: repeated \\textunderscore, \\ldots chains, \\underline{\\hspace{...}}, dotfill, etc.
    """
    if not text:
        return text
    # Remove runs of \textunderscore (2+ consecutive)
    text = re.sub(r'(\\textunderscore){2,}', '', text)
    # Remove \underline{\hspace{...}} — blank line placeholders
    text = re.sub(r'\\underline\{\\hspace\{[^}]*\}\}', '', text)
    # Remove \dotfill
    text = re.sub(r'\\dotfill', '', text)
    # Remove mark-bracket suffixes like [2] or [1] at end of lines
    text = re.sub(r'\s*\[\d+\]\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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
    "paperNumber": 0, "session": None, "year": 0, 
    "paper_reference_key": "", "unified_paper_key": "", "canonical_question_id": "", "parent_canonical_id": "",
    "question_number_metadata": QuestionNumberMetadata().model_dump(),
    "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
    "isTemplatizable": False, "variables": [], "question_latex": "", "question_id": "", "final_answer": "",
    "total_marks": 0, "method_steps": [], "official_marking_scheme_latex": None, "diagram_urls": [], "needs_review": False,
    "cognitive_demand": "MEDIUM", "difficulty_override": None,
}

_METADATA_DEFAULTS: dict = {
    "curriculum": "", "program": None, "subjectCode": "", "tier": None, "paperNumber": 0, "session": None,
    "year": 0, "paper_reference_key": "", "unified_paper_key": "",
    "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
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


def _normalize_question(raw: dict, fallback_metadata: dict, document_type: str, question_normalizer: QuestionNumberNormalizer) -> dict:
    if not isinstance(raw, dict): return dict(_QUESTION_DEFAULTS)

    raw = _remap_keys(raw, _QUESTION_FIELD_ALIASES)
    result = dict(_QUESTION_DEFAULTS)

    for k in result:
        if k in raw: result[k] = raw[k]

    # Apply question number normalization
    question_id_for_normalization = result.get("question_id") or result.get("question_latex") or ""
    if question_id_for_normalization and fallback_metadata.get("paper_reference_key"):
        normalized_data = question_normalizer.normalize(
            raw_question_id=question_id_for_normalization,
            paper_reference_key=fallback_metadata["paper_reference_key"]
        )
        result.update(normalized_data)
        result["question_number_metadata"] = QuestionNumberMetadata(**normalized_data["question_number_metadata"]).model_dump()

    result["document_type"] = document_type
    result["tier"] = _normalize_tier(result.get("tier"))

    for meta_key in (
        "curriculum", "program", "subjectCode", "tier", "paperNumber", "session", "year",
        "paper_reference_key", "unified_paper_key", "validation_status", "validation_warnings",
        "ref_code_base", "ref_code_full"
    ):
        if not result.get(meta_key) and fallback_metadata.get(meta_key):
            result[meta_key] = fallback_metadata[meta_key]

    if fallback_metadata.get("curriculum"):
        result["curriculum"] = fallback_metadata["curriculum"]
    # paper_reference_key is now also populated by the normalizer, but keep fallback for safety
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

    # Coerce cognitive_demand — guard against Gemini returning unexpected values
    _VALID_DEMANDS = {"LOW", "MEDIUM", "HIGH"}
    if str(result.get("cognitive_demand", "")).upper() not in _VALID_DEMANDS:
        result["cognitive_demand"] = "MEDIUM"
    else:
        result["cognitive_demand"] = str(result["cognitive_demand"]).upper()

    # difficulty_override is always null from Gemini; only humans set it via dashboard
    if result.get("difficulty_override") not in {"Easy", "Medium", "Hard", None}:
        result["difficulty_override"] = None

    if not result["diagram_urls"]:
        q_latex = (result.get("question_latex") or "").lower()
        if has_diagram_indicator or "diagram" in q_latex or "graph" in q_latex or "figure" in q_latex:
            result["diagram_urls"] = []

    if not isinstance(result["diagram_urls"], list): result["diagram_urls"] = []
    result["curriculum"] = result["curriculum"] or ""
    result["subjectCode"] = result["subjectCode"] or ""
    result["question_latex"] = _sanitize_answer_blanks(result.get("question_latex") or "")

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
    # Ensure unified_paper_key is propagated to metadata for validation
    # Instantiate normalizer once at the top
    question_normalizer = QuestionNumberNormalizer()

    if not meta_normalized.get("unified_paper_key") and meta_normalized.get("paper_reference_key"):
        meta_normalized["unified_paper_key"] = question_normalizer._generate_unified_paper_key(
            meta_normalized["paper_reference_key"]
        )

    questions_raw = parsed.get("questions_array") or []
    if not isinstance(questions_raw, list): questions_raw = []

    questions: list[ExtractedQuestion] = []
    qp_parent_ids = set()
    ms_parent_ids = set()

    for i, q in enumerate(questions_raw):
        try:
            # Pass the normalizer instance to _normalize_question
            normalized = _normalize_question(q, meta_normalized, document_type, question_normalizer)
            schema_fields = set(ExtractedQuestion.model_fields.keys())
            filtered = {k: v for k, v in normalized.items() if k in schema_fields}
            question_obj = ExtractedQuestion(**filtered)
            questions.append(question_obj)

            # Collect parent IDs for validation
            if question_obj.parent_canonical_id and question_obj.document_type == "Question Paper":
                qp_parent_ids.add(question_obj.parent_canonical_id)
            elif question_obj.parent_canonical_id and question_obj.document_type == "Marking Scheme":
                ms_parent_ids.add(question_obj.parent_canonical_id)

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

    # --- Internal Sequence Gap Check Validation ---
    validation_status = "ok"
    validation_warnings = []
    recommendation = "proceed"

    # Decide which IDs to check based on the current document (Safely handles one-by-one uploads)
    parent_ids_to_check = qp_parent_ids if document_type == "Question Paper" else ms_parent_ids

    if parent_ids_to_check:
        # Convert IDs to integers for sequence checking (ignoring non-numeric parents like 'A1')
        int_parents = [int(pid) for pid in parent_ids_to_check if str(pid).isdigit()]
        
        if int_parents:
            int_parents.sort()
            expected_sequence = set(range(min(int_parents), max(int_parents) + 1))
            actual_sequence = set(int_parents)
            missing_in_sequence = expected_sequence - actual_sequence
            
            if missing_in_sequence:
                validation_status = "warning"
                recommendation = "review"
                validation_warnings.append(
                    f"Sequence gap detected in {document_type}. Missing parent questions: "
                    f"{', '.join(map(str, sorted(list(missing_in_sequence))))}"
                )

    # Update metadata strictly for fallback
    meta_normalized["validation_status"] = validation_status
    meta_normalized["validation_warnings"] = validation_warnings

    # Create the exact Validation Report envelope Node.js expects
    val_report = ValidationReport(
        status=validation_status,
        recommendation=recommendation,
        message=" | ".join(validation_warnings) if validation_warnings else "Sequence is continuous.",
        checks={"sequence_gaps": len(validation_warnings) > 0}
    )

    return SlicedQuestionsResponse(
        metadata=ExtractedPaperMetadata(**meta_normalized),
        questions_array=questions,
        validation_report=val_report
    )


# ---------------------------------------------------------------------------
# JSON Parser (UPDATED WITH ITERATIVE AUTO-HEAL)
# ---------------------------------------------------------------------------

def _parse_json_payload(content: str) -> dict:
    if not content or not content.strip():
        return {"metadata": {}, "questions_array": []}
    
    # 1. Clean markdown formatting
    cleaned_text = content.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    cleaned_text = cleaned_text.strip()

    # 2. SMART REGEX FOR LATEX
    # Matches a backslash NOT preceded by a backslash (ignores \\),
    # and NOT followed by a quote ("), backslash (\\), or newline (n).
    cleaned_text = re.sub(r'(?<!\\)\\(?!["\\n])', r'\\\\', cleaned_text)

    # 3. ITERATIVE AUTO-HEAL LOOP
    for attempt in range(10):
        try:
            parsed_dict = json.loads(cleaned_text)
            return parsed_dict
        except json.JSONDecodeError as e:
            err_msg = str(e)
            if "Invalid \\escape" in err_msg or "Invalid \\u" in err_msg:
                pos = e.pos
                while pos > 0 and cleaned_text[pos] != '\\':
                    pos -= 1
                
                if cleaned_text[pos] == '\\':
                    cleaned_text = cleaned_text[:pos] + '\\\\' + cleaned_text[pos:]
                    continue  # Retry parsing
                else:
                    print(f"CRITICAL PARSE FAIL (Auto-Heal Failed): {err_msg}")
                    break
            else:
                print(f"CRITICAL PARSE FAIL (Structure Error): {err_msg}")
                break
                
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
        time.sleep(1.0)
    raise TimeoutError(f"File not ACTIVE before timeout. Last state: {last_state}")


# ---------------------------------------------------------------------------
# Retry helper for transient 503 / rate-limit errors
# ---------------------------------------------------------------------------

def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: dict,
    retries: int = 3,
    delay: float = 5.0,
):
    """Wraps generate_content with exponential-step retry for transient errors."""
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
            is_transient = any(
                code in err_str
                for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
            )
            if is_transient and attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(
                    f"⚠️  [Gemini] Transient error on attempt {attempt + 1}/{retries}, "
                    f"retrying in {wait:.0f}s… ({e})"
                )
                time.sleep(wait)
                continue
            raise   # non-transient or final attempt — propagate immediately
    raise last_exc


# ---------------------------------------------------------------------------
# Model priority — change this list to update preference order globally
# ---------------------------------------------------------------------------

_MODEL_PRIORITY: list[str] = [
    "gemini-2.5-flash",       # Primary
    "gemini-2.5-flash-lite",  # Lighter, separate quota, less congestion
    "gemini-1.5-flash",       # Old but stable last resort
]


def _pick_available_model(
    client: genai.Client,
    exclude: list[str] | None = None,
) -> str:
    """
    Returns the highest-priority available Gemini model (no 'models/' prefix).

    Fetches the live model list from the API and walks _MODEL_PRIORITY in order,
    returning the first name that is both available and not in `exclude`.
    """
    exclude_set: set[str] = set(exclude or [])

    try:
        available: set[str] = {
            m.name.replace("models/", "")
            for m in client.models.list()
        }
    except Exception as list_exc:
        print(f"⚠️  [_pick_available_model] Could not fetch model list: {list_exc}. "
              "Proceeding with priority defaults.")
        available = set(_MODEL_PRIORITY)

    # First pass: confirmed available AND not excluded
    for model_name in _MODEL_PRIORITY:
        if model_name not in exclude_set and model_name in available:
            return model_name

    # Second pass: not excluded, even if not confirmed in listing
    for model_name in _MODEL_PRIORITY:
        if model_name not in exclude_set:
            return model_name

    return _MODEL_PRIORITY[0]


# ---------------------------------------------------------------------------
# Core extraction (sync, runs in a thread) (UPDATED WITH MODEL PRIORITY LOOP)
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
                        if sess_digits == "25":
                            session = "may"
                        elif sess_digits in ["11", "00"]:
                            session = "november"
                        else:
                            session = "november"
                            print(f"⚠️ [IB Extraction] Unrecognized session digits \'{sess_digits}\'. Defaulting to \'november\'.")
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

        system_prompt = _build_pdf_system_prompt(document_type, paper_reference_key, board)

        # ── Model selection & generation ─────────────────────────────────────
        # Loops through ALL models in priority order before raising.
        response = None
        last_exc = None
        
        for model_name in _MODEL_PRIORITY:
            try:
                print(f"ℹ️  [Gemini Native PDF] Trying model '{model_name}'…")
                response = _generate_with_retry(
                    client,
                    model=model_name,  
                    contents=[system_prompt, uploaded_file],
                    config={
                        "response_mime_type": "application/json",
                        # Disable extended thinking — not needed for structured extraction.
                        # Cuts latency significantly on gemini-2.5-flash.
                        "thinking_config": {"thinking_budget": 0},
                    },
                    retries=3,
                    delay=5.0,
                )
                print(f"✅ [Gemini Native PDF] Model '{model_name}' succeeded.")
                break  # stop as soon as one model works
            except Exception as exc:
                print(
                    f"⚠️  [Gemini Native PDF] Model '{model_name}' failed "
                    f"({exc}). Trying next…"
                )
                last_exc = exc
                continue
                
        if response is None:
            raise PipelineServiceError(
                stage="pdf_native_gemini",
                message="All models failed.",
                details={
                    "provider": "gemini",
                    "reason": str(last_exc),
                    "exception_type": type(last_exc).__name__,
                },
            )

        # ── raw_text assigned HERE — outside every try/except ────────────────
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
                                      # Add 5% safety padding to avoid cutting top/bottom boundaries
                                      PADDING = 0.05
                                      safe_y0 = max(0.0, float(y_range[0]) - PADDING)
                                      safe_y1 = min(1.0, float(y_range[1]) + PADDING)

                                      y0 = safe_y0 * page.rect.height
                                      y1 = safe_y1 * page.rect.height
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