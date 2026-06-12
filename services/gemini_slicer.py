# File: services/gemini_slicer.py
"""
Single-Pass Multimodal Extraction Engine — Gemini 2.5 Flash (STRICT MODE v3)
=============================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH v2 — BUG AUDIT & SURGICAL FIXES (Principal Engineer Review)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIX A — SILENT KILLER: `_is_geometric_content` always returned False.
  Root cause: It read `region.get("visual_description", "")` but "visual_description"
  was not a field in the Gemini response schema — so region_text was always "".
  Every call hit `if not region_text: return False`, silently dropping ALL
  diagram regions from every QP page. Detection worked; validation killed it.
  Fix: (1) Add `visual_description` to the Gemini prompt schema so the model
  describes each region it finds. (2) Change the fallback: if visual_description
  is absent/empty, default to True (high recall — let other filters decide).

FIX B — DEAD CODE: `_validate_diagram_region` was the well-structured canonical
  validator (with correct padding, needs_review logic, and bounds checking) but
  was NEVER CALLED. `_extract_strict_diagrams` used the broken filter pair instead.
  Fix: Make `_validate_diagram_region` the single source of truth. Remove the
  duplicated, broken filter calls from `_extract_strict_diagrams`. The region
  validation and padding logic now lives in exactly ONE place.

FIX C — LOGIC INVERSION in `_validate_diagram_completeness`:
  It returned False (rejected) for diagrams touching the top/bottom 5% of the page.
  But valid IGCSE diagrams regularly start near the top of content pages (the prism
  in Q2, the triangle in Q4). Rejecting them silently was data loss.
  Fix: Remove the function. Fold boundary detection into `_validate_diagram_region`
  as a flag (needs_review=True, clipping_risk=True), never as a hard rejection.

FIX D — needs_review FLAG OVERWRITE RACE:
  `_validate_diagram_region` correctly set `needs_review = True` when a boundary
  clip was detected (line 854), but 10 lines later overwrite it with
  `entry["needs_review"]` (which Gemini reports as False). Result: boundary
  warnings were silently suppressed.
  Fix: Use OR logic — `needs_review = needs_review or entry.get("needs_review", False)`.
  Once True, the flag is never cleared.

FIX 1 (User Bug) — HEADER/FOOTER REGION EXTRACTION:
  Prompt said "DO NOT CAPTURE headers/footers" but gave no coordinate constraints,
  so Gemini still reported coords in the header/footer zone during drift.
  Fix: (1) Prompt now hard-constrains: "y_start_pct MUST be ≥ 5.0 and y_end_pct
  MUST be ≤ 95.0 — regions outside these bounds are DISQUALIFIED." (2) Server-side
  `_validate_diagram_region` clamps and flags any coordinate that violates this,
  never rejecting silently — we clamp to 5%/95% and set clipping_risk=True.

FIX 2 (User Bug) — EXTRA DOT / NOISE CROPS:
  `_MIN_DIAGRAM_HEIGHT_PCT = 2.0` allowed ~17px crops at 150 DPI (stray dots,
  bullet points, single characters). Also the geometric filter was broken (FIX A).
  Fix: Raise `_MIN_DIAGRAM_HEIGHT_PCT` to 5.0 (~42px minimum — still catches small
  labels and boxes, rejects obvious noise). Add explicit noise rejection rules
  to the prompt: "A valid diagram must have height ≥ 5% of page height. Stray marks,
  dots, isolated numbers, and bullet points are NEVER diagrams."

FIX 3 (User Bug) — QP vs MS CONTEXT:
  The MS-side correctly disabled diagram detection (prompt + `is_ms` check).
  But for QP, the geometric filter was broken (see FIX A), and the schema had no
  way for Gemini to describe what it found in each region.
  Fix: Add `visual_description` to the QP schema. For MS, the prompt retains the
  hard "diagram_regions MUST be []" rule. The `is_ms` guard in the extraction loop
  now correctly skips all region processing for MS pages.

PERF — No concurrency changes. asyncio.Semaphore(3) unchanged. All validation
  happens synchronously during region iteration (zero async overhead added).
  The only latency delta is that `visual_description` adds ~5 tokens per region
  to the Gemini response — negligible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH v3 — ARCHITECTURAL BUG FIXES (Principal Engineer Review)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIX v3-1 — 503 RATE-LIMIT & CONCURRENCY BURST (Jitter):
  Even with Semaphore(3), launching 20+ tasks simultaneously causes a massive
  initial burst. All tasks arrive at the semaphore at t=0, and all retries
  sleep the same deterministic backoff duration — re-colliding in lockstep.
  Fix: `_launch_with_stagger` now adds `random.uniform(0.0, _LAUNCH_JITTER_S)`
  on top of the deterministic ramp. Retried tasks re-arrive spread across a
  window instead of re-creating the original burst.

FIX v3-2 — PREAMBLE MERGING (Q8 missing its "(a)" part):
  The AI merged question preambles (e.g. "8 Darpan runs 12km…") with part (a),
  labeling the entire block "8" and skipping "(a)" entirely.
  Fix: Added explicit "PREAMBLE MERGING IS FORBIDDEN" rule in _build_system_prompt
  STEP 3, with detection signal, mandatory ❌/✅ rules, and a Cambridge example.

FIX v3-3 — PAGE NUMBERS AS QUESTION IDs (Pydantic frozen model):
  `_validate_extracted_root` used `root_int <= last_root_int + 1` which was too
  permissive (allowed backward jumps). The `was_corrected` block tried to mutate
  a frozen Pydantic model in-place via `try: model.question_id = …; except: pass`,
  silently dropping the correction every time.
  Fix (a): Tightened continuity test to strict equality/+1 only.
  Fix (b): Replaced silent in-place mutation with dump→patch→re-instantiate pattern.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH v5 — UNIFIED CONTINUITY GUARD & MS METADATA INHERITANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POINT 1 — UNIFIED CONTINUITY GUARD (document-agnostic):

  Root cause: Three separate `if document_type != "marking scheme"` gates
  were suppressing the orphan fix, the poisoning guard, and the cross-page
  parent context injection for ALL Marking Scheme documents.

  The assumption behind those gates — that MS pages do not suffer from
  ghost numbering — is incorrect. Gemini reads the same Cambridge page
  format for both QP and MS. Mark brackets like "[4]" at line-end are
  identical in both. Page numbers printed top-center are identical in both.
  Orphaned sub-parts like "(b)" with no visible root integer appear in both.

  FIX v5-1 — _build_extracted_question:
    Removed `if document_type.strip().lower() != "marking scheme"` gate
    before _validate_extracted_root call. Root validation now runs for
    both document types unconditionally.

  FIX v5-2 — extract_pages_batch Pass 2 loop:
    Removed the outer `if document_type.strip().lower() != "marking scheme"`
    conditional that wrapped the TASK 2 orphan fix block and the
    post-extraction hallucination guard. Both now run for QP and MS.

  FIX v5-3 — extract_page_with_gemini cross-page context injection:
    Removed `if last_parent and not is_ms` gate. The last_known_parent_id
    context block is now appended to the MS system prompt on pages 2+ just
    as it is for QP. The MS prompt already contains the rule "inherit the
    parent from context above" — without injecting a concrete value, that
    rule has nothing to work from.

  The ONLY document-type gates that remain are diagram-related:
    - `if not is_ms` in the diagram_regions processing loop
      (extract_page_with_gemini, line ~1654)
    - The `is_ms` guard in the diagram prompt section
  These are correct and intentional. Diagram extraction is disabled for MS.
  Question ID correction is not diagram extraction — the gates are separate.

POINT 2 — MS METADATA INHERITANCE (see extract_router.py):

  See the separately delivered extract_router.py for the full implementation.
  The slicer's side of this contract:

  FIX v5-4 — _sync_metadata_to_all_pages MS path:
    The existing `if not is_ms` gate that skipped metadata sync for MS
    documents is replaced with a dual-path:
      - QP path: scan page 0 for authoritative metadata (unchanged).
      - MS path: use the injected `fallback_metadata` (pre-populated with
        QP session/year/tier from the PaperRegistry lookup in the router)
        as `auth_meta` directly — no cover page scan, because MS files have
        no cover page carrying session data.
    If the router did not inject fallback_metadata (emergency fallback),
    the MS sync is skipped with a WARNING log — not silently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


FIX v4-1 — SPLIT-BRAIN SESSION BUG (_sync_metadata_to_all_pages):
  Every `try: model.session = auth_session; except: pass` block (and
  the identical pattern for year/tier/subjectCode) silently raised
  ValidationError on the frozen Pydantic model and was swallowed.
  The sync NEVER actually wrote anything — pages 1–N kept their
  hallucinated sessions, causing MongoDB PaperRegistry split-brain.
  _rewrite_reference_keys had the same setattr + swallowed-exception
  pattern and was equally dead.
  Fix: Replaced the entire function body with dump→patch→re-instantiate.
  _rewrite_reference_keys is inlined into the same dict patch so both
  scalars AND reference-key strings are corrected in a single atomic
  re-instantiation. _rewrite_reference_keys is retained as dead code
  for reference but is no longer called.

FIX v4-2 — ORPHAN FIX SILENT FAILURE (_fix_orphan_question_id):
  `try: model.question_id = corrected_id; except: pass` silently failed.
  The function returned True (corrected!) even though the model was
  unchanged. The caller re-read raw_id = getattr(model, "question_id")
  which was still the orphan "(b)", so _validate_extracted_root saw the
  wrong ID and last_known_parent_id was never updated correctly.
  Fix: Return type changed bool → Optional[model]. Returns the new
  re-instantiated model on success, None if no correction needed or if
  re-instantiation fails. Caller updates item["model"] and local `model`.

FIX v4-3 — STALE MODEL AFTER was_corrected RECONSTRUCTION:
  After the `was_corrected` reconstruction block replaced item["model"],
  the local `model` variable still pointed at the old object. The
  last_known_parent_id tracker update at the bottom of the loop read from
  item.get("model") correctly — but the local `model` used for subsequent
  logic within the same iteration was stale.
  Fix: Added `model = item["model"]` immediately after reconstruction.

FIX v4-4 — GHOST "4" LAST_KNOWN_PARENT_ID POISONING:
  After Q12, the LLM reads mark-bracket "[4]" at line-end and emits
  question_id="4". The tracker accepted it (4 < 12 is a backward jump)
  and poisoned last_known_parent_id = "4". Every subsequent orphan "(iii)"
  became "4(iii)" instead of "12(iii)".
  Fix: Added a plausibility guard — only accept a candidate root if
  candidate_int >= last_int (forward or same). Backward jumps are logged
  as POISONING GUARD rejections and the tracker is left unchanged.

FIX v4-5 — SYSTEM PROMPT: MARK BRACKET [4] AS CATEGORY B ARTIFACT:
  The existing BUG 3 FIX block mentioned "[1]", "[2]", "[3]" mark brackets
  but did not give them a dedicated zero-tolerance rule separate from the
  page-number rule. The LLM could still interpret "[4]" as a root question.
  Fix: Added an explicit CATEGORY B rule (distinct from CATEGORY A page
  numbers) with zero-tolerance language, examples, and a 3-question
  self-check decision tree the model must apply before finalizing any root.
"""

from __future__ import annotations

import asyncio
import base64
from difflib import SequenceMatcher
import json
import logging
import os
import random          # FIX v3-1: required for jitter in _launch_with_stagger
import re
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from schemas.ingestion_schema import ExtractedQuestion, QuestionNumberMetadata
from services.extraction_cost import record_gemini_failure, record_gemini_usage
from services.pipeline_errors import PipelineServiceError
from services.gemini_runtime import run_gemini_async
from utils.question_normalizer import QuestionNumberNormalizer

load_dotenv()
logger = logging.getLogger(__name__)
_QUESTION_NUMBER_NORMALIZER = QuestionNumberNormalizer()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Semaphore: max 3 concurrent Gemini calls — keeps us inside burst quota
# without sacrificing parallelism on multi-page documents.
_GEMINI_SEMAPHORE = asyncio.Semaphore(3)

_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
_QP_PRIMARY_MODEL = (
    os.getenv("GEMINI_QP_PRIMARY_MODEL")
    or os.getenv("GEMINI_QP_MODEL")
    or "gemini-2.5-flash-lite"
).strip()
_QP_RESCUE_MODEL = (os.getenv("GEMINI_QP_RESCUE_MODEL") or _MODEL_NAME).strip()
_QP_MS_ANCHOR_MODEL = (
    os.getenv("GEMINI_QP_MS_ANCHOR_MODEL")
    or _QP_RESCUE_MODEL
    or _MODEL_NAME
).strip()
_MS_MODEL = (os.getenv("GEMINI_MS_MODEL") or _MODEL_NAME).strip()
_QP_LITE_FIRST_ENABLED = os.getenv("GEMINI_QP_LITE_FIRST", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_QP_MS_ANCHOR_FLASH_FIRST = os.getenv("GEMINI_QP_MS_ANCHOR_FLASH_FIRST", "false").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_QP_DEEP_ANCHOR_FLASH_FIRST = os.getenv("GEMINI_QP_DEEP_ANCHOR_FLASH_FIRST", "false").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_QP_DEEP_ANCHOR_MIN_DEPTH = max(3, int(os.getenv("GEMINI_QP_DEEP_ANCHOR_MIN_DEPTH", "3")))
_QP_TARGETED_RESCUE_FLASH_FIRST = os.getenv("GEMINI_QP_TARGETED_RESCUE_FLASH_FIRST", "false").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Retry configuration for provider-side failures.
# A single exhausted page currently invalidates the whole QP response, so one
# extra targeted retry is cheaper than asking the user to re-run all 20 pages.
_MAX_RETRIES = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "3")))
_QP_PRIMARY_RETRIES = max(1, int(os.getenv("GEMINI_QP_PRIMARY_RETRIES", "2")))
_QP_RESCUE_RETRIES = max(0, int(os.getenv("GEMINI_QP_RESCUE_RETRIES", "0")))
_RETRY_BASE_DELAY_S = 4.0          # 4 s → 8 s → 16 s
_TRANSIENT_ERROR_CODES = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "Too Many Requests")
_FATAL_ERROR_PHRASES = (
    "monthly spending cap",
    "project has exceeded its monthly spending cap",
)

# Hard limits for bounding-box sanity guards (mirrors gemini_pdf_service.py values)
_MAX_DIAGRAM_HEIGHT_PCT = 55.0
_MAX_BUFFERED_DIAGRAM_HEIGHT_PCT = 70.0

# REGRESSION FIX: Reset from 5.0 → 2.0.
# 5.0% was too aggressive — it rejected real compact diagrams (small graphs,
# geometric figures, labelled boxes). The noise-rejection benefit did not
# outweigh the data loss. 2.0% is the original production value.
_MIN_DIAGRAM_HEIGHT_PCT = 2.0

# FIX 1: Hard exclusion zone for page headers and footers (as % of page height).
# Anything reporting y_start < this value or y_end > (100 - this value)
# is either a page number, a watermark, or a "Turn over" footer — not a diagram.
_HEADER_FOOTER_ZONE_PCT = 5.0

# Thinking budget:
# QP defaults to 0 for cost control. Diagram recovery no longer depends solely on
# Gemini spatial reasoning because PyMuPDF fallback deterministically assigns
# crop regions when Gemini returns no/invalid diagram_regions. Set
# GEMINI_QP_THINKING_BUDGET=1024 only when you deliberately want model-side
# bbox reasoning for a quality audit run.
_QP_THINKING_BUDGET = int(os.getenv("GEMINI_QP_THINKING_BUDGET", "0"))
_MS_THINKING_BUDGET = int(os.getenv("GEMINI_MS_THINKING_BUDGET", "0"))

# STRICT MODE: Page-specific extraction rules
_METADATA_PAGES_ONLY = (0, 1)  # 0-indexed: pages 1-2
_DIAGRAM_PAGES_START = 2       # 0-indexed: page 3 onwards
_DIAGRAM_PADDING_PCT = 10.0    # 10% padding on all sides (applied ONCE in _validate_diagram_region)

# FIX v5: Ghost-page-number bypass guard.
# Matches remainder strings that open with a continuation sub-part — i.e. the
# first token after the root integer is (b), (c), (ii), (iii), etc.
# A *new* legitimate root question ALWAYS opens with (a) or bare preamble text,
# never mid-alphabet. If the root equals the Cambridge page number AND the
# remainder starts with one of these, the root is a page-number hallucination.
# Explicitly excludes (a) so real "Q3(a)" is never misclassified.
_CONTINUATION_SUBPART_RE = re.compile(
    r'^\s*\('
    r'(?:'
    r'[b-z]'           # (b) (c) (d) … (z)  — NOT (a)
    r'|i{2,}'          # (ii) (iii) (iv) …
    r'|iv|vi{0,3}|ix'  # common Roman numerals
    r')',
    re.IGNORECASE,
)

# Defaults that mirror _QUESTION_DEFAULTS in gemini_pdf_service.py
_QUESTION_DEFAULTS: Dict[str, Any] = {
    "document_type": "Question Paper",
    "curriculum": "", "program": None,
    "subjectCode": "", "tier": None,
    "paperNumber": 0, "session": None, "year": 0,
    "paper_reference_key": "", "unified_paper_key": "",
    "canonical_question_id": "", "parent_canonical_id": "",
    "question_number_metadata": {},
    "validation_status": "pending", "validation_warnings": [],
    "ref_code_base": "", "ref_code_full": "",
    "isTemplatizable": False, "variables": [],
    "question_latex": "", "question_id": "",
    "final_answer": "", "total_marks": 0, "method_steps": [],
    "official_marking_scheme_latex": None,
    "diagram_urls": [],
    "needs_review": False,
    "cognitive_demand": "MEDIUM",
    "difficulty_override": None,
    "diagram_page_number": None,
    "diagram_y_range": [],
}


# ===========================================================================
# SECTION 0 — Strict Extraction Validators
# ===========================================================================
# These enforce page-specific rules and are the single source of truth for
# all region validation. All fixes from the v2 audit live in this section.

def _extract_strict_metadata(raw_metadata: Dict[str, Any], page_num: int) -> Dict[str, Any]:
    """
    STRICT METADATA EXTRACTION (Pages 1-2 ONLY).
    Returns "N/A" for any missing value — never hallucinate.
    Pages 3+ return all N/A to prevent metadata bleed from content pages.
    """
    if page_num not in _METADATA_PAGES_ONLY:
        logger.debug(f"[StrictExtraction] Page {page_num}: Skipping metadata (outside pages 1-2)")
        return {
            "subject_code": "N/A",
            "paper_number": "N/A",
            "session": "N/A",
            "year": "N/A",
        }
    return {
        "subject_code": str(raw_metadata.get("subjectCode") or "").strip() or "N/A",
        "paper_number": str(raw_metadata.get("paperNumber") or "").strip() or "N/A",
        "session": str(raw_metadata.get("session") or "").strip() or "N/A",
        "year": str(raw_metadata.get("year") or "").strip() or "N/A",
    }


def _is_geometric_content(region: Dict[str, Any]) -> bool:
    """
    FIX A: Geometric content filter — revised to not silently reject everything.

    Previous version read `region.get("visual_description", "")` which was always
    "" because the field didn't exist in the schema → returned False for every region.

    New logic:
    1. If `visual_description` IS present (after schema fix in prompt), check it.
    2. If `visual_description` is absent/empty, default to True (high recall).
       The height/area filters in `_validate_diagram_region` act as the noise gate.

    Returns True only for regions that are geometric or when classification
    is impossible (benefit of the doubt → high recall).
    """
    region_text = (region.get("visual_description") or "").strip()

    if not region_text:
        # FIX A: Field absent (PyMuPDF region, or model omitted it).
        # Default to True — let the coordinate bounds filters decide.
        return True

    # Text IS present: apply the text-line heuristic.
    # If there are >3 lines of pure prose with no geometric markers, it's
    # likely a text-answer cell, not a diagram.
    lines = region_text.split("\n")
    text_lines = [
        ln.strip() for ln in lines
        if ln.strip() and not any(c in ln for c in "□△◯┌└┐┘─│")
    ]
    if len(text_lines) > 3:
        return False

    # Accept if any geometric keyword appears in the description.
    geometric_markers = [
        "axis", "axes", "curve", "line", "circle", "polygon", "triangle",
        "rectangle", "square", "graph", "plot", "histogram", "venn",
        "grid", "coordinate", "vertex", "vertices", "point", "points",
        "prism", "cylinder", "diagram", "shape", "angle", "arc",
        "box", "plot", "chart", "bar", "sector", "region",
    ]
    region_lower = region_text.lower()
    return any(marker in region_lower for marker in geometric_markers)


def _extract_strict_diagrams(
    raw_regions: List[Dict[str, Any]],
    page_num: int,
) -> List[Dict[str, Any]]:
    """
    FIX B + FIX A: Diagram extraction gate for pages 3+.

    Delegates validation entirely to `_validate_diagram_region` (the canonical
    validator) instead of the broken `_is_geometric_content` + `_validate_diagram_completeness`
    pair that silently dropped everything.

    Rules:
    1. Only process pages 3+ (0-indexed: page_num >= 2).
    2. Apply geometric content filter with the fixed `_is_geometric_content`.
    3. Delegate coordinate validation + padding to `_validate_diagram_region`.

    Returns a list of validated, padded region dicts.
    """
    if page_num < _DIAGRAM_PAGES_START:
        logger.debug(
            f"[StrictExtraction] Page {page_num}: Skipping (pages 1-2 are metadata only)"
        )
        return []

    validated_regions = []
    for region in raw_regions:
        if not isinstance(region, dict):
            logger.warning(
                f"[StrictExtraction] Page {page_num}: Invalid region type {type(region)} — skipped"
            )
            continue

        # FIX A: Use the repaired geometric filter.
        if not _is_geometric_content(region):
            logger.warning(
                f"[StrictExtraction] Page {page_num} | q={region.get('question_number')!r} "
                f"REJECTED: text-only region (visual_description: {region.get('visual_description', '')[:80]!r})"
            )
            continue

        # FIX B: Delegate to the canonical validator (previously dead code).
        # Padding is applied exactly once here.
        validated = _validate_diagram_region(region, page_num)
        if validated is not None:
            validated_regions.append(validated)

    return validated_regions


# ===========================================================================
# SECTION 1 — Gemini client
# ===========================================================================
# BUG 2 FIX: module-level singleton — constructed once, reused for process lifetime.
_CLIENT_SINGLETON: Optional[genai.Client] = None

def _get_client() -> genai.Client:
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise PipelineServiceError(
                stage="gemini_slicer",
                message="GEMINI_API_KEY is not configured.",
                details={"provider": "gemini"},
            )
        _CLIENT_SINGLETON = genai.Client(api_key=api_key)
    return _CLIENT_SINGLETON


def _thinking_budget_for_document(document_type: str) -> int:
    if str(document_type or "").strip().lower() == "marking scheme":
        return max(0, _MS_THINKING_BUDGET)
    return max(0, _QP_THINKING_BUDGET)


def _model_attempt_plan(
    document_type: str,
    *,
    ms_anchor_active: bool = False,
    deep_anchor_active: bool = False,
    targeted_rescue_active: bool = False,
) -> List[str]:
    """Return the ordered Gemini model plan for one page extraction."""
    if str(document_type or "").strip().lower() == "marking scheme":
        return [_MS_MODEL] * _MAX_RETRIES

    if targeted_rescue_active and _QP_TARGETED_RESCUE_FLASH_FIRST:
        anchor_model = _QP_MS_ANCHOR_MODEL or _QP_RESCUE_MODEL or _MODEL_NAME
        return [anchor_model] * _MAX_RETRIES

    if deep_anchor_active and _QP_DEEP_ANCHOR_FLASH_FIRST:
        anchor_model = _QP_MS_ANCHOR_MODEL or _QP_RESCUE_MODEL or _MODEL_NAME
        return [anchor_model] * _MAX_RETRIES

    if ms_anchor_active and _QP_MS_ANCHOR_FLASH_FIRST:
        anchor_model = _QP_MS_ANCHOR_MODEL or _QP_RESCUE_MODEL or _MODEL_NAME
        return [anchor_model] * _MAX_RETRIES

    if not _QP_LITE_FIRST_ENABLED:
        return [_MODEL_NAME] * _MAX_RETRIES

    primary = _QP_PRIMARY_MODEL or _MODEL_NAME
    rescue = _QP_RESCUE_MODEL or _MODEL_NAME
    plan = [primary] * _QP_PRIMARY_RETRIES
    if rescue and rescue != primary and _QP_RESCUE_RETRIES > 0:
        plan.extend([rescue] * _QP_RESCUE_RETRIES)
    return plan[:_MAX_RETRIES] or [_MODEL_NAME]


def _log_gemini_usage(
    response: Any,
    *,
    page_num: int,
    document_type: str,
    attempt: int,
    model_name: str,
) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        logger.info(
            "[GeminiUsage] model=%s page=%s type=%s attempt=%s usage_metadata=missing",
            model_name,
            page_num,
            document_type,
            attempt,
        )
        return

    def get_metric(name: str) -> Any:
        return getattr(usage, name, None)

    logger.info(
        "[GeminiUsage] model=%s page=%s type=%s attempt=%s prompt_tokens=%s "
        "candidates_tokens=%s thoughts_tokens=%s total_tokens=%s",
        model_name,
        page_num,
        document_type,
        attempt,
        get_metric("prompt_token_count"),
        get_metric("candidates_token_count"),
        get_metric("thoughts_token_count"),
        get_metric("total_token_count"),
    )
    record_gemini_usage(
        model=model_name,
        document_type=document_type,
        page_num=page_num,
        attempt=attempt,
        component="gemini_slicer",
        usage=usage,
    )


# ===========================================================================
# SECTION 2 — JSON sanitization (CRITICAL — fixes raw LaTeX backslashes)
# ===========================================================================

def _sanitize_json_string(raw: str) -> str:
    """
    Aggressively fix JSON strings that contain raw LaTeX backslashes so that
    Python's json.loads() can parse them without raising JSONDecodeError.

    Strategy (ordered — do NOT reorder):
    1.  Strip optional markdown code fences (```json … ```).
    2.  Replace every *lone* backslash (not already doubled, not a JSON escape
        character) with a double backslash.
    3.  Run an iterative auto-heal loop that nudges the position of bad
        escapes when json.loads still fails — up to 20 attempts.
    """
    if not raw:
        return raw

    cleaned = raw.strip()

    # 1. Strip markdown fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 2. Fix lone backslashes.
    #    Regex: match \ not preceded by \ (negative lookbehind) and not
    #    followed by a valid JSON escape char (" \ / b f n r t u).
    cleaned = re.sub(r'(?<!\\)\\(?!["\\\/bfnrtu])', r'\\\\', cleaned)

    # 3. Iterative auto-heal for any remaining broken escapes
    for attempt in range(20):
        try:
            json.loads(cleaned)   # dry run
            break
        except json.JSONDecodeError as exc:
            err_msg = str(exc)
            if "Invalid \\escape" in err_msg or "Invalid \\u" in err_msg:
                pos = exc.pos
                while pos > 0 and cleaned[pos] != "\\":
                    pos -= 1
                if cleaned[pos] == "\\":
                    cleaned = cleaned[:pos] + "\\\\" + cleaned[pos:]
                    continue
            break   # Non-escape structural error — cannot auto-heal

    return cleaned


# ===========================================================================
# SECTION 3 — System prompt builder
# ===========================================================================

def _build_system_prompt(document_type: str, board: str, paper_reference_key: str = "") -> str:
    """
    Build a rich, schema-aware system prompt for single-pass extraction.

    FIX 1 (Header/Footer): The QP diagram_regions schema now requires
    y_start_pct ≥ 5.0 and y_end_pct ≤ 95.0. Violating entries are
    "DISQUALIFIED" per the prompt.

    FIX 2 (Noise): Minimum height rule is now explicit: y_end - y_start ≥ 5.0.
    Stray dots, bullet points, and isolated numbers are called out by name.

    FIX A (visual_description): Added to the diagram_regions schema so the
    model describes what each region contains. This feeds the `_is_geometric_content`
    filter server-side with actual signal instead of an always-empty string.

    FIX v3-2 (Preamble Merging): Added explicit "PREAMBLE MERGING IS FORBIDDEN"
    rule in STEP 3 with Cambridge example. Prevents the AI from merging question
    preamble text into a root-level "8" object and silently swallowing "(a)".
    """

    board_upper = board.upper()

    SHARED_LATEX_RULES = """
MANDATORY LATEX ESCAPING — READ CAREFULLY:
- Extract ALL mathematics as LaTeX. Inline: $...$   Block: $$...$$
- NEVER use Unicode math symbols (², ±, α, ∫). Use LaTeX: ^2, \\pm, \\alpha, \\int.
- CRITICAL: Every backslash in the JSON output MUST be double-escaped.
  WRONG → "\\frac{a}{b}"   RIGHT → "\\\\frac{a}{b}"
  WRONG → "\\sin(x)"       RIGHT → "\\\\sin(x)"
  WRONG → "\\sqrt{3}"      RIGHT → "\\\\sqrt{3}"
  If you output a single backslash inside a JSON string, the parser will crash.
- ANSWER SPACES: Ignore blank lines (......), ruled lines (____), empty boxes,
  and mark-bracket annotations like [2]. Do NOT convert them to \\textunderscore,
  \\underline, or \\dotfill. Omit them entirely.

BUG 2 FIX — SPACING COMMANDS ARE FORBIDDEN:
- NEVER output \\quad, \\qquad, \\hspace, \\vspace, \\medspace, \\thinspace,
  \\enspace, \\noindent, \\smallskip, \\medskip, \\bigskip, or ANY other LaTeX
  spacing/layout command.
- Physical gaps on the page (blank answer-writing areas, vertical white space between
  sub-parts, horizontal indentation) are NOT mathematical content. They exist so students
  can write their answers. They must be COMPLETELY IGNORED.
- Use a single plain space character wherever you need to separate words.
- LaTeX is ONLY for mathematical notation inside $...$ or $$...$$. It is NEVER used
  to replicate the visual layout or spacing of the printed page.
  WRONG: "Find x.\\quad\\quad [3]"
  RIGHT: "Find x."
  WRONG: "\\vspace{2cm} Calculate the area."
  RIGHT: "Calculate the area."
""".strip()

    prk_line = (
        f'  "paper_reference_key": "{paper_reference_key}"'
        if paper_reference_key
        else '  "paper_reference_key": ""'
    )

    if board_upper == "IB":
        difficulty_rule = (
            "DIFFICULTY & COGNITIVE DEMAND (IB - AO-BASED):\n"
            "LOW (AO1): State, Write down, List, Label, Draw, Plot, Define, Identify, Name.\n"
            "MEDIUM (AO2): Find, Calculate, Show, Determine, Solve, Construct, Sketch, Verify, Justify.\n"
            "HIGH (AO3/AO4): Derive, Prove, Explain, Analyse, Interpret, Comment, Discuss, Evaluate, "
            "or multi-step reasoning.\n"
            'Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.'
        )
    else:
        difficulty_rule = (
            "DIFFICULTY & COGNITIVE DEMAND (IGCSE UNIVERSAL - OFFICIAL COMMAND WORDS):\n"
            "Mark count is PRIMARY. Command word is secondary confirmation.\n"
            "Use the OFFICIAL Cambridge command word definitions and maths patterns below.\n\n"
            "LOW (1 mark OR these command words regardless of marks):\n"
            "State, Write down, Give, Write, Plot, Name, Identify, List, Label, Recall.\n\n"
            "MEDIUM (2 marks OR these command words):\n"
            "Work out, Calculate, Describe, Sketch, Determine, Construct, Complete, Measure, Outline, Suggest.\n"
            "CORE MATH: Solve, Expand, Factorise, Simplify.\n\n"
            "HIGH (3+ marks OR these command patterns regardless of marks):\n"
            "Show, Explain, Comment, Compare, Revise, Make [variable] the subject of, "
            "Find [expression] in terms of [variable], Draw a histogram/graph/cumulative frequency curve, "
            "Find the average [speed/rate/density], Hence show, Hence or otherwise, Derive, Justify, Prove, Analyse.\n"
            'Return exactly one of: "LOW", "MEDIUM", "HIGH". Always set "difficulty_override" to null.'
        )

    # ======================================================================
    # ── STRICT SESSION & TIER MAPPING (CRITICAL — FIXES DATA INTEGRITY) ──
    # ======================================================================
    session_mapping_rule = """
🔒 STRICT SESSION MAPPING (ZERO TOLERANCE — NO DEFAULTS):
Extract the session from the cover page cover page ONLY if explicitly stated.
NEVER default to "s" (summer). NEVER use abbreviations or informal names.

MAPPING RULES (EXACT):
  • If the page explicitly mentions "February" or "March" → return EXACTLY "m"
    Examples: "February/March 2024", "March 2023", "Feb/Mar", "February"
  • If the page explicitly mentions "May", "June", or "Summer" → return EXACTLY "s"
    Examples: "May/June 2024", "June 2023", "June/July 2024", "July 2024", "Summer 2023"
  • If the page explicitly mentions "October", "November", or "Winter" → return EXACTLY "w"
    Examples: "October/November 2024", "November 2023", "Oct/Nov", "Winter 2023"
  • If NONE of these months/seasons appear → return null (NOT an empty string, NOT "s")

CRITICAL ANTI-HALLUCINATION RULES:
  ❌ DO NOT default to "s" if you cannot find the month/season.
  ❌ DO NOT use the full month name (e.g. "march"). Only "m", "s", or "w".
  ❌ DO NOT infer the session from the year or filename. Use cover page only.
  ❌ DO NOT return "may/june" or "february/march" — only the letter code.
  ✅ ALWAYS return exactly "m", "s", "w", or null — no variations.
"""

    tier_mapping_rule = """
🔒 STRICT TIER MAPPING (ZERO TOLERANCE):
Extract the tier ONLY from explicit cover page text.

MAPPING RULES (EXACT):
  • If the page contains the word "Extended" (case-insensitive) → return EXACTLY "Extended"
  • If the page contains the word "Core" (case-insensitive) → return EXACTLY "Core"
  • If NEITHER word appears → return "N/A" (never null, never empty string)

CRITICAL ANTI-HALLUCINATION RULES:
  ❌ DO NOT assume "Extended" is the default. Extract ONLY what's printed.
  ❌ DO NOT use abbreviations (e.g. "Ext", "C"). Use exact names.
  ✅ ALWAYS return "Extended", "Core", or "N/A" — no other values.
"""

    # ======================================================================
    # ── MARKING SCHEME PROMPT ─────────────────────────────────────────────
    # ======================================================================
    if document_type.strip().lower() == "marking scheme":
        return f"""
You are an {board} mathematics MARKING SCHEME extraction engine performing a SINGLE-PASS ATOMIC analysis.
Your task: read this exam page image and return ONLY the following JSON object — no prose, no fences.

OUTPUT SCHEMA:
{{
  "metadata": {{
    "curriculum": "{board}",
    "program": null,
    "subjectCode": "<string>",
    "tier": "<string or null>",
    "paperNumber": <integer 0 if unknown>,
    "session": "<string or null>",
    "year": <integer 0 if unknown>,
{prk_line}
  }},
  "questions_array": [
    {{
      "document_type": "Marking Scheme",
      "curriculum": "{board}",
      "program": null,
      "subjectCode": "<string>",
      "tier": "<string or null>",
      "paperNumber": <integer>,
      "session": "<string or null>",
      "year": <integer>,
      "paper_reference_key": "<same as metadata>",
      "question_latex": "<question number as a string, e.g. 3(a)(i)>",
      "question_id": "<same as question_latex>",
      "final_answer": "<concise final answer in LaTeX>",
      "total_marks": <integer>,
      "method_steps": [
        {{ "type": "<M1|A1|B1|ft|oe|dep|allow|accept|SC1>", "description": "<what earns this mark>" }}
      ],
      "official_marking_scheme_latex": "<full marking scheme text in LaTeX>",
      "diagram_urls": [],
      "diagram_regions": [],
      "needs_review": false,
      "cognitive_demand": "<LOW|MEDIUM|HIGH>",
      "difficulty_override": null,
      "isTemplatizable": false,
      "variables": []
    }}
  ]
}}

ATOMIC EXTRACTION RULES:
1. ALWAYS extract ALL mark points into method_steps. Never omit a mark point.
2. Question number MUST include the top-level integer (e.g., "3(a)(i)" not "(a)(i)").
   BUG 3 FIX — NUMBERING ANTI-HALLUCINATION (MANDATORY):
   - Extract question numbers EXACTLY as printed. Never output "unknown", "INVALID", or "?".
   - If a sub-question (e.g. "(ii)") lacks its parent on this page, inherit the parent from
     context above (e.g. if the last root question was "3", output "3(a)(ii)").
   - The bare integer printed at the TOP-CENTER of every page is the PAGE NUMBER — never
     a question number. Do NOT use it as a root question ID under any circumstances.
   - The mark brackets "[1]", "[2]", "[3]" at the end of mark entries are mark counts —
     never part of a question ID. Do NOT include them in question_id or question_latex.
   - If you cannot determine the number with certainty, use your best inference and set
     "needs_review": true. A wrong number that can be corrected is better than "unknown".
2b. ⭐ EXHAUSTIVE EXTRACTION MANDATE (ZERO TOLERANCE — MOST CRITICAL RULE):
   EVERY VISIBLE QUESTION LABEL in Column 1 on THIS PAGE is a separate question object.
   The QP and MS share an IDENTICAL numbering hierarchy. Your output MUST mirror
   that hierarchy completely — every visible label at every depth level.

   MANDATORY COMPLETENESS RULES:
   ✅ "3(a)(i)" and "3(a)(ii)" are TWO SEPARATE objects — never merge them into "3(a)".
   ✅ If Column 1 shows "1(a)(i)", "1(a)(ii)", "1(b)" → that is THREE objects.
   ✅ Include every VISIBLE label even if it has total_marks=0 (preamble rows, continuation headers).
   ✅ Continue the numbering sequence from the previous page — do NOT reset.
   ❌ NEVER skip a visible label because it "seems like a sub-step".
   ❌ NEVER collapse depth levels: "3(a)(i)" is not the same object as "3(a)".
   ❌ NEVER invent a new subpart for an unlabeled continuation answer row.
      If Column 1 is blank/row-spanned under the previous visible label, merge that
      answer/mark row into the previous question object.

   SELF-CHECK (MANDATORY before returning JSON):
   Visually scan Column 1 top-to-bottom and count every unique label you saw.
   Your questions_array length MUST equal that count. If it does not, find the
   missing labels and add them before returning.

3. ⛔ DIAGRAM DETECTION IS DISABLED FOR MARKING SCHEMES ⛔
   - "diagram_urls" MUST always be [] (empty array).
   - "diagram_regions" MUST always be [] (empty array).
   - REASON: MS pages contain worked text only. Outputting diagram coordinates for
     text-based answers causes blank crops that corrupt the database. This is a
     HARD rule with ZERO exceptions.
3b. ⭐ CAMBRIDGE 4-COLUMN TABLE FORMAT (MANDATORY — READ CAREFULLY):
   Cambridge mark schemes are printed as a 4-column table:
     Column 1: Question label  (e.g. "1(a)(i)", "4(b)(i)(a)", "9(b)(ii)")
     Column 2: Answer/working  (the correct answer or key working steps)
     Column 3: Mark integer    (total marks for this part, e.g. 1, 2, 3, 4, 6)
     Column 4: Partial marks   (M1/A1/B1/B2/B3/SC1/FT lines with conditions)

   EXTRACTION RULES FOR THIS FORMAT:
   - "question_latex"  = Column 1 label VERBATIM. E.g. "1(a)(i)". Nothing else.
     ❌ NEVER put the answer text into question_latex. It is a label, not a question.
   - "question_id"     = identical to question_latex.
   - "final_answer"    = Column 2 content. May be a number, expression, or short phrase.
     If Column 2 has multiple acceptable answers (e.g. "66.7 or 66.66 to 66.67"), include all.
   - "total_marks"     = Column 3 integer.
   - "method_steps"    = parse Column 4 into individual entries, one per mark-earning line.
     Each M1/A1/B1/B2/B3/SC1/dep/FT/oe line → one entry:
       {{ "type": "M1", "description": "for 18540 ÷ 9 soi" }}
       {{ "type": "B2", "description": "for 207360 oe" }}
       {{ "type": "SC1", "description": "for answer –10.1 and 7.4" }}
     - "dep" means dependent on previous mark — include it in the type: "M2dep"
     - "FT" means follow-through — include it: "FT" as type, condition as description
     - "oe" (or equivalent) and "isw" (ignore subsequent working) belong in description
     - "OR" separating ALTERNATIVE METHODS: add a step with type "OR_ALT" and
       description "Alternative method:" before the alternative method's steps.
       This preserves the structure for human reviewers.
   - "official_marking_scheme_latex" = full verbatim content of Columns 2 + 4 combined,
     rendered as LaTeX. Preserve all fractions, square roots, and expressions exactly.

   ONE OBJECT PER QUESTION LABEL — ZERO EXCEPTIONS:
   ❌ NEVER split a single question label into multiple objects because it has many mark steps.
      "1(c)" has 5 marks with M4/B2/M1/M2/OR structure → still ONE object with question_latex="1(c)".
   ❌ NEVER create a separate object for each M1/A1 line.
   ✅ ALL partial mark steps for one label → inside that label's method_steps array.

4. {difficulty_rule}
5. If this page has no MS entries (e.g. cover page or Generic Marking Principles page),
   return "questions_array": [].
   ⚠️ The "Generic Marking Principles" page and "Abbreviations" page contain NO question
   entries. Return questions_array: [] for these pages — do NOT extract principles as questions.
6. PARENT CONTEXT: Always include the top-level question number. "3(a)" not just "(a)".
7. ⭐ METADATA EXTRACTION (COVER PAGE ONLY — PAGES 1-2):
   {session_mapping_rule}
   {tier_mapping_rule}

🚫🚫🚫 ATOMIC COUNT MATCHING (ZERO TOLERANCE) 🚫🚫🚫
RULE: Do NOT create new root objects for M1/A1 lines or OR alternatives — those go in method_steps.
      Do NOT merge separate Column 1 labels into one object — each label is its own object.
- Every UNIQUE label in Column 1 on THIS PAGE → exactly ONE object in questions_array.
- "3(a)(i)" and "3(a)(ii)" appearing on this page → TWO objects. Not one.
- Ignore all generic header, footer, and marking principle pages (return []).

✅ CORRECT — Column 1 shows: "11(a)(i)", "11(a)(ii)", "11(b)"
   → THREE objects. Each has its own method_steps with ALL the M1/A1/B1 lines for that label.

🔴 WRONG — Merging "11(a)(i)" and "11(a)(ii)" into a single "11(a)" object.
   WRONG — Creating separate objects for each M1, A1, B1 line within one label.

9. {SHARED_LATEX_RULES}
""".strip()


    # ======================================================================
    # ── QUESTION PAPER PROMPT ─────────────────────────────────────────────
    # FIX 1: Hard coordinate constraints added to diagram_regions schema.
    # FIX 2: Minimum height constraint and noise exclusion rules added.
    # FIX A: visual_description field added to diagram_regions schema.
    # FIX v3-2: PREAMBLE MERGING IS FORBIDDEN rule added to STEP 3.
    # ======================================================================
    return f"""
You are a VISUAL-FIRST {board} mathematics question extraction engine.
Your PRIMARY responsibility is TWO-FOLD:
  (A) Extract all numbered math questions as structured JSON.
  (B) Detect ALL visual diagram regions on this page with bounding-box coordinates.
Both are EQUALLY important. Failing (B) is a CRITICAL EXTRACTION FAILURE.
Your task: read this exam page image and return ONLY the following JSON — no prose, no fences.

OUTPUT SCHEMA:
{{
  "metadata": {{
    "curriculum": "{board}",
    "program": null,
    "subjectCode": "<string>",
    "tier": "<string or null>",
    "paperNumber": <integer 0 if unknown>,
    "session": "<string or null>",
    "year": <integer 0 if unknown>,
{prk_line}
  }},
  "questions_array": [
    {{
      "document_type": "Question Paper",
      "curriculum": "{board}",
      "program": null,
      "subjectCode": "<string>",
      "tier": "<string or null>",
      "paperNumber": <integer>,
      "session": "<string or null>",
      "year": <integer>,
      "paper_reference_key": "<same as metadata>",
      "question_latex": "<full question text, starting with the question number>",
      "question_id": "<label only, e.g. 4(a) or 9(d)(ii); never body text>",
      "official_marking_scheme_latex": null,
      "diagram_urls": [],
      "diagram_regions": [
        {{
          "question_number": "<e.g. 4 or 4(a)>",
          "y_start_pct": <float — top edge of diagram as % of total page height>,
          "y_end_pct":   <float — bottom edge of diagram as % of total page height>,
          "visual_description": "<one-sentence description: e.g. 'triangular prism with labeled vertices A-F'>",
          "needs_review": false
        }}
      ],
      "needs_review": false,
      "cognitive_demand": "<LOW|MEDIUM|HIGH>",
      "difficulty_override": null,
      "isTemplatizable": <true|false>,
      "variables": ["x", "n"]
    }}
  ]
}}

⚡ STEP 1 — VISUAL SCAN (DO THIS FIRST, BEFORE WRITING ANY JSON) ⚡
Before processing any text, scan the ENTIRE page image for visual elements.

  ✅ ALWAYS CAPTURE:
     • Geometric figures: triangles, circles, quadrilaterals, polygons, 3D solids
     • Coordinate axes or grids WITH data already plotted (curves, points, shapes drawn on them)
     • Function graphs: curves, straight lines, parabolas, trig curves
     • Statistical charts: bar charts, pie charts, histograms, cumulative frequency diagrams,
       stem-and-leaf plots, box plots
     • Number lines (any line with numeric labels or tick marks)
     • Venn diagrams, tree diagrams, probability diagrams
     • Any other non-text mathematical visual — when in doubt, INCLUDE IT
     • Cambridge "NOT TO SCALE" diagrams: any geometric figure labelled "NOT TO SCALE"
       is a REAL diagram and MUST be captured. The label is an annotation, not a reason to skip.

  ❌ DO NOT CAPTURE:
     • Blank answer-writing spaces or ruled lines for student work
     • Dotted lines (......) that are fill-in-the-blank spaces
     • Pure text tables listing x-y values (no chart or graph present)
     • Page headers, footers, watermarks, or "Turn over" text
     • Page numbers (at top or bottom margins)
     • Stray dots, bullet points, or isolated single characters — these are NEVER diagrams
     • EMPTY student graph grids — Cambridge prints pre-drawn axis grids (with tick marks,
       axis labels, and a numbered scale) for students to plot their answers onto. These grids
       are EMPTY — no curve, no points, no shape has been plotted yet.
       RULE: if a grid/axis box is empty (no data drawn inside it), DO NOT CAPTURE IT.
       RULE: if a grid/axis box already has a curve, points, or shape drawn on it, DO capture it.
       HOW TO TELL: an empty answer grid has a clean interior with only the axis lines and
       tick-mark labels. A diagram grid has curves, scatter points, or geometric shapes inside.

⚡ STEP 2 — DIAGRAM REGION RULES (MANDATORY — STRICT COORDINATE CONSTRAINTS) ⚡

1.  **COORDINATE BOUNDS (HARD RULE — FIX 1)**:
    - y_start_pct MUST be ≥ 5.0 (top 5% is the page header zone — DISQUALIFIED).
    - y_end_pct MUST be ≤ 95.0 (bottom 5% is the page footer zone — DISQUALIFIED).
    - Any region that violates these bounds is a page artifact (number, watermark,
      "Turn over" footer) — it is NOT a diagram. Do NOT report it.

2.  **MINIMUM SIZE (HARD RULE — FIX 2)**:
    - y_end_pct − y_start_pct MUST be ≥ 5.0 percentage points.
    - Anything smaller than 5% of the page height is a stray mark or bullet point —
      NOT a mathematical diagram. Do NOT report it.

3.  **MAXIMUM SIZE**:
    - y_end_pct − y_start_pct MUST be ≤ 65.0 percentage points.
    - If a diagram spans more than 65% of the page, split it into sub-regions.

4.  **MANDATORY VISUAL BUFFER (10% PADDING)**:
    Include 10% of page height as padding on all sides to ensure axis labels,
    vertices, and keys are NEVER clipped.
    Example: If visual spans y=25% to y=50%, report y_start=15%, y_end=60%.
    (Apply padding BEFORE checking bounds — the bounds check applies to the
    padded values, not the raw visual boundary.)

5.  **visual_description (REQUIRED)**:
    For every region, write a one-sentence description of what you see.
    Examples: "triangular prism with vertices labeled A–F and dimensions 10cm, 5.2cm, 18cm"
              "cumulative frequency S-curve graph, y-axis 0–80, x-axis time in seconds 0–100"
              "triangle ABC with angle 79° at A, sides 8m and 13m labeled, NOT TO SCALE"
              "hemisphere with radius 6cm labeled, NOT TO SCALE"
              "triangle DEF with angle 30° at F, sides (x+4)m and (4x-5)m, NOT TO SCALE"
              "circle with center O, points A(0,5) and B(-3,4) on circumference"
    This field is used for server-side validation. An empty string causes
    the region to be treated as unclassified (it will still pass validation,
    but logging will be less informative).

6.  **TEXTUAL REJECTION**:
    If a region contains MORE text lines than geometric content, set
    "needs_review": true. Heuristic: >3 lines of prose = likely not a diagram.

7.  **HIGH RECALL IS REQUIRED**. If unsure whether something is a diagram,
    INCLUDE IT. A false positive crop is recoverable.
    A MISSED DIAGRAM means diagram_urls = [] in the database — UNRECOVERABLE.

8.  **DIAGRAM OWNERSHIP**: If a diagram appears above sub-parts (a), (b)…,
    assign it to the FIRST sub-part (e.g., "4(a)"). Do NOT duplicate.

9.  **CROSS-PAGE DIAGRAM OWNERSHIP (CAMBRIDGE-SPECIFIC)**:
    Cambridge frequently prints a diagram on page N alongside a parent question (e.g. Q4),
    then continues sub-parts on page N+1 with NO repeated diagram.
    On page N+1 you will NOT see the diagram — output diagram_regions: [] for those sub-parts.
    NEVER invent a region on page N+1 if no diagram is visible there.
    The crop injection pipeline handles propagation from page N automatically.

10. **CAMBRIDGE "NOT TO SCALE" CONVENTION**:
    Every Cambridge geometric diagram has "NOT TO SCALE" printed beside it as a small
    annotation. This is NEVER a reason to skip the diagram. Any page containing labeled
    geometric figures (triangles, prisms, spheres, circles with labeled vertices or
    dimensions) MUST have those figures captured as diagram regions.

11. ⚠️ RETURNING diagram_regions: [] FOR A PAGE WITH VISIBLE DIAGRAMS IS A
    CRITICAL EXTRACTION FAILURE.

⚡ STEP 3 — QUESTION EXTRACTION RULES ⚡
1. HIERARCHICAL NUMBERING (MANDATORY):
   Prepend the parent integer to EVERY sub-question.
   If you see "(a)" under question 4, output "4(a)". Never output just "(a)".
   If you see "(ii)" under "5(a)", output "5(a)(ii)". The full hierarchy is ALWAYS required.

   🚫 PREAMBLE MERGING IS FORBIDDEN (ZERO TOLERANCE — FIX v3-2) 🚫
   Cambridge questions frequently use this structure:
     "8  Darpan runs 12 km at an average speed of v km/h.
      (a)  Write down an expression, in terms of v, for the time taken."

   In this structure the introductory sentence ("Darpan runs 12 km…") is the
   QUESTION STEM — it belongs to part (a), NOT to a standalone root object "8".

   MANDATORY RULES:
   ❌ NEVER create a root-level question object with question_id "8" that contains
      only the preamble text and no mark allocation. That object would swallow the
      preamble and cause part (a) to be silently lost from the output.
   ✅ The introductory/preamble text MUST be prepended to the FIRST sub-part.
      The first sub-part's question_id MUST be "8(a)", NEVER bare "8".
   ✅ If sub-parts (a), (b), (c) follow a shared preamble, each carries the preamble
      in its question_latex so the context is self-contained:
        question_id "8(a)": "8(a) [preamble text] (a) [part text]"
        question_id "8(b)": "8(b) [preamble text] (b) [part text]"
   ✅ A root question object with id "8" (no sub-part suffix) is ONLY valid when
      the question has NO sub-parts at all — i.e. it is a single standalone question
      with its own mark allocation printed directly after the question text.

   OVERRIDE FOR SHARED PREAMBLES:
   The first child under a shared stem is the context anchor. Put the shared
   preamble in question_latex for that first child only. Later siblings must keep
   the full hierarchy in question_id but must not repeat the same preamble text.
   Example: "8(a) [preamble] [part a text]", then "8(b) [part b text only]".

   DETECTION SIGNAL: If you see a question number (e.g. "8") followed by a sentence
   and then immediately "(a)", that sentence is the preamble of "(a)". The correct
   output is ONE object with question_id="8(a)" — NOT two objects "8" and "8(a)".

BUG 3 FIX — QUESTION NUMBERING ANTI-HALLUCINATION (ZERO TOLERANCE):

⚠️⚠️⚠️ TWO CATEGORIES OF ARTIFACTS ARE NEVER QUESTION NUMBERS ⚠️⚠️⚠️

CATEGORY A — PAGE NUMBERS (top-center integers):
   Cambridge prints a bare integer CENTERED at the VERY TOP of every page.
   This is the PAGE NUMBER. It sits ALONE on its line with no question text
   beside it or below it on the SAME LINE.
   ❌ NEVER use the top-center integer as a root question ID.
   ❌ EXAMPLE BUG: page 9 shows "9" at top-center, body has "(c)(i)" →
      you output "9(c)(i)". THIS IS WRONG. The question root is inherited
      from the previous page (e.g. "4"), correct output is "4(c)(i)".

CATEGORY B — MARK ALLOCATION BRACKETS (end-of-line integers in [brackets]):
   Cambridge prints "[1]", "[2]", "[3]", "[4]" etc. at the END of every
   sub-question's answer space to show how many marks it is worth.
   EXAMPLE: "Find the value of x.    [4]" — the "[4]" means FOUR MARKS.
   ❌ NEVER interpret a mark bracket as a question number.
   ❌ NEVER output question_id = "4" because you saw "[4]" at line-end.
   ❌ NEVER create a new question object just because you see a bracket.
   ✅ Strip mark brackets entirely. They appear ONLY at the end of a line,
      inside square brackets. Real question numbers are LEFT-ALIGNED at
      the START of a line, followed immediately by text or "(a)".

BEFORE FINALIZING ANY ROOT QUESTION NUMBER — ASK YOURSELF:
   "Is this integer printed top-center with nothing else on its line?" → PAGE NUMBER. Skip it.
   "Is this integer inside [square brackets] at the end of a line?"   → MARK COUNT. Skip it.
   "Does this integer appear LEFT-ALIGNED at the start of a question line,
    followed by text or a sub-part label?"                            → REAL QUESTION. Use it.

   a. Extract question numbers EXACTLY as printed. Do not rephrase or abbreviate.
   b. If a sub-question (e.g. "(ii)") appears without its parent number on THIS
      page, INHERIT the parent from context above on the same page or from the
      last question number you identified.
      EXAMPLE: Last identified question was "5(a)". Next entry is "(ii)".
               Output MUST be "5(a)(ii)" — NEVER just "(ii)".
   c. NEVER output "unknown", "INVALID", "N/A", "?", or any placeholder.
      If you cannot determine the number, use the next logical number in
      sequence and set "needs_review": true.
   d. NEVER invent a question number. If guessing, set "needs_review": true
      but always output a real number.
   e. The question number belongs at the START of question_latex.

2. INFERRED NUMBERING: If a question has no visible number, infer it from sequence context.
3. PARENT CONTEXT (MULTI-PAGE CRITICAL):
   If sub-parts appear without the parent stem on this page, still include the parent
   question number at the start of question_latex.
4. 🚫 CRITICAL: IGNORE PAGE NUMBERS AND MARK BRACKETS — NEITHER ARE QUESTION NUMBERS 🚫
   See the CATEGORY A and CATEGORY B rules above. They apply with ZERO EXCEPTIONS.
   WRONG: Page shows "9" at top-center then "(c)(i)" in body → "9(c)(i)"  ❌
   WRONG: Line ends with "[4]" → you emit question_id="4"                  ❌
   CORRECT: Inherit root from last known question, strip brackets entirely  ✅
5. {difficulty_rule}
6. If this page has no numbered mathematical questions, return "questions_array": [].
7. Duplicate ALL metadata fields inside every question object.
8. ⭐ METADATA EXTRACTION (COVER PAGE ONLY — PAGES 1-2):
   {session_mapping_rule}
   {tier_mapping_rule}
9. {SHARED_LATEX_RULES}
10. VARIABLES — FLAT STRING ARRAY ONLY (CRITICAL):
    "variables" MUST be a flat array of plain variable name strings.
    CORRECT:   "variables": ["x", "n", "r"]
    WRONG:     "variables": [{{"name": "x", "type": "integer", "min": 0}}]
""".strip()


# ===========================================================================
# SECTION 4 — Response parser & validator
# ===========================================================================

def _parse_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse and sanitize Gemini's raw text response into a Python dict.
    Returns {"metadata": {}, "questions_array": []} on any failure.
    """
    if not raw_text or not raw_text.strip():
        return {"metadata": {}, "questions_array": []}

    sanitized = _sanitize_json_string(raw_text)

    try:
        parsed = json.loads(sanitized)
        if isinstance(parsed, dict):
            return parsed
        logger.warning("[GeminiSlicer] Response top-level is not a dict.")
        return {"metadata": {}, "questions_array": []}
    except json.JSONDecodeError as exc:
        logger.error(
            f"[GeminiSlicer] JSON parse failed after sanitization: {exc}. "
            f"First 500 chars: {sanitized[:500]}"
        )
        return {"metadata": {}, "questions_array": []}


def _validate_diagram_region(entry: Any, page_num: int) -> Optional[Dict[str, Any]]:
    """
    CANONICAL VALIDATOR for a single diagram_region dict.

    FIX B: This function is now the single source of truth for region validation.
    Previously it was dead code — it was never called. Now `_extract_strict_diagrams`
    delegates here instead of using the broken parallel filter pair.

    FIX C: Boundary proximity no longer REJECTS regions — it sets needs_review=True
    and clipping_risk=True, then clamps the coordinate to the safe zone. Legitimate
    diagrams at the top of IGCSE content pages are no longer silently dropped.

    FIX D: needs_review uses OR logic. Once set to True (by boundary detection),
    it cannot be overwritten by the model reporting False.

    FIX 1: Hard header/footer clamp applied here as a defense-in-depth layer.
    If the prompt constraints fail (model drift), the server still clamps.

    FIX 2: Minimum height raised to _MIN_DIAGRAM_HEIGHT_PCT (5.0).

    Returns None if the entry is structurally invalid (missing fields, inverted
    coordinates, over-height). Never returns None for boundary proximity alone.
    """
    if not isinstance(entry, dict):
        logger.debug(f"[GeminiSlicer] Diagram region is not a dict: {type(entry)}")
        return None

    q_num = str(entry.get("question_number") or "").strip()
    y0 = entry.get("y_start_pct")
    y1 = entry.get("y_end_pct")

    if not q_num or y0 is None or y1 is None:
        logger.debug(
            f"[GeminiSlicer] Incomplete region data: q_num={q_num!r}, y0={y0}, y1={y1}"
        )
        return None

    try:
        y0_f, y1_f = float(y0), float(y1)
    except (ValueError, TypeError):
        logger.debug(f"[GeminiSlicer] Non-numeric y-coordinates: y0={y0}, y1={y1}")
        return None

    if y0_f >= y1_f or y0_f < 0.0 or y1_f > 100.0:
        logger.debug(
            f"[GeminiSlicer] Invalid coordinate range: y0={y0_f}, y1={y1_f} "
            f"(must be 0 ≤ y0 < y1 ≤ 100)"
        )
        return None

    height_pct = y1_f - y0_f

    # Hard reject: over-size (likely a full-page scan or coordinate hallucination)
    if height_pct > _MAX_DIAGRAM_HEIGHT_PCT:
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | q={q_num!r} REJECTED: "
            f"oversized height={height_pct:.1f}% (max {_MAX_DIAGRAM_HEIGHT_PCT}%)"
        )
        return None

    # FIX 2: Hard reject: under-size (stray dot, bullet, single character).
    # Threshold raised from 2.0% → 5.0%.
    if height_pct < _MIN_DIAGRAM_HEIGHT_PCT:
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | q={q_num!r} REJECTED: "
            f"microscopic height={height_pct:.1f}% — noise artifact "
            f"(min {_MIN_DIAGRAM_HEIGHT_PCT}%)"
        )
        return None

    # ──────────────────────────────────────────────────────────────────────
    # FIX D: needs_review uses OR logic — starts False, can only go to True.
    # ──────────────────────────────────────────────────────────────────────
    # Initialize from the model's report (if it's a valid boolean).
    needs_review = bool(entry.get("needs_review", False))
    clipping_risk = False

    # ──────────────────────────────────────────────────────────────────────
    # FIX 1 + FIX C: Header/footer zone handling.
    # If coordinates fall in the exclusion zone, clamp them and flag —
    # never hard-reject. The content itself might be valid (a diagram at the
    # very top of a content page is legitimate; a "page 5" number at y=1% is not).
    # After clamping, re-check the minimum height.
    # ──────────────────────────────────────────────────────────────────────
    if y0_f < _HEADER_FOOTER_ZONE_PCT:
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | q={q_num!r} y_start={y0_f:.1f}% "
            f"is in header zone (<{_HEADER_FOOTER_ZONE_PCT}%). Clamping to "
            f"{_HEADER_FOOTER_ZONE_PCT}% and flagging for review."
        )
        y0_f = _HEADER_FOOTER_ZONE_PCT
        clipping_risk = True
        needs_review = True   # FIX D: OR — never cleared after this

    if y1_f > (100.0 - _HEADER_FOOTER_ZONE_PCT):
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | q={q_num!r} y_end={y1_f:.1f}% "
            f"is in footer zone (>{100.0 - _HEADER_FOOTER_ZONE_PCT}%). Clamping to "
            f"{100.0 - _HEADER_FOOTER_ZONE_PCT}% and flagging for review."
        )
        y1_f = 100.0 - _HEADER_FOOTER_ZONE_PCT
        clipping_risk = True
        needs_review = True   # FIX D: OR — never cleared after this

    # After clamping, re-validate the height
    height_pct_clamped = y1_f - y0_f
    if height_pct_clamped < _MIN_DIAGRAM_HEIGHT_PCT:
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | q={q_num!r} REJECTED after "
            f"header/footer clamping: height={height_pct_clamped:.1f}% < "
            f"{_MIN_DIAGRAM_HEIGHT_PCT}% — was entirely in exclusion zone"
        )
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Apply 10% padding (ONCE — this is the single padding application point)
    # ──────────────────────────────────────────────────────────────────────
    y0_buffered = max(_HEADER_FOOTER_ZONE_PCT, y0_f - _DIAGRAM_PADDING_PCT)
    y1_buffered = min(100.0 - _HEADER_FOOTER_ZONE_PCT, y1_f + _DIAGRAM_PADDING_PCT)
    buffered_height_pct = y1_buffered - y0_buffered

    if buffered_height_pct > _MAX_BUFFERED_DIAGRAM_HEIGHT_PCT:
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | q={q_num!r} REJECTED: "
            f"buffered crop height={buffered_height_pct:.1f}% "
            f"(max {_MAX_BUFFERED_DIAGRAM_HEIGHT_PCT}%) — refusing likely full-page crop"
        )
        return None

    # If padding would push past the header/footer zone, that's also a clipping risk
    if y0_f - _DIAGRAM_PADDING_PCT < _HEADER_FOOTER_ZONE_PCT:
        clipping_risk = True
        needs_review = True   # FIX D: OR

    if y1_f + _DIAGRAM_PADDING_PCT > (100.0 - _HEADER_FOOTER_ZONE_PCT):
        clipping_risk = True
        needs_review = True   # FIX D: OR

    visual_desc = str(entry.get("visual_description") or "").strip()

    logger.debug(
        f"[GeminiSlicer] ✅ Valid region: page={page_num}, q={q_num!r}, "
        f"y={y0_f:.1f}→{y0_buffered:.1f}%–{y1_f:.1f}→{y1_buffered:.1f}%, "
        f"height={height_pct_clamped:.1f}%, needs_review={needs_review}, "
        f"desc={visual_desc[:60]!r}"
    )

    return {
        "question_number":      q_num,
        "y_start_pct":          round(y0_buffered, 2),
        "y_end_pct":            round(y1_buffered, 2),
        "y_start_pct_original": round(y0_f, 2),
        "y_end_pct_original":   round(y1_f, 2),
        "page_num":             page_num,
        "needs_review":         needs_review,
        "clipping_risk":        clipping_risk,
        "visual_description":   visual_desc,
    }


# ===========================================================================
# SECTION 5 — Pydantic normalization helpers
# ===========================================================================
# BUG 6 FIX: computed once at import time, not per question.
_SCHEMA_FIELDS: frozenset = frozenset(ExtractedQuestion.model_fields.keys())

def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return bool(value)
    return default


def _normalize_cognitive_demand(raw: Any) -> str:
    val = str(raw or "").upper().strip()
    return val if val in {"LOW", "MEDIUM", "HIGH"} else "MEDIUM"


def _normalize_tier(tier: Any) -> str:
    if not tier or not isinstance(tier, str):
        return "N/A"
    t = tier.lower().strip()
    if "higher" in t or t == "hl":
        return "HL"
    if "standard" in t or t == "sl":
        return "SL"
    if "core" in t:
        return "Core"
    if "extended" in t:
        return "Extended"
    return "N/A"


def _strip_mark_brackets(raw_id: str) -> str:
    """
    BUG 2 FIX — Strip Cambridge mark-allocation brackets from a question ID string.

    Cambridge prints "[1]", "[2]", "[3]" etc. at the end of every answer space.
    Gemini sometimes reads these into the question_id/question_latex field, which
    then gets normalised into a spurious sub-part suffix (e.g. "4.b.ii.4").

    This strips any trailing [ integer ] sequences from the candidate ID string.
    It is applied to the raw question_id BEFORE canonical ID normalisation.
    (Full question_latex cleanup is handled separately by _sanitize_answer_blanks.)
    """
    if not raw_id:
        return raw_id
    return re.sub(r'\s*\[\d+\]\s*$', '', raw_id.strip())


def _extract_safe_question_label(*candidates: Any) -> str:
    """
    Extract only the leading visual question label from candidate strings.

    This protects payload text such as "f(x) = 2x - 3" from being absorbed into
    question_id when Gemini omits a dedicated QP question_id field.
    """
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        if re.match(r"^(?:unknown|unk|invalid|n/?a|none|null|\?)(?:\s*[\(\.\-].*)?$", text, re.IGNORECASE):
            continue
        label = _QUESTION_NUMBER_NORMALIZER.extract_leading_label(
            _strip_mark_brackets(text)
        )
        if label:
            return label
    return ""


def _question_parts_from_model(model: Any) -> List[str]:
    raw_id = str(getattr(model, "question_id", "") or "").strip()
    raw_text = str(getattr(model, "question_latex", "") or "").strip()
    return _QUESTION_NUMBER_NORMALIZER.extract_parts(raw_id or raw_text)


def _canonical_from_parts(parts: List[str]) -> str:
    return _QUESTION_NUMBER_NORMALIZER.canonical_from_parts(parts) if parts else ""


def _model_text_for_overlap(model: Any) -> str:
    q_text = str(getattr(model, "question_latex", "") or "")
    ms_text = str(getattr(model, "official_marking_scheme_latex", "") or "")
    return q_text or ms_text


def _normalize_overlap_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _text_similarity(a: str, b: str) -> float:
    a_norm = _normalize_overlap_text(a)
    b_norm = _normalize_overlap_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _common_prefix_len_at_word_boundary(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    while idx > 0 and not a[idx - 1].isspace():
        idx -= 1
    return idx


def _replace_leading_question_label(text: str, new_label: str) -> str:
    """
    Replace only the parsed leading label. Never regex-slice arbitrary payload
    text, which is how inline math can disappear.
    """
    if not text:
        return new_label
    _old_label, remainder = _QUESTION_NUMBER_NORMALIZER.split_label_and_remainder(text)
    if not _old_label:
        return text
    return f"{new_label} {remainder}".strip() if remainder else new_label


def _dedupe_shared_context_text(
    previous_text: str,
    current_text: str,
    corrected_parts: List[str],
) -> str:
    """
    Trim duplicated shared stem text only when a corrected sibling clearly embeds
    the next sub-indicator inside a repeated block.

    This runs after a structural duplicate has been repaired, so the new label is
    already known. If the marker cannot be found safely, the function leaves the
    text alone rather than risking content loss.
    """
    corrected_label = _QUESTION_NUMBER_NORMALIZER.format_parts(corrected_parts)
    if not previous_text or not current_text or not corrected_label:
        return _replace_leading_question_label(current_text, corrected_label)

    _prev_label, prev_remainder = _QUESTION_NUMBER_NORMALIZER.split_label_and_remainder(previous_text)
    _cur_label, cur_remainder = _QUESTION_NUMBER_NORMALIZER.split_label_and_remainder(current_text)
    if not prev_remainder or not cur_remainder:
        return _replace_leading_question_label(current_text, corrected_label)

    similarity = _text_similarity(prev_remainder, cur_remainder)
    if similarity < 0.72:
        return _replace_leading_question_label(current_text, corrected_label)

    terminal = corrected_parts[-1].lower() if corrected_parts else ""
    marker_patterns = [
        rf"(?<![A-Za-z0-9])\(\s*{re.escape(terminal)}\s*\)",
        rf"(?<![A-Za-z0-9]){re.escape(terminal)}\s*[\).]",
    ]

    marker_match = None
    for pattern in marker_patterns:
        marker_match = re.search(pattern, cur_remainder, flags=re.IGNORECASE)
        if marker_match:
            break

    if marker_match and marker_match.start() > 0:
        suffix = cur_remainder[marker_match.end():].lstrip()
        if suffix:
            return f"{corrected_label} {suffix}".strip()
        return corrected_label

    prefix_len = _common_prefix_len_at_word_boundary(prev_remainder, cur_remainder)
    if prefix_len >= 40:
        suffix = cur_remainder[prefix_len:].lstrip()
        if len(suffix) >= 12:
            return f"{corrected_label} {suffix}".strip()

    return _replace_leading_question_label(current_text, corrected_label)


def _rebuild_model_with_structural_id(
    model: Any,
    corrected_parts: List[str],
    page_num: int,
    previous_text: str = "",
) -> Optional[Any]:
    corrected_label = _QUESTION_NUMBER_NORMALIZER.format_parts(corrected_parts)
    corrected_canonical = _canonical_from_parts(corrected_parts)
    if not corrected_label or not corrected_canonical:
        return None

    try:
        model_dict = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    except Exception as dump_exc:
        logger.error(
            f"[GeminiSlicer][SequenceGuard] Page {page_num}: model dump failed "
            f"for structural repair to {corrected_label!r}: {dump_exc!r}"
        )
        return None

    current_text = str(model_dict.get("question_latex") or "")
    repaired_text = _dedupe_shared_context_text(
        previous_text=previous_text,
        current_text=current_text,
        corrected_parts=corrected_parts,
    )

    model_dict["question_id"] = corrected_label
    model_dict["canonical_question_id"] = corrected_canonical
    model_dict["parent_canonical_id"] = _QUESTION_NUMBER_NORMALIZER.parent_from_parts(corrected_parts)
    model_dict["question_number_metadata"] = QuestionNumberMetadata(
        **_QUESTION_NUMBER_NORMALIZER.build_question_metadata(corrected_parts)
    )
    model_dict["question_latex"] = repaired_text
    model_dict["needs_review"] = True

    warnings = model_dict.get("validation_warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(
        "Structural sequence guard corrected a duplicated Gemini question label."
    )
    model_dict["validation_warnings"] = warnings

    try:
        return type(model)(**model_dict)
    except Exception as recon_exc:
        logger.error(
            f"[GeminiSlicer][SequenceGuard] Page {page_num}: failed to rebuild "
            f"model for {corrected_label!r}: {recon_exc!r}"
        )
        return None


def _rebuild_model_with_anchor_id(
    model: Any,
    corrected_parts: List[str],
    warning: str,
) -> Optional[Any]:
    """
    Rebuild a question model using an externally trusted anchor ID.

    This is intentionally separate from the duplicate structural repair helper:
    MS-anchor repairs are not "next sibling" guesses. They use the saved Marking
    Scheme's ordered canonical IDs to undo page-number hallucinations from QP
    extraction, especially with cheaper Lite models.
    """
    corrected_label = _QUESTION_NUMBER_NORMALIZER.format_parts(corrected_parts)
    corrected_canonical = _canonical_from_parts(corrected_parts)
    if not corrected_label or not corrected_canonical:
        return None

    try:
        model_dict = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    except Exception:
        return None

    model_dict["question_id"] = corrected_label
    model_dict["canonical_question_id"] = corrected_canonical
    model_dict["parent_canonical_id"] = _QUESTION_NUMBER_NORMALIZER.parent_from_parts(corrected_parts)
    model_dict["question_number_metadata"] = QuestionNumberMetadata(
        **_QUESTION_NUMBER_NORMALIZER.build_question_metadata(corrected_parts)
    )
    model_dict["question_latex"] = _replace_leading_question_label(
        str(model_dict.get("question_latex") or ""),
        corrected_label,
    )
    model_dict["needs_review"] = True

    warnings = model_dict.get("validation_warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(warning)
    model_dict["validation_warnings"] = warnings

    try:
        return type(model)(**model_dict)
    except Exception:
        return None


def _expected_anchor_parts(fallback_metadata: Optional[Dict[str, Any]]) -> List[List[str]]:
    expected = (fallback_metadata or {}).get("expected_canonical_ids") or []
    if not isinstance(expected, list):
        return []

    out: List[List[str]] = []
    seen: set[str] = set()
    for value in expected:
        parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(str(value or ""))
        canonical = _canonical_from_parts(parts)
        if parts and canonical and canonical not in seen:
            out.append(parts)
            seen.add(canonical)
    return out


def _has_deep_anchor_ids(expected_ids: List[str]) -> bool:
    for value in expected_ids or []:
        parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(str(value or ""))
        if len(parts) >= _QP_DEEP_ANCHOR_MIN_DEPTH:
            return True
    return False


def _next_expected_anchor_for_suffix(
    expected_parts: List[List[str]],
    used_canonical_ids: set[str],
    raw_suffix_parts: List[str],
    min_root: Optional[int] = None,
) -> Optional[List[str]]:
    """
    Pick the next unused saved-MS ID whose suffix matches the hallucinated QP row.

    Examples:
      raw 8(a)(i) on printed page 8 -> suffix a.i -> next expected 6.a.i
      raw 9(ii)   on printed page 9 -> suffix ii  -> next expected 6.a.ii
      raw 13(b)   on printed page 13 -> suffix b  -> next expected 8.b
    """
    suffix = [str(part).lower() for part in raw_suffix_parts if str(part).strip()]
    if not expected_parts or not suffix:
        return None

    for parts in expected_parts:
        canonical = _canonical_from_parts(parts)
        if not canonical or canonical in used_canonical_ids:
            continue
        if min_root is not None:
            try:
                if int(parts[0]) < min_root:
                    continue
            except (ValueError, TypeError, IndexError):
                continue
        parts_lower = [str(part).lower() for part in parts]
        if len(parts_lower) >= len(suffix) and parts_lower[-len(suffix):] == suffix:
            return parts
    return None


def _next_expected_anchor_for_page(
    expected_parts: List[List[str]],
    used_canonical_ids: set[str],
    page_expected_ids: List[str],
) -> Optional[List[str]]:
    """
    Pick the next unused saved-MS ID that the local QP skeleton assigned to
    this rendered page. This is a fallback for printed-page hallucinations
    where even the suffix is wrong, e.g. raw 12(c)(iii) on a page whose local
    skeleton expects 12.b.iii.
    """
    if not expected_parts or not page_expected_ids:
        return None

    page_expected = {
        str(value).strip().lower()
        for value in page_expected_ids
        if str(value).strip()
    }
    if not page_expected:
        return None

    for parts in expected_parts:
        canonical = _canonical_from_parts(parts)
        if not canonical or canonical in used_canonical_ids:
            continue
        if canonical in page_expected:
            return parts
    return None


def _next_unused_parts(parts: List[str], used_canonical_ids: set[str]) -> Optional[List[str]]:
    candidate = [str(p).lower() for p in parts]
    for _ in range(20):
        candidate = _QUESTION_NUMBER_NORMALIZER.increment_terminal_part(candidate) or []
        if not candidate:
            return None
        if _canonical_from_parts(candidate) not in used_canonical_ids:
            return candidate
    return None


def _apply_sequential_duplicate_guard(
    item: Dict[str, Any],
    previous_model: Optional[Any],
    used_canonical_ids: set[str],
    page_num: int,
    is_ms: bool = False,
) -> None:
    """
    Stateful array-level guard for duplicate Gemini emissions.

    If Gemini emits two distinct consecutive objects with the same structural
    label, the second object is reassigned to the next sibling (5(c)(i) ->
    5(c)(ii), 5(c) -> 5(d), etc.) instead of leaving a duplicate key for Mongo.
    """
    model = item.get("model")
    if model is None:
        return

    parts = _question_parts_from_model(model)
    canonical = _canonical_from_parts(parts)
    if is_ms:
        if canonical:
            used_canonical_ids.add(canonical)
        return

    if not canonical or len(parts) < 2:
        if canonical:
            used_canonical_ids.add(canonical)
        return

    current_text = _model_text_for_overlap(model)
    previous_text = _model_text_for_overlap(previous_model) if previous_model is not None else ""
    duplicate_seen = canonical in used_canonical_ids
    consecutive_duplicate = False

    if previous_model is not None:
        prev_parts = _question_parts_from_model(previous_model)
        consecutive_duplicate = canonical == _canonical_from_parts(prev_parts)

    if not duplicate_seen and not consecutive_duplicate:
        used_canonical_ids.add(canonical)
        return

    distinct_text = _normalize_overlap_text(current_text) != _normalize_overlap_text(previous_text)
    high_overlap = _text_similarity(current_text, previous_text) >= 0.72
    if previous_model is not None and not distinct_text and not high_overlap:
        used_canonical_ids.add(canonical)
        return

    corrected_parts = _next_unused_parts(parts, used_canonical_ids)
    if not corrected_parts:
        used_canonical_ids.add(canonical)
        return

    repaired = _rebuild_model_with_structural_id(
        model=model,
        corrected_parts=corrected_parts,
        page_num=page_num,
        previous_text=previous_text,
    )
    if repaired is None:
        used_canonical_ids.add(canonical)
        return

    logger.warning(
        f"[GeminiSlicer][SequenceGuard] Page {page_num}: duplicate structural "
        f"ID {canonical!r} repaired to {_canonical_from_parts(corrected_parts)!r}."
    )
    item["model"] = repaired
    used_canonical_ids.add(_canonical_from_parts(corrected_parts))


def _align_qp_model_to_visible_label(item: Dict[str, Any], used_canonical_ids: set[str]) -> None:
    """
    Prefer the visible label at the start of QP question_latex over a stale
    question_id/canonical pair.

    Example seen in review:
      canonical_question_id = 6.b.i
      question_latex        = "6(c) Complete the table."

    The visible PDF label is the stronger signal. Align internal identity before
    duplicate/sequence repair runs so the guard does not push the row into an
    invented child such as 6.f.i.
    """
    model = item.get("model")
    if model is None:
        return

    question_text = str(getattr(model, "question_latex", "") or "").strip()
    visible_label = _extract_safe_question_label("", question_text)
    if not visible_label:
        return

    visible_parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(visible_label)
    visible_canonical = _canonical_from_parts(visible_parts)
    current_canonical = _canonical_from_parts(_question_parts_from_model(model))
    if not visible_canonical or visible_canonical == current_canonical:
        return

    if visible_canonical in used_canonical_ids:
        return

    try:
        model_dict = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    except Exception:
        return

    model_dict["question_id"] = _QUESTION_NUMBER_NORMALIZER.format_parts(visible_parts)
    model_dict["canonical_question_id"] = visible_canonical
    model_dict["parent_canonical_id"] = _QUESTION_NUMBER_NORMALIZER.parent_from_parts(visible_parts)
    model_dict["question_number_metadata"] = QuestionNumberMetadata(
        **_QUESTION_NUMBER_NORMALIZER.build_question_metadata(visible_parts)
    )

    try:
        item["model"] = type(model)(**model_dict)
    except Exception:
        return


def _apply_backward_root_guard(
    item: Dict[str, Any],
    previous_model: Optional[Any],
    used_canonical_ids: set[str],
    last_known_parent_id: str,
    page_num: int,
) -> None:
    """
    Deprecated safety valve.

    The earlier implementation rewrote any backward root to the last tracker
    root. In real QPs, a stale/page-number-poisoned tracker can be larger than
    the actual next question root, which turns valid 8/9/10/11/12 questions into
    nonsense like 13.c, 13.d, etc. Keep the function as a no-op so old call sites
    remain harmless while duplicate repair stays handled by the sequence guard.
    """
    return


_ROMAN_SEQUENCE = [
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
]


def _is_missing_or_unknown_ms_id(model: Any) -> bool:
    raw_id = str(getattr(model, "question_id", "") or "").strip().lower()
    canonical = _canonical_from_parts(_question_parts_from_model(model))
    return (
        not raw_id
        or raw_id in {"unknown", "n/a", "none", "null"}
        or not canonical
        or canonical in {"unknown", "n.a"}
    )


def _next_roman_parts(parts: List[str]) -> Optional[List[str]]:
    if not parts:
        return None
    candidate = [str(p).lower() for p in parts]
    terminal = candidate[-1]
    if terminal not in _ROMAN_SEQUENCE:
        return None
    idx = _ROMAN_SEQUENCE.index(terminal)
    if idx + 1 >= len(_ROMAN_SEQUENCE):
        return None
    candidate[-1] = _ROMAN_SEQUENCE[idx + 1]
    return candidate


def _rebuild_ms_model_with_parts(
    model: Any,
    parts: List[str],
    warning: str,
) -> Optional[Any]:
    if not parts:
        return None
    label = _QUESTION_NUMBER_NORMALIZER.format_parts(parts)
    canonical = _canonical_from_parts(parts)
    if not label or not canonical:
        return None
    try:
        model_dict = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    except Exception:
        return None

    model_dict["question_id"] = label
    model_dict["question_latex"] = label
    model_dict["canonical_question_id"] = canonical
    model_dict["parent_canonical_id"] = _QUESTION_NUMBER_NORMALIZER.parent_from_parts(parts)
    model_dict["question_number_metadata"] = QuestionNumberMetadata(
        **_QUESTION_NUMBER_NORMALIZER.build_question_metadata(parts)
    )
    warnings = model_dict.get("validation_warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(warning)
    model_dict["validation_warnings"] = warnings
    model_dict["needs_review"] = True

    try:
        return type(model)(**model_dict)
    except Exception:
        return None


def _merge_text_lines(*values: Any) -> str:
    merged: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in merged:
            merged.append(text)
    return "\n".join(merged)


def _merge_ms_models(primary: Any, continuation: Any) -> Any:
    try:
        primary_dict = primary.model_dump() if hasattr(primary, "model_dump") else primary.dict()
        cont_dict = continuation.model_dump() if hasattr(continuation, "model_dump") else continuation.dict()
    except Exception:
        return primary

    primary_dict["final_answer"] = _merge_text_lines(
        primary_dict.get("final_answer"),
        cont_dict.get("final_answer"),
    )
    primary_dict["official_marking_scheme_latex"] = _merge_text_lines(
        primary_dict.get("official_marking_scheme_latex"),
        cont_dict.get("official_marking_scheme_latex"),
    )

    try:
        primary_marks = int(primary_dict.get("total_marks") or 0)
        continuation_marks = int(cont_dict.get("total_marks") or 0)
        primary_dict["total_marks"] = primary_marks + continuation_marks
    except Exception:
        primary_dict["total_marks"] = primary_dict.get("total_marks") or cont_dict.get("total_marks") or 0

    primary_steps = primary_dict.get("method_steps") if isinstance(primary_dict.get("method_steps"), list) else []
    cont_steps = cont_dict.get("method_steps") if isinstance(cont_dict.get("method_steps"), list) else []
    primary_dict["method_steps"] = [*primary_steps, *cont_steps]

    warnings = []
    for source in (primary_dict.get("validation_warnings"), cont_dict.get("validation_warnings")):
        if isinstance(source, list):
            warnings.extend(str(w) for w in source if str(w).strip())
    warnings.append("MS row-span continuation merged into the visible question label.")
    primary_dict["validation_warnings"] = list(dict.fromkeys(warnings))
    primary_dict["needs_review"] = bool(primary_dict.get("needs_review") or cont_dict.get("needs_review"))

    try:
        return type(primary)(**primary_dict)
    except Exception:
        return primary


def _roman_index(part: Any) -> Optional[int]:
    token = str(part or "").strip().lower()
    if token not in _ROMAN_SEQUENCE:
        return None
    return _ROMAN_SEQUENCE.index(token)


def _immediate_parent_parts(parts: List[str]) -> List[str]:
    if len(parts) <= 1:
        return []
    return [str(p).lower() for p in parts[:-1]]


def _ms_step_types(model: Any) -> List[str]:
    steps = getattr(model, "method_steps", None) or []
    types: List[str] = []
    for step in steps:
        raw_type = ""
        if isinstance(step, dict):
            raw_type = step.get("type") or ""
        else:
            raw_type = getattr(step, "type", "") or ""
        clean = re.sub(r"\s+", "", str(raw_type).strip().lower())
        if clean:
            types.append(clean)
    return types


def _is_likely_ms_answer_continuation(model: Any) -> bool:
    """
    Detect Cambridge MS row-span continuation rows.

    Example:
      8(a)(i)  <working formula>      M1
                41.74 to 41.75       A1
      8(a)(ii) 5.9[0] ...            4

    Gemini sometimes invents 8(a)(ii) for the unlabeled A1 row, shifting the real
    8(a)(ii) to 8(a)(iii).  This predicate is intentionally narrow: it only
    accepts short answer/accuracy rows with at most one mark and no rich method
    structure.
    """
    try:
        marks = int(getattr(model, "total_marks", 0) or 0)
    except Exception:
        marks = 0
    if marks > 1:
        return False

    final_answer = str(getattr(model, "final_answer", "") or "").strip()
    official = str(getattr(model, "official_marking_scheme_latex", "") or "").strip()
    combined = " ".join(part for part in (final_answer, official) if part).strip()
    if not combined or len(combined) > 220:
        return False

    step_types = _ms_step_types(model)
    if not step_types:
        return marks <= 1 and len(combined) <= 120

    allowed_accuracy_types = {
        "a1", "b1", "ft", "oe", "cao", "nfww", "mark", "answer", "final",
    }
    if any(step_type.startswith(("m2", "m3", "m4", "b2", "b3", "sc")) for step_type in step_types):
        return False
    return all(
        step_type in allowed_accuracy_types
        or step_type.startswith(("a1", "b1", "ft"))
        for step_type in step_types
    )


def _previous_model_has_ms_method_mark(model: Any) -> bool:
    step_types = _ms_step_types(model)
    return any(step_type.startswith(("m1", "m2", "m3", "m4")) for step_type in step_types)


def _maybe_merge_shifted_ms_roman_rowspans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Repair a common shifted-roman MS pattern:

      previous: 8.a.i   (method row, e.g. M1)
      current:  8.a.ii  (unlabeled answer continuation, e.g. A1)
      next:     8.a.iii (real visible 8(a)(ii), shifted by Gemini)

    The current row is merged into previous, then later roman siblings under the
    same parent are shifted down by one. This is deliberately conservative so a
    genuine one-mark subpart is not casually swallowed.
    """
    output: List[Dict[str, Any]] = []
    shift_rules: Dict[str, int] = {}
    i = 0

    while i < len(items):
        item = items[i]
        model = item.get("model")
        parts = _question_parts_from_model(model) if model is not None else []
        parent_parts = _immediate_parent_parts(parts)
        parent_key = _canonical_from_parts(parent_parts)
        terminal_idx = _roman_index(parts[-1]) if parts else None

        if parent_key and terminal_idx is not None and parent_key in shift_rules:
            start_idx = shift_rules[parent_key]
            if terminal_idx >= start_idx and terminal_idx > 0:
                shifted_parts = [str(p).lower() for p in parts]
                shifted_parts[-1] = _ROMAN_SEQUENCE[terminal_idx - 1]
                rebuilt = _rebuild_ms_model_with_parts(
                    model,
                    shifted_parts,
                    "MS row-span repair shifted roman label after merged continuation.",
                )
                if rebuilt is not None:
                    item = {**item, "model": rebuilt}
                    model = rebuilt
                    parts = shifted_parts
                    terminal_idx = terminal_idx - 1

        prev_item = output[-1] if output else None
        next_item = items[i + 1] if i + 1 < len(items) else None
        prev_model = prev_item.get("model") if prev_item else None
        next_model = next_item.get("model") if next_item else None
        prev_parts = _question_parts_from_model(prev_model) if prev_model is not None else []
        next_parts = _question_parts_from_model(next_model) if next_model is not None else []
        prev_parent = _immediate_parent_parts(prev_parts)
        next_parent = _immediate_parent_parts(next_parts)
        prev_idx = _roman_index(prev_parts[-1]) if prev_parts else None
        next_idx = _roman_index(next_parts[-1]) if next_parts else None

        should_merge_current = (
            model is not None
            and prev_model is not None
            and next_model is not None
            and parent_parts
            and parent_parts == prev_parent == next_parent
            and prev_idx is not None
            and terminal_idx is not None
            and next_idx is not None
            and terminal_idx == prev_idx + 1
            and next_idx == terminal_idx + 1
            and _previous_model_has_ms_method_mark(prev_model)
            and _is_likely_ms_answer_continuation(model)
        )

        if should_merge_current and prev_item is not None:
            prev_item["model"] = _merge_ms_models(prev_item["model"], model)
            if parent_key:
                shift_rules[parent_key] = terminal_idx
            logger.info(
                "[GeminiSlicer][MSRowSpanRepair] Merged likely unlabeled answer "
                "continuation %s into %s; subsequent roman siblings under %s shift down.",
                _canonical_from_parts(parts),
                _canonical_from_parts(prev_parts),
                parent_key,
            )
            i += 1
            continue

        output.append(item)
        i += 1

    return output


def _repair_and_merge_ms_rowspan_continuations(flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cambridge MS tables often use a row-spanned label: one visible "6(a)"
    covers several answer rows. Those continuation rows must be merged under
    the same MS label, not promoted to 6(b), 6(c), etc. Roman continuations are
    the exception: after 3(d)(ii), an unlabeled next row is usually 3(d)(iii).
    """
    repaired: List[Dict[str, Any]] = []
    previous_model: Optional[Any] = None

    for item in flat:
        model = item.get("model")
        if model is None:
            repaired.append(item)
            continue

        if _is_missing_or_unknown_ms_id(model) and previous_model is not None:
            previous_parts = _question_parts_from_model(previous_model)
            next_roman = _next_roman_parts(previous_parts)
            if next_roman:
                rebuilt = _rebuild_ms_model_with_parts(
                    model,
                    next_roman,
                    "MS continuation row inferred as the next roman subpart.",
                )
            else:
                rebuilt = _rebuild_ms_model_with_parts(
                    model,
                    previous_parts,
                    "MS row-span continuation assigned to previous visible label.",
                )
            if rebuilt is not None:
                item = {**item, "model": rebuilt}
                model = rebuilt

        repaired.append(item)
        previous_model = item.get("model") or previous_model

    merged: List[Dict[str, Any]] = []
    by_canonical: Dict[str, int] = {}
    for item in repaired:
        model = item.get("model")
        canonical = _canonical_from_parts(_question_parts_from_model(model)) if model is not None else ""
        if canonical and canonical in by_canonical:
            existing_idx = by_canonical[canonical]
            existing_item = merged[existing_idx]
            existing_item["model"] = _merge_ms_models(existing_item["model"], model)
            continue
        if canonical:
            by_canonical[canonical] = len(merged)
        merged.append(item)

    # Keep this repair disabled by default. It was designed for rare Cambridge
    # row-spanned MS tables where Gemini invents a roman label for an unlabeled
    # continuation answer row. In normal MS tables, however, genuine visible
    # labels such as 1(a)(ii) can also be short one-mark answer rows, so applying
    # the heuristic globally silently shifts all later numbering and merges
    # answers/marks into the wrong canonical ID.
    if str(os.getenv("GEMINI_MS_ROWSPAN_SHIFT_REPAIR", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        merged = _maybe_merge_shifted_ms_roman_rowspans(merged)

    return merged


def _validate_extracted_root(
    raw_id: str,
    last_known_parent_id: str,
    page_num: int,
) -> "tuple[str, bool]":
    """
    FIX v5 — POST-EXTRACTION GUARD: Detect and correct page-number-as-root-ID
    hallucinations.  Supersedes the v3-3a guard.

    Cambridge pages print a bare integer at the top-center of every page (the
    page number). On continuation pages Gemini sometimes reads this number as the
    root question ID instead of inheriting the correct parent.

    THE SPECIFIC BUG THIS VERSION CLOSES (v3-3a missed it):
        last_known_parent_id = "2"
        page_num = 2  →  cambridge_page_number = 3
        Gemini emits "3(b)"

        v3-3a guard:
          root_int(3) == cambridge_page_number(3)  ← fires
          root_int(3) == last_root_int(2) + 1      ← True → passes! BUG.

        v5 guard adds SIGNAL 2:
          remainder "(b)" starts with a continuation sub-part
          → definitively a hallucination regardless of continuity check.

    SIGNALS (any one is sufficient to flag hallucination):

    SIGNAL 1 — Strict jump implausibility:
        root_int is not the same question (== last_root_int) and not
        the immediately next question (== last_root_int + 1).

    SIGNAL 2 — Continuation sub-part on a page-number root  ← NEW in v5:
        root_int == cambridge_page_number AND remainder starts with
        (b), (c), (d), (ii), (iii) … (anything except (a)).
        A real new root question ALWAYS opens with (a) or preamble text —
        never mid-alphabet. Finding Q3(b) where page==3 is always wrong.

    Signal 2 intentionally overrides Signal 1's "plausible continuation"
    exemption. "3 == 2+1" is NOT enough to save "3(b)" when page==3.

    Returns:
        (corrected_id: str, was_corrected: bool)

    The caller sets needs_review=True when was_corrected is True.
    """
    if not raw_id:
        return raw_id, False

    root_match = re.match(r'^(\d+)(.*)', raw_id.strip())
    if not root_match:
        return raw_id, False

    root_str  = root_match.group(1)
    remainder = root_match.group(2)  # e.g. "(b)(ii) Calculate …"

    try:
        root_int = int(root_str)
    except ValueError:
        return raw_id, False

    # 1-based Cambridge page number for this 0-indexed page
    cambridge_page_number = page_num + 1

    # Guard only fires when root matches the printed page number AND we have context.
    if root_int != cambridge_page_number or not last_known_parent_id:
        return raw_id, False

    try:
        last_root_int = int(last_known_parent_id)
    except ValueError:
        # Non-numeric last parent — correct unconditionally; continuity is unknowable.
        corrected_id = last_known_parent_id + remainder
        logger.warning(
            f"[GeminiSlicer] Page {page_num} | HALLUCINATION (non-numeric parent): "
            f"root={root_str!r} == page {cambridge_page_number}. "
            f"Correcting '{raw_id}' → '{corrected_id}'. needs_review=True."
        )
        return corrected_id, True

    # ── SIGNAL 2 (v5 — NEW): continuation sub-part on a page-number root ─────
    # Detected via _CONTINUATION_SUBPART_RE (compiled at module level).
    # Overrides the continuity exemption below intentionally.
    is_continuation_subpart = bool(_CONTINUATION_SUBPART_RE.match(remainder))

    # ── SIGNAL 1: strict continuity — same question or exactly next ───────────
    is_plausible_continuation = (
        root_int == last_root_int           # same question still running
        or root_int == last_root_int + 1    # immediately next question
    )

    # Hallucination if Signal 2 fires OR Signal 1 fails.
    is_hallucination = is_continuation_subpart or (not is_plausible_continuation)

    if not is_hallucination:
        return raw_id, False

    corrected_id = last_known_parent_id + remainder
    if corrected_id == raw_id:
        return raw_id, False

    if is_continuation_subpart and is_plausible_continuation:
        reason = (
            f"remainder '{remainder[:20]}' is a continuation sub-part "
            f"(Signal 2) — root {root_str} == page {cambridge_page_number} "
            f"even though {root_str} == {last_root_int}+1 (Signal 1 would pass)"
        )
    else:
        reason = (
            f"implausible jump from last_parent={last_known_parent_id!r} "
            f"(expected {last_root_int} or {last_root_int + 1}, got {root_int})"
        )

    logger.warning(
        f"[GeminiSlicer] Page {page_num} | PAGE-NUMBER HALLUCINATION DETECTED: "
        f"root={root_str!r} matches Cambridge page {cambridge_page_number} — {reason}. "
        f"Correcting '{raw_id}' → '{corrected_id}'. needs_review=True."
    )
    return corrected_id, True


def _sanitize_answer_blanks(text: str) -> str:
    """Strip LaTeX answer-blank artefacts that Gemini sometimes emits."""
    if not text:
        return text
    text = re.sub(r'(\\textunderscore){2,}', '', text)
    text = re.sub(r'\\underline\{\\hspace\{[^}]*\}\}', '', text)
    text = re.sub(r'\\dotfill', '', text)
    text = re.sub(r'\s*\[\d+\]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _build_extracted_question(
    raw_q: Dict[str, Any],
    document_type: str,
    fallback_metadata: Dict[str, Any],
    page_num: int = 0,
    last_known_parent_id: str = "",
) -> Optional[ExtractedQuestion]:
    """
    Merge raw question dict from Gemini with fallback metadata and build an
    ExtractedQuestion Pydantic model.  Returns None on unrecoverable error.
    """
    if not isinstance(raw_q, dict):
        return None

    merged = dict(_QUESTION_DEFAULTS)

    # Apply fallback metadata first (lowest priority)
    for k, v in fallback_metadata.items():
        if k in merged:
            merged[k] = v

    # Apply question-level fields (higher priority)
    for k, v in raw_q.items():
        if k in merged:
            merged[k] = v

    # Override document_type authoritatively
    merged["document_type"] = document_type

    # ── NUCLEAR FIX: FORCE KEYS & SESSION FROM NODE.JS FALLBACK ──
    # Prevents AI hallucinated 's' from overwriting the authoritative 'm' from Node.js
    if fallback_metadata.get("paper_reference_key"):
        merged["paper_reference_key"] = fallback_metadata["paper_reference_key"]
    if fallback_metadata.get("unified_paper_key"):
        merged["unified_paper_key"] = fallback_metadata["unified_paper_key"]
    if fallback_metadata.get("session"):
        merged["session"] = fallback_metadata["session"]

    # ── BUG 2 FIX: Strip mark-allocation brackets from question_id ───────────
    # Cambridge mark brackets like "[3]" at the end of answer spaces must NOT
    # be treated as part of the question ID. Strip them before any normalisation.
    explicit_q_id = str(merged.get("question_id") or "").strip()
    q_latex_for_id = str(merged.get("question_latex") or "").strip()
    raw_q_id = _extract_safe_question_label(explicit_q_id, q_latex_for_id)
    if not raw_q_id:
        raw_q_id = _strip_mark_brackets(explicit_q_id or q_latex_for_id)

    # ── BUG 1/3 FIX: Detect and correct page-number-as-root hallucinations ───
    # On continuation pages the model may read the top-center page number (e.g. "9")
    # as the root question number, or read a mark bracket "[4]" as a root ID.
    # _validate_extracted_root detects this and replaces the hallucinated root
    # with the last known correct parent ID.
    #
    # POINT 1 — UNIFIED CONTINUITY GUARD:
    # The old gate `if document_type != "marking scheme"` has been removed.
    # Both QP and MS pages can hallucinate page numbers as root IDs and can
    # read mark brackets like "[4]" as question IDs. The guard is structural
    # and document-agnostic; the document_type distinction belongs only to
    # diagram extraction, not to question ID correction.
    if raw_q_id and last_known_parent_id:
        corrected_id, was_corrected = _validate_extracted_root(
            raw_q_id, last_known_parent_id, page_num
        )
        if was_corrected:
            merged["question_id"] = corrected_id
            # Also fix the leading question number in question_latex.
            old_latex = str(merged.get("question_latex") or "")
            old_root_m = re.match(r'^(\d+)', raw_q_id)
            new_root_m = re.match(r'^(\d+)', corrected_id)
            if old_root_m and new_root_m and old_root_m.group(1) != new_root_m.group(1):
                old_root_digits = old_root_m.group(1)
                new_root_digits = new_root_m.group(1)
                # Anchor at start, require word-boundary after the digits so "9(" and
                # "9 " both match but "90" or "9.5" do not.
                merged["question_latex"] = re.sub(
                    r'^' + re.escape(old_root_digits) + r'\b',
                    new_root_digits,
                    old_latex,
                )
            merged["needs_review"] = True  # Flag for human review
        elif raw_q_id != str(merged.get("question_id") or ""):
            # At minimum write back the safe label-only version.
            merged["question_id"] = raw_q_id
    elif raw_q_id != str(merged.get("question_id") or ""):
        merged["question_id"] = raw_q_id
    # ── END BUG FIXES ─────────────────────────────────────────────────────────

    safe_parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(
        merged.get("question_id") or ""
    )
    if safe_parts:
        merged["canonical_question_id"] = _QUESTION_NUMBER_NORMALIZER.canonical_from_parts(safe_parts)
        merged["parent_canonical_id"] = _QUESTION_NUMBER_NORMALIZER.parent_from_parts(safe_parts)
        merged["question_number_metadata"] = _QUESTION_NUMBER_NORMALIZER.build_question_metadata(safe_parts)

    merged["paperNumber"]    = _coerce_int(merged.get("paperNumber"), 0)
    merged["year"]           = _coerce_int(merged.get("year"), 0)
    merged["total_marks"]    = _coerce_int(merged.get("total_marks"), 0)
    merged["isTemplatizable"] = _coerce_bool(merged.get("isTemplatizable"), False)
    merged["needs_review"]   = _coerce_bool(merged.get("needs_review"), False)
    merged["tier"]           = _normalize_tier(merged.get("tier"))
    merged["cognitive_demand"] = _normalize_cognitive_demand(merged.get("cognitive_demand"))

    if merged.get("difficulty_override") not in {"Easy", "Medium", "Hard", None}:
        merged["difficulty_override"] = None

    merged["question_latex"] = _sanitize_answer_blanks(str(merged.get("question_latex") or ""))

    raw_urls = merged.get("diagram_urls")
    if not isinstance(raw_urls, list):
        merged["diagram_urls"] = []
    else:
        merged["diagram_urls"] = [
            u for u in raw_urls
            if isinstance(u, str) and (
                u.startswith("http") or
                u.startswith("data:image") or
                u.startswith("//") or
                u == "[NEEDS_CROP]" or
                (u.startswith("[") and u.endswith("]"))
            )
        ]

    if document_type.strip().lower() == "marking scheme":
        raw_steps = merged.get("method_steps") or []
        if isinstance(raw_steps, list):
            merged["method_steps"] = [
                {
                    "type": str(s.get("type", "")).strip(),
                    "description": str(s.get("description", "")).strip(),
                }
                for s in raw_steps if isinstance(s, dict)
            ]
        else:
            merged["method_steps"] = []
        if not merged.get("question_id"):
            merged["question_id"] = merged.get("question_latex", "")
    else:
        merged["method_steps"] = []

    raw_vars = merged.get("variables")
    if not isinstance(raw_vars, list):
        merged["variables"] = []
    else:
        coerced: List[str] = []
        for v in raw_vars:
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip()
                if s:
                    coerced.append(s)
            elif isinstance(v, dict):
                name = (v.get("name") or v.get("variable") or v.get("label") or v.get("symbol"))
                if name and isinstance(name, str) and name.strip():
                    coerced.append(name.strip())
                elif isinstance(v.get("options"), list):
                    for opt in v["options"]:
                        if isinstance(opt, str) and opt.strip():
                            coerced.append(opt.strip())
            else:
                coerced.append(str(v))
        merged["variables"] = coerced

    filtered = {k: v for k, v in merged.items() if k in _SCHEMA_FIELDS}

    try:
        return ExtractedQuestion(**filtered)
    except Exception as exc:
        return None


# ===========================================================================
# SECTION 6 — Core async extraction function
# ===========================================================================

async def extract_page_with_gemini(
    page_jpeg_b64: str,
    page_num: int,
    document_type: str,
    board: str = "IGCSE",
    paper_reference_key: str = "",
    fallback_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Single-pass extraction of one exam page using Gemini 2.5 Flash.

    Parameters
    ----------
    page_jpeg_b64      : Base64-encoded JPEG of the rendered page.
    page_num           : 0-based page index.
    document_type      : "Question Paper" or "Marking Scheme".
    board              : "IGCSE" or "IB".
    paper_reference_key: Pre-generated key injected into the prompt.
    fallback_metadata  : Metadata dict to fill in fields the model may miss.

    Returns
    -------
    List[dict] — Each dict:
        {
            "model": ExtractedQuestion,
            "diagram_regions": [{"question_number": ..., "y_start_pct": ..., ...}],
            "page_num": int,
        }

    On total failure (all retries exhausted), returns [] — never raises.
    """
    if not page_jpeg_b64:
        return []

    fallback_metadata = fallback_metadata or {}
    is_ms = document_type.strip().lower() == "marking scheme"

    system_prompt = _build_system_prompt(document_type, board, paper_reference_key)

    local_qp_page_hints = fallback_metadata.get("local_qp_page_hints") or []
    page_expected_ids: List[str] = []
    if not is_ms and isinstance(local_qp_page_hints, list):
        page_hint = None
        if 0 <= page_num < len(local_qp_page_hints) and isinstance(local_qp_page_hints[page_num], dict):
            page_hint = local_qp_page_hints[page_num]
        if page_hint:
            printed_page_number = str(page_hint.get("printed_page_number") or page_num + 1)
            likely_root = str(page_hint.get("likely_root") or "").strip()
            expected_page_ids = [
                str(value).strip().lower()
                for value in (page_hint.get("expected_ids") or [])
                if str(value).strip()
            ]
            page_expected_ids = expected_page_ids
            visible_subparts = [
                str(value).strip().lower()
                for value in (page_hint.get("visible_subparts") or [])
                if str(value).strip()
            ]
            if likely_root or expected_page_ids:
                system_prompt += f"""

LOCAL PDF TEXT SKELETON FOR THIS PAGE (USE AS A HARD NUMBERING HINT):
- This rendered page's printed Cambridge page number is: {printed_page_number}
- The local text pass believes the visible/active question root on this page is: {likely_root or "unknown"}
- Expected saved-MS IDs likely belonging to this page/root: {", ".join(expected_page_ids[:24]) or "none"}
- Orphan subpart markers seen near the top of this page: {", ".join(visible_subparts[:8]) or "none"}

Numbering rule:
- Do NOT use the printed Cambridge page number as a question root.
- If your visual read conflicts with this local text skeleton, prefer the saved-MS/local skeleton unless the PDF visibly shows a different question label.
- For example, if printed page number is 2 but local root is 1, output 1(a), 1(b), etc., never 2(a).
"""
                logger.info(
                    "[GeminiSlicer] Page %s: Injecting local QP skeleton hint root=%r expected=%s.",
                    page_num,
                    likely_root,
                    len(expected_page_ids),
                )

    # ── FIX 3: ID INHERITANCE — Cross-page parent context injection ──────────
    # When a sub-question like '15(c)' starts at the top of a new page, the
    # parent root '15' is not visible to the model. Without this context, the
    # model outputs 'unknown(c)' or '(c)' — stripping the parent ID.
    # We inject the last known root question number from the previous page as
    # an explicit CONTEXT block appended to the system prompt. The model is then
    # instructed to use it for inheritance, not invent a new root number.
    last_parent = str(fallback_metadata.get("_last_known_parent_id") or "").strip()
    if last_parent:
        # Append as a clearly delimited context block so it overrides any
        # default "unknown" fallback the model might apply.
        # Compute next expected root safely (used in the prompt hint below)
        try:
            _next_root_hint = str(int(last_parent) + 1)
        except (ValueError, TypeError):
            _next_root_hint = "(next question number)"

        system_prompt += f"""

⚡ CONTEXT FROM PREVIOUS PAGE (MANDATORY — READ BEFORE EXTRACTING) ⚡
The last identified root question number on the PREVIOUS page was: {last_parent}

ID INHERITANCE RULE (NON-NEGOTIABLE):
- If this page begins with a sub-question such as '(b)', '(c)', '(ii)', or any
  sub-part WITHOUT a visible root integer, you MUST prepend '{last_parent}' to it.
  EXAMPLE: Page starts with '(c) Find the gradient.' → output ''{last_parent}(c) Find the gradient.''
- Do NOT output '(c)' alone. Do NOT output 'unknown(c)'. Do NOT invent a new root number.
- If a new root question IS visible (e.g. '{_next_root_hint}'), use that new number
  for all subsequent sub-parts — the inheritance context only applies until the next
  explicit root number appears.
- The page number printed at the TOP-CENTER of the page (e.g. "9", "11", "15") is NOT
  a root question number. It is the page number. Do NOT use it as a parent ID.
- Set needs_review: true on any question where you applied this inheritance rule,
  so a human reviewer can verify the assignment.
"""
        logger.info(
            f"[GeminiSlicer] Page {page_num}: Injecting parent context "
            f"'last_known_parent_id={last_parent}' for ID inheritance."
        )
    # ── END FIX 3 ────────────────────────────────────────────────────────────

    expected_canonical_ids = fallback_metadata.get("expected_canonical_ids") or []
    expected_ids_clean: List[str] = []
    if not is_ms and isinstance(expected_canonical_ids, list) and expected_canonical_ids:
        expected_ids_clean = [
            str(value).strip().lower()
            for value in expected_canonical_ids
            if str(value).strip()
        ]
        page_anchor_ids = page_expected_ids[:24]
        if page_anchor_ids:
            expected_id_text = ", ".join(page_anchor_ids)
            system_prompt += f"""

PAGE-LOCAL SAVED MARKING-SCHEME NUMBERING CONTRACT (MANDATORY):
The matching Marking Scheme has already been saved. For THIS rendered page,
the local PDF text pass expects only these canonical IDs:
{expected_id_text}

Use this page-local list as the numbering whitelist for this page.
- Prefer these IDs over any page-number-looking root emitted by the model.
- Do NOT copy IDs from other pages of the paper.
- Do NOT output an ID outside this page-local list unless the PDF visibly shows a different question label on this same page.
- If one visible QP block contains multiple expected subparts, split it into separate
  questions_array objects matching the expected IDs.
- If a printed QP parent contains child text that the MS splits, output child labels
  only when they are visibly present in the QP text.
"""
            logger.info(
                "[GeminiSlicer] Page %s: Injecting %s page-local saved MS canonical ID(s) as QP anchor.",
                page_num,
                len(page_anchor_ids),
            )

    rescue_missing_ids = fallback_metadata.get("rescue_missing_ids") or []
    rescue_ids_clean: List[str] = []
    if not is_ms and isinstance(rescue_missing_ids, list) and rescue_missing_ids:
        rescue_ids_clean = [
            str(value).strip().lower()
            for value in rescue_missing_ids
            if str(value).strip()
        ]
        page_rescue_ids: List[str] = rescue_ids_clean
        rescue_page_text = ""
        if isinstance(local_qp_page_hints, list) and 0 <= page_num < len(local_qp_page_hints):
            page_hint = local_qp_page_hints[page_num]
            if isinstance(page_hint, dict):
                hinted_ids = [
                    str(value).strip().lower()
                    for value in (page_hint.get("rescue_target_ids") or [])
                    if str(value).strip()
                ]
                if hinted_ids:
                    page_rescue_ids = hinted_ids
                rescue_page_text = str(page_hint.get("rescue_page_text_excerpt") or "").strip()
        if page_rescue_ids:
            system_prompt += f"""

TARGETED MISSING-ID RESCUE MODE (STRICT):
This is NOT a full page extraction. The user already extracted the paper, but
these exact QP canonical IDs are missing:
{", ".join(page_rescue_ids)}

Rules for this rescue call:
- Return questions_array objects ONLY for the target IDs listed above when their
  text is visible on this rendered page.
- If the page contains a grouped parent block that includes one of these target
  subparts, split out the target subpart as its own object and label it with the
  exact target ID.
- Do NOT output unrelated questions, sibling IDs, page numbers, or placeholder rows.
- Do NOT repair a target ID to a nearby different ID. Exact target ID or nothing.
- If none of the target IDs are visible on this page, return an empty
  questions_array.
"""
            if rescue_page_text:
                system_prompt += f"""

Native PDF text from this page for orientation only:
{rescue_page_text[:2500]}
"""
            logger.info(
                "[GeminiSlicer] Page %s: Targeted rescue mode for IDs=%s.",
                page_num,
                page_rescue_ids,
            )

    try:
        image_bytes = base64.b64decode(page_jpeg_b64)
    except Exception as exc:
        logger.error(f"[GeminiSlicer] Page {page_num}: base64 decode failed: {exc}")
        return []

    last_exc: Optional[Exception] = None
    model_plan = _model_attempt_plan(
        document_type,
        ms_anchor_active=bool(expected_ids_clean),
        deep_anchor_active=_has_deep_anchor_ids(page_expected_ids),
        targeted_rescue_active=bool(rescue_ids_clean),
    )
    total_attempts = len(model_plan)

    for attempt, model_name in enumerate(model_plan):
        try:
            # Gemini client is created OUTSIDE the semaphore context to avoid
            # holding the semaphore during the (fast but nonzero) client init.
            client = _get_client()

            async with _GEMINI_SEMAPHORE:
                response = await run_gemini_async(
                    lambda: asyncio.to_thread(
                        client.models.generate_content,
                        model=model_name,
                        contents=[
                            system_prompt,
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=_thinking_budget_for_document(document_type)
                            ),
                        ),
                    )
                )

            _log_gemini_usage(
                response,
                page_num=page_num,
                document_type=document_type,
                attempt=attempt + 1,
                model_name=model_name,
            )

            raw_text = (getattr(response, "text", "") or "").strip()
            logger.debug(
                f"[GeminiSlicer] Page {page_num} raw response "
                f"(first 500 chars): {raw_text[:500]}"
            )

            parsed = _parse_response(raw_text)
            questions_raw: List[Dict] = parsed.get("questions_array") or []
            page_metadata: Dict = parsed.get("metadata") or {}

            # Debug: count raw regions before any validation
            total_raw_regions = sum(
                len(q.get("diagram_regions") or [])
                for q in questions_raw
                if isinstance(q, dict)
            )
            logger.info(
                f"[GeminiSlicer] 🔍 Page {page_num}: "
                f"Gemini detected {total_raw_regions} raw visual region(s) across "
                f"{len(questions_raw)} question(s) | "
                f"doc_type={document_type!r} | is_ms={is_ms}"
            )
            if total_raw_regions == 0 and not is_ms and questions_raw:
                logger.warning(
                    f"[GeminiSlicer] ⚠️  Page {page_num}: Gemini returned 0 diagram "
                    f"regions for {len(questions_raw)} QP question(s). "
                    f"PyMuPDF fallback will run in the crop-injection step."
                )

            # Merge page-level metadata into fallback
            effective_meta = {**fallback_metadata, **page_metadata}
            if paper_reference_key:
                effective_meta.setdefault("paper_reference_key", paper_reference_key)

            # Apply page-specific metadata extraction rules
            strict_metadata = _extract_strict_metadata(effective_meta, page_num)
            logger.info(
                f"[StrictExtraction] Page {page_num}: Metadata = {strict_metadata}"
            )
            effective_meta.update(strict_metadata)

            result: List[Dict[str, Any]] = []

            for raw_q in questions_raw:
                if not isinstance(raw_q, dict):
                    continue

                # ── FIX B: Extract diagram_regions and validate via the unified path ──
                # For MS: diagram_regions is always [] (enforced in prompt).
                # The `is_ms` guard here is a defense-in-depth layer.
                raw_regions: List = raw_q.pop("diagram_regions", []) or []
                validated_regions: List[Dict] = []

                if not is_ms and isinstance(raw_regions, list):
                    # _extract_strict_diagrams → _validate_diagram_region (unified path)
                    validated_regions = _extract_strict_diagrams(raw_regions, page_num)

                if validated_regions:
                    logger.info(
                        f"[GeminiSlicer] Page {page_num} | q={raw_q.get('question_latex', '?')[:40]!r}: "
                        f"{len(validated_regions)}/{len(raw_regions)} region(s) passed validation"
                    )
                elif raw_regions and not is_ms:
                    logger.warning(
                        f"[GeminiSlicer] Page {page_num} | q={raw_q.get('question_latex', '?')[:40]!r}: "
                        f"All {len(raw_regions)} region(s) failed validation "
                        f"(PyMuPDF fallback will run)"
                    )

                # Build Pydantic model
                model = _build_extracted_question(
                    raw_q, document_type, effective_meta,
                    page_num=page_num,
                    last_known_parent_id=last_parent,
                )
                if model is None:
                    logger.warning(
                        f"[GeminiSlicer] Page {page_num}: skipped one question "
                        f"(model build failed). raw_q keys: {list(raw_q.keys())}"
                    )
                    continue

                result.append({
                    "model": model,
                    "diagram_regions": validated_regions,
                    "page_num": page_num,
                })

            logger.info(
                f"[GeminiSlicer] ✅ Page {page_num}: "
                f"{len(result)} question(s) extracted, "
                f"{sum(len(r['diagram_regions']) for r in result)} validated diagram region(s)."
            )
            return result

        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            is_fatal = any(phrase in err_str.lower() for phrase in _FATAL_ERROR_PHRASES)
            is_transient = any(code in err_str for code in _TRANSIENT_ERROR_CODES)
            if is_fatal:
                logger.error(
                    f"[GeminiSlicer] Page {page_num} fatal provider/billing error "
                    f"(attempt {attempt + 1}/{total_attempts}, model={model_name}); not retrying: {exc}"
                )
                record_gemini_failure(
                    model=model_name,
                    document_type=document_type,
                    page_num=page_num,
                    attempt=attempt + 1,
                    component="gemini_slicer",
                    error=str(exc),
                )
                break
            record_gemini_failure(
                model=model_name,
                document_type=document_type,
                page_num=page_num,
                attempt=attempt + 1,
                component="gemini_slicer",
                error=str(exc),
            )
            if attempt < total_attempts - 1:
                wait = (
                    (_RETRY_BASE_DELAY_S * (2 ** attempt)) + random.uniform(0.0, 1.5)
                    if is_transient
                    else 0.75 + random.uniform(0.0, 0.5)
                )
                logger.warning(
                    f"[GeminiSlicer] Page {page_num} provider/parse error "
                    f"(attempt {attempt + 1}/{total_attempts}, model={model_name}), "
                    f"retrying in {wait:.0f}s: {exc}"
                )
                await asyncio.sleep(wait)
            else:
                exhausted = "transient retries exhausted" if is_transient else "non-transient error"
                logger.error(
                    f"[GeminiSlicer] Page {page_num} {exhausted} "
                    f"(attempt {attempt + 1}/{total_attempts}, model={model_name}): {exc}"
                )
                break

    logger.error(
        f"[GeminiSlicer] ❌ Page {page_num} failed after {total_attempts} attempts. "
        f"Last error: {last_exc}"
    )
    return [{
        "_page_failed": True,
        "page_num": page_num,
        "error": str(last_exc),
    }]


# ===========================================================================
# SECTION 7 — Public batch interface
# ===========================================================================

def _extract_authoritative_metadata(
    all_page_results: List[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    TASK 1 HELPER — Extract the authoritative document metadata from page 0 (the
    cover page).

    Page 0 is the ONLY page that contains printed session/year/tier/subjectCode
    data.  All other pages are isolated from it during the concurrent gather and
    may hallucinate defaults (most commonly session="s" / Summer).

    Returns a dict with the four authoritative scalar fields, or None if page 0
    produced no extractable model (e.g. blank cover page with no questions).

    The returned dict intentionally contains ONLY the fields that must be
    forcefully propagated — not the full model dump — so callers can apply them
    with a targeted overwrite rather than a blanket merge.
    """
    if not all_page_results:
        return None

    page_0_results = all_page_results[0] if all_page_results else []

    for item in page_0_results:
        model = item.get("model")
        if model is None:
            continue

        session      = getattr(model, "session",     None)
        year         = getattr(model, "year",        None)
        tier         = getattr(model, "tier",        None)
        subject_code = getattr(model, "subjectCode", None)

        # Accept this item as authoritative if ANY meaningful field is populated.
        # "N/A" and 0 are non-authoritative sentinels — skip them.
        has_session      = session and str(session).strip() not in ("", "N/A", "None", "null")
        has_year         = year and int(year) > 0
        has_tier         = tier and str(tier).strip() not in ("", "N/A", "None", "null")
        has_subject_code = subject_code and str(subject_code).strip() not in ("", "N/A", "None", "null")

        if has_session or has_year or has_tier or has_subject_code:
            auth = {
                "session":     str(session).strip() if has_session else None,
                "year":        int(year) if has_year else None,
                "tier":        str(tier).strip() if has_tier else None,
                "subjectCode": str(subject_code).strip() if has_subject_code else None,
            }
            logger.info(
                f"[GeminiSlicer][MetaSync] Authoritative metadata locked from page 0: {auth}"
            )
            return auth

    logger.warning(
        "[GeminiSlicer][MetaSync] Page 0 yielded no authoritative metadata "
        "(blank cover page or all-N/A fields). No metadata sync will be applied."
    )
    return None


def _rewrite_reference_keys(
    model: Any,
    auth_session: Optional[str],
    auth_year: Optional[int],
) -> None:
    """
    TASK 1 HELPER — Rewrite paper_reference_key and unified_paper_key on *model*
    so that any hallucinated session letter embedded in those keys is replaced with
    the authoritative one from page 0.

    Key structure expected (lower-case):
        paper_reference_key : "igcse_0580_s20_42"
        unified_paper_key   : "igcse_0580_s20"   (or similar)

    The session letter occupies the position between the year digits and the
    subject/paper suffix, e.g. the "s" in "s20".  We match it with a regex
    anchored to the two-digit year so we never corrupt unrelated substrings.

    If auth_session or auth_year are None the key is left untouched.
    """
    if not auth_session or not auth_year:
        return

    year_2digit = str(auth_year)[-2:]   # e.g. 2020 → "20"

    # Regex: match any single lowercase letter immediately followed by the
    # two-digit year, capturing context so we can rebuild the surrounding string.
    # Example match in "igcse_0580_s20_42": group(1)="s", group(2)="20"
    key_pattern = re.compile(
        r'(?<=[_\-])([a-z])(' + re.escape(year_2digit) + r')(?=[_\-]|$)',
        re.IGNORECASE,
    )

    for attr in ("paper_reference_key", "unified_paper_key"):
        old_val = str(getattr(model, attr, "") or "")
        if not old_val:
            continue

        new_val = key_pattern.sub(
            lambda m: auth_session.lower() + m.group(2),
            old_val,
        )

        if new_val != old_val:
            try:
                setattr(model, attr, new_val)
                logger.debug(
                    f"[GeminiSlicer][MetaSync] Rewrote {attr}: "
                    f"{old_val!r} → {new_val!r}"
                )
            except Exception:
                pass  # Frozen / validated model — best effort; log and move on


def _sync_metadata_to_all_pages(
    all_page_results: List[List[Dict[str, Any]]],
    auth_meta: Dict[str, Any],
    document_type: str,
) -> None:
    """
    TASK 1 — Post-Gather Authoritative Metadata Sync.

    FIX v4-1: All `try: model.attr = val; except: pass` blocks have been REMOVED.
    ExtractedQuestion is a frozen Pydantic model — silent in-place attribute
    assignment was always raising ValidationError / TypeError and being swallowed,
    meaning the authoritative session from page 0 was NEVER actually written.

    Replacement: dump → patch dict → re-instantiate (same frozen-model-safe
    pattern used in the _validate_extracted_root reconstruction block).
    _rewrite_reference_keys is now inlined here so both scalars AND reference-key
    strings are patched inside the SAME model_dict before the single re-instantiation.

    Parameters
    ----------
    all_page_results : Raw output of asyncio.gather — list-of-lists, indexed by page.
    auth_meta        : Dict from _extract_authoritative_metadata (page-0 values).
    document_type    : Used for MS guard — MS documents never carry session metadata.
    """
    if not auth_meta:
        return

    auth_session      = auth_meta.get("session")
    auth_year         = auth_meta.get("year")
    auth_tier         = auth_meta.get("tier")
    auth_subject_code = auth_meta.get("subjectCode")

    # Pre-compile reference-key rewrite pattern once (outside loop).
    # Matches the session letter immediately before the two-digit year:
    # e.g. "s" in "igcse_0580_s21_qp_42"
    key_pattern: Optional[re.Pattern] = None
    if auth_session and auth_year:
        year_2digit = str(auth_year)[-2:]
        key_pattern = re.compile(
            r'(?<=[_\-])([a-z])(' + re.escape(year_2digit) + r')(?=[_\-]|$)',
            re.IGNORECASE,
        )

    total_patched = 0

    # Start from index 1 — page 0 IS the authority; never overwrite it.
    for page_idx, page_result in enumerate(all_page_results[1:], start=1):
        if not isinstance(page_result, list):
            continue

        for item in page_result:
            model = item.get("model")
            if model is None:
                continue

            # ── Step A: Dump frozen model to a mutable dict ───────────────────
            try:
                model_dict = (
                    model.model_dump()
                    if hasattr(model, "model_dump")
                    else model.dict()
                )
            except Exception as dump_exc:
                logger.error(
                    f"[GeminiSlicer][MetaSync] Page {page_idx}: "
                    f"model_dump() FAILED — {dump_exc!r}. Skipping item."
                )
                continue

            patched_fields: List[str] = []

            # ── Step B: Overwrite the four authoritative scalar fields ─────────
            if auth_session is not None:
                old = model_dict.get("session")
                if str(old or "").strip() != auth_session:
                    model_dict["session"] = auth_session
                    patched_fields.append(f"session({old!r}→{auth_session!r})")

            if auth_year is not None:
                old = model_dict.get("year")
                if old != auth_year:
                    model_dict["year"] = auth_year
                    patched_fields.append(f"year({old}→{auth_year})")

            if auth_tier is not None:
                old = model_dict.get("tier")
                if str(old or "").strip() != auth_tier:
                    model_dict["tier"] = auth_tier
                    patched_fields.append(f"tier({old!r}→{auth_tier!r})")

            if auth_subject_code is not None:
                old = model_dict.get("subjectCode")
                if str(old or "").strip() != auth_subject_code:
                    model_dict["subjectCode"] = auth_subject_code
                    patched_fields.append(f"subjectCode({old!r}→{auth_subject_code!r})")

            # ── Step C: Rewrite baked-in session letter in reference key strings
            # (was _rewrite_reference_keys — inlined here so it's part of the
            # same atomic dict patch before re-instantiation)
            if key_pattern is not None:
                for key_field in ("paper_reference_key", "unified_paper_key"):
                    old_val = str(model_dict.get(key_field) or "")
                    if not old_val:
                        continue
                    new_val = key_pattern.sub(
                        lambda m: auth_session.lower() + m.group(2),
                        old_val,
                    )
                    if new_val != old_val:
                        model_dict[key_field] = new_val
                        patched_fields.append(
                            f"{key_field}({old_val!r}→{new_val!r})"
                        )

            # Skip re-instantiation if nothing actually changed.
            if not patched_fields:
                continue

            # ── Step D: Re-instantiate the frozen model with corrected dict ────
            try:
                item["model"] = type(model)(**model_dict)
                total_patched += 1
                logger.debug(
                    f"[GeminiSlicer][MetaSync] Page {page_idx}: "
                    f"patched → {patched_fields}"
                )
            except Exception as recon_exc:
                # Log visibly — never swallow silently.
                logger.error(
                    f"[GeminiSlicer][MetaSync] Page {page_idx}: "
                    f"Re-instantiation FAILED for fields {patched_fields}: "
                    f"{recon_exc!r}. Original (unpatched) model retained."
                )

    logger.info(
        f"[GeminiSlicer][MetaSync] Sync complete — "
        f"{total_patched} question(s) across {len(all_page_results) - 1} "
        f"non-cover page(s) overwritten with authoritative metadata from "
        f"page 0: {auth_meta}"
    )


def _fix_orphan_question_id(
    model: Any,
    last_known_parent_id: str,
    page_num: int,
) -> Optional[Any]:
    """
    TASK 2 — Page-Break Orphan Numbering Fix.

    FIX v4-2: RETURN TYPE CHANGED from bool → Optional[model].
      OLD: Returned True and tried `model.question_id = corrected_id` (silent
           failure on frozen Pydantic model — correction was always lost).
      NEW: Returns a NEW re-instantiated model with corrected fields, or None if
           no correction is needed. The caller updates item["model"] with the
           returned value.

    Detects sub-questions with no root digit (e.g. "(b)", "c", "(ii)") and
    prepends last_known_parent_id to produce the fully-qualified ID ("3(b)").
    Both question_id and question_latex are corrected atomically in the same
    dict patch before the single re-instantiation.

    Returns:
        New corrected ExtractedQuestion model  — if orphan detected and fixed.
        None                                   — if no orphan, or fix failed.
    """
    if not last_known_parent_id:
        return None

    raw_id = str(getattr(model, "question_id", "") or "").strip()
    if not raw_id:
        return None

    # An orphan starts with a letter or open parenthesis — never a digit.
    # Examples: "b", "(b)", "c", "(ii)", "(a)(i)"
    is_orphan = bool(re.match(r'^(?:[a-zA-Z]|\()', raw_id))
    if not is_orphan:
        return None

    # Normalise: bare letter → wrapped: "b" → "(b)"; "(b)" / "(ii)" unchanged.
    context_parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(last_known_parent_id)
    if not context_parts:
        return None

    if re.match(r'^[a-zA-Z]+$', raw_id):
        orphan_parts = [raw_id.lower()]
    else:
        orphan_parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(raw_id)

    if not orphan_parts:
        return None

    orphan_token = orphan_parts[0].lower()
    if orphan_token in _QUESTION_NUMBER_NORMALIZER._ROMAN_SET:
        if len(context_parts) >= 3 and context_parts[-1].lower() in _QUESTION_NUMBER_NORMALIZER._ROMAN_SET:
            parent_parts = context_parts[:-1]
        elif len(context_parts) >= 2:
            parent_parts = context_parts
        else:
            parent_parts = context_parts
    elif len(orphan_token) == 1 and orphan_token.isalpha():
        parent_parts = context_parts[:1]
    else:
        parent_parts = context_parts

    corrected_parts = parent_parts + orphan_parts
    corrected_id = _QUESTION_NUMBER_NORMALIZER.format_parts(corrected_parts)

    logger.warning(
        f"[GeminiSlicer][OrphanFix] Page {page_num}: "
        f"Orphan detected — question_id={raw_id!r} has no root digit. "
        f"Using last hierarchy context={last_known_parent_id!r}: "
        f"'{raw_id}' → '{corrected_id}'. needs_review=True."
    )

    # ── Dump frozen model ─────────────────────────────────────────────────────
    try:
        model_dict = (
            model.model_dump()
            if hasattr(model, "model_dump")
            else model.dict()
        )
    except Exception as dump_exc:
        logger.error(
            f"[GeminiSlicer][OrphanFix] Page {page_num}: "
            f"model_dump() FAILED — {dump_exc!r}. Correction lost."
        )
        return None

    # ── Patch the dict ────────────────────────────────────────────────────────
    model_dict["question_id"]  = corrected_id
    model_dict["needs_review"] = True
    model_dict["canonical_question_id"] = _QUESTION_NUMBER_NORMALIZER.canonical_from_parts(corrected_parts)
    model_dict["parent_canonical_id"] = _QUESTION_NUMBER_NORMALIZER.parent_from_parts(corrected_parts)
    model_dict["question_number_metadata"] = _QUESTION_NUMBER_NORMALIZER.build_question_metadata(corrected_parts)

    # Also fix the leading ID prefix in question_latex (anchored to position 0
    # so mid-text occurrences of the sub-part label are never mangled).
    old_latex = str(model_dict.get("question_latex") or "")
    if old_latex:
        raw_label_pattern = (
            r'[\(\[]?\s*' + re.escape(raw_id) + r'\s*[\)\]]?'
            if re.match(r'^[A-Za-z]+$', raw_id)
            else re.escape(raw_id)
        )
        orphan_prefix = re.compile(
            r'^\s*' + raw_label_pattern + r'(?=\s|$)',
            re.IGNORECASE,
        )
        model_dict["question_latex"] = orphan_prefix.sub(
            corrected_id,
            old_latex,
            count=1,
        )

    # ── Re-instantiate ────────────────────────────────────────────────────────
    try:
        new_model = type(model)(**model_dict)
        return new_model
    except Exception as recon_exc:
        # Log visibly — never swallow silently.
        logger.error(
            f"[GeminiSlicer][OrphanFix] Page {page_num}: "
            f"Re-instantiation FAILED for '{raw_id}' → '{corrected_id}': "
            f"{recon_exc!r}. Correction lost."
        )
        return None


async def extract_pages_batch(
    page_jpeg_b64_list: List[str],
    document_type: str,
    board: str = "IGCSE",
    paper_reference_key: str = "",
    fallback_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Concurrently extract all pages in page_jpeg_b64_list.

    The _GEMINI_SEMAPHORE(3) inside each call ensures we never exceed 3 parallel
    Gemini API requests regardless of how many pages are in the document.

    FIX 3 (ID Inheritance): This function threads a `last_known_parent_id`
    value across pages. When a sub-question (e.g. '15(c)') wraps onto a new page
    its parent root ('15') is no longer visible to the model. By injecting the
    last root question from the previous page into the per-page prompt context,
    the model can correctly resolve '(c)' → '15(c)' instead of 'unknown(c)'.

    TASK 1 — Post-Gather Authoritative Metadata Sync:
    After asyncio.gather completes (and therefore BEFORE the sequential flatten),
    page 0 (the cover page) is interrogated for the authoritative session, year,
    tier, and subjectCode.  These four values are then forcefully written onto
    every question model on every subsequent page, eliminating the "Split-Brain"
    race condition where isolated concurrent tasks hallucinate session="s" because
    they never saw the cover page.  paper_reference_key and unified_paper_key
    strings that contain a baked-in (wrong) session letter are also rewritten.

    TASK 2 — Page-Break Orphan Numbering Fix:
    During the sequential post-processing pass, if the LLM extracted an orphaned
    sub-question ID (e.g. "b", "(b)", "(c)") with no root integer, and we have a
    valid last_known_parent_id, the parent is forcefully prepended.  Both
    model.question_id and model.question_latex are updated safely.

    FIX v3-1 (Jitter): _launch_with_stagger now adds random.uniform() jitter on
    top of the deterministic ramp. This prevents re-collisions in lockstep when
    multiple tasks retry after 503 errors — they re-arrive spread across a window
    instead of re-creating the original burst at the same wall-clock instant.

    FIX v3-3b (Pydantic Frozen Model): The was_corrected block now uses the
    dump→patch→re-instantiate pattern instead of silent in-place mutation.

    Returns a FLAT list of dicts in page order:
        [{"model": ExtractedQuestion, "diagram_regions": [...], "page_num": N}, ...]

    Failures on individual pages return [] for that page and are logged but
    never propagate to the caller — partial results are better than nothing.
    """
    if not page_jpeg_b64_list:
        return []

    fallback_metadata = fallback_metadata or {}
    expected_anchor_parts = _expected_anchor_parts(fallback_metadata)
    rescue_ids_for_plan = fallback_metadata.get("rescue_missing_ids") or []
    anchored_model_plan = _model_attempt_plan(
        document_type,
        ms_anchor_active=bool(expected_anchor_parts),
        deep_anchor_active=False,
        targeted_rescue_active=bool(rescue_ids_for_plan),
    )
    logger.warning(
        "[GeminiSlicerConfig] doc_type=%s pages=%s q_model_plan=%s "
        "page_number_leap_guard=on cache_bypass_depends_on_request_use_cache "
        "anchor_flash_first=%s rescue_flash_first=%s",
        document_type,
        len(page_jpeg_b64_list),
        anchored_model_plan,
        bool(expected_anchor_parts) and _QP_MS_ANCHOR_FLASH_FIRST,
        bool(rescue_ids_for_plan) and _QP_TARGETED_RESCUE_FLASH_FIRST,
    )
    local_qp_page_hints = fallback_metadata.get("local_qp_page_hints") or []
    if document_type.strip().lower() != "marking scheme":
        if expected_anchor_parts:
            logger.warning(
                "[GeminiSlicerConfig] saved_ms_anchor=on expected_ids=%s",
                len(expected_anchor_parts),
            )
        else:
            logger.warning(
                "[GeminiSlicerConfig] saved_ms_anchor=off expected_ids=0"
            )

    # ── FIX v3-1: Staggered concurrent launch with random jitter ─────────────
    #
    # Root cause of the 503 storm: asyncio.gather() with no delay launches ALL
    # coroutines simultaneously. Even with Semaphore(3), all N tasks start and
    # immediately contend for the semaphore at time=0. Gemini sees a burst of N
    # simultaneous connection attempts and rate-limits them, causing attempt-1
    # 503s across every page.
    #
    # The deterministic stagger (idx × _LAUNCH_STAGGER_S) smooths the arrival
    # rate from a pulse to a ramp. The random jitter on top prevents lockstep
    # re-collisions: without jitter, tasks that retry after a 503 all sleep the
    # same deterministic backoff and re-burst together at identical wall-clock
    # times. With jitter, retried tasks spread across a window and never re-create
    # the original burst pattern.
    # ────────────────────────────────────────────────────────────────────────

    _LAUNCH_STAGGER_S = 0.5   # 500 ms deterministic ramp between successive tasks
    _LAUNCH_JITTER_S  = 0.3   # max 300 ms random spread (uniform) on top of the ramp

    async def _launch_with_stagger(idx: int, b64: str) -> List[Dict[str, Any]]:
        """
        Delay task `idx` by (idx × stagger + jitter) seconds before calling the
        extractor.  idx=0 still receives jitter so the very first task does not
        fire at exactly t=0 alongside Python's own event-loop startup work.
        """
        # Always apply jitter; only apply the deterministic ramp for idx > 0.
        jitter = random.uniform(0.0, _LAUNCH_JITTER_S)
        delay  = (idx * _LAUNCH_STAGGER_S) + jitter
        if delay > 0:
            await asyncio.sleep(delay)
        return await extract_page_with_gemini(
            page_jpeg_b64=b64,
            page_num=idx,
            document_type=document_type,
            board=board,
            paper_reference_key=paper_reference_key,
            fallback_metadata=dict(fallback_metadata or {}),
        )

    # ── Pass 1: staggered concurrent extraction ───────────────────────────────
    # asyncio.gather preserved — all pages are still sent concurrently within
    # the Semaphore(3) cap.  The race condition is healed in the sync step below,
    # NOT by making this sequential.
    is_ms = document_type.strip().lower() == "marking scheme"

    all_page_results: List[List[Dict[str, Any]]] = list(
        await asyncio.gather(
            *[_launch_with_stagger(idx, b64) for idx, b64 in enumerate(page_jpeg_b64_list)],
            return_exceptions=False,
        )
    )

    failed_pages: List[Dict[str, Any]] = []
    for page_result in all_page_results:
        if not isinstance(page_result, list):
            continue
        for item in page_result:
            if isinstance(item, dict) and item.get("_page_failed"):
                failed_pages.append(item)

    if failed_pages:
        failed_nums = [int(item.get("page_num", -1)) + 1 for item in failed_pages]
        sample_error = str(failed_pages[0].get("error") or "")
        raise PipelineServiceError(
            stage="gemini_slicer",
            message=(
                "Gemini failed on one or more pages; refusing to return a partial "
                "paper extraction."
            ),
            details={
                "provider": "gemini",
                "failed_pages": failed_nums,
                "failed_page_count": len(failed_pages),
                "total_pages": len(page_jpeg_b64_list),
                "reason": sample_error[:500],
            },
        )

    # ── TASK 1: Post-Gather Authoritative Metadata Sync ───────────────────────
    #
    # POINT 2 — MS METADATA INHERITANCE:
    # The old gate `if not is_ms` has been replaced with a dual-path:
    #
    #   QP path: page 0 IS a cover page carrying session/year/tier. Scan it
    #            with _extract_authoritative_metadata as before.
    #
    #   MS path: MS files have no cover page carrying session data. The
    #            authoritative session/year/tier must come from the paired QP
    #            and is injected by the router via `fallback_metadata` before
    #            extract_pages_batch is called. We use that dict directly as
    #            auth_meta rather than scanning a cover page that can never
    #            produce the right values.
    #
    #            If the router did NOT inject the required fields (emergency
    #            fallback path, or legacy call site), we log a WARNING and skip
    #            rather than silently writing "N/A" to every MS question.
    # ─────────────────────────────────────────────────────────────────────────
    if not is_ms:
        # ── QP path (unchanged) ──────────────────────────────────────────────
        if len(all_page_results) > 1:
            auth_meta = _extract_authoritative_metadata(all_page_results)
            if auth_meta:
                _sync_metadata_to_all_pages(all_page_results, auth_meta, document_type)
    else:
        # ── MS path ──────────────────────────────────────────────────────────
        # Build auth_meta from the injected fallback_metadata. The router is
        # responsible for pre-populating these fields from the paired QP's
        # PaperRegistry entry before calling this function.
        _fb = fallback_metadata or {}
        ms_session      = _fb.get("session")
        ms_year         = _fb.get("year")
        ms_tier         = _fb.get("tier")
        ms_subject_code = _fb.get("subjectCode") or _fb.get("subject_code")

        has_ms_session = ms_session and str(ms_session).strip() not in ("", "N/A", "None", "null")
        has_ms_year    = ms_year and _coerce_int(ms_year, 0) > 0
        has_ms_tier    = ms_tier and str(ms_tier).strip() not in ("", "N/A", "None", "null")
        has_ms_sc      = ms_subject_code and str(ms_subject_code).strip() not in ("", "N/A", "None", "null")

        if has_ms_session or has_ms_year or has_ms_tier or has_ms_sc:
            ms_auth_meta: Dict[str, Any] = {}
            if has_ms_session:
                ms_auth_meta["session"] = str(ms_session).strip()
            if has_ms_year:
                ms_auth_meta["year"] = _coerce_int(ms_year, 0)
            if has_ms_tier:
                ms_auth_meta["tier"] = str(ms_tier).strip()
            if has_ms_sc:
                ms_auth_meta["subjectCode"] = str(ms_subject_code).strip()

            logger.info(
                f"[GeminiSlicer][MetaSync] MS path: using router-injected metadata "
                f"as authoritative source (no cover page scan): {ms_auth_meta}"
            )
            _sync_metadata_to_all_pages(all_page_results, ms_auth_meta, document_type)
        else:
            logger.warning(
                "[GeminiSlicer][MetaSync] MS path: fallback_metadata contains no "
                "usable session/year/tier. The router did not inject QP metadata "
                "before this call. MS questions will retain Gemini-extracted values "
                "(likely 'N/A'). Fix: call PaperRegistry.findOne for the paired QP "
                "and pass its session/year/tier in fallback_metadata before "
                "calling extract_pages_batch for the MS document."
            )
    # ── END TASK 1 ────────────────────────────────────────────────────────────

    # ── Pass 2: sequential scan — thread last_known_parent_id and apply ───────
    # post-extraction root corrections where the hallucination guard fires.
    # TASK 2 orphan detection also runs inside this same sequential pass.
    flat: List[Dict[str, Any]] = []
    last_known_parent_id: str = ""
    last_known_question_id: str = ""
    previous_model: Optional[Any] = None
    used_canonical_ids: set[str] = set()
    first_expected_qp_hint_index: Optional[int] = None
    if not is_ms and expected_anchor_parts and isinstance(local_qp_page_hints, list):
        for hint_idx, hint in enumerate(local_qp_page_hints):
            if isinstance(hint, dict) and hint.get("expected_ids"):
                first_expected_qp_hint_index = hint_idx
                break
    # FIX v6-3 (FLAT-ARRAY PROGRESSIVE CONSTRAINTS):
    # Tracks the canonical ID of the most recently written parent boundary
    # (e.g. "5.c" after "5(c)" is committed). Used to enforce that the next
    # child without an explicit indicator transitions to "5.c.i" rather than
    # stacking or colliding with an already-written key.
    last_written_parent_canonical: str = ""

    for idx, page_result in enumerate(all_page_results):
        if not isinstance(page_result, list):
            logger.warning(
                f"[GeminiSlicer] Page {idx} returned unexpected type: {type(page_result)}"
            )
            continue

        _page_hint: Dict[str, Any] = {}
        if (
            not is_ms
            and isinstance(local_qp_page_hints, list)
            and 0 <= idx < len(local_qp_page_hints)
            and isinstance(local_qp_page_hints[idx], dict)
        ):
            _page_hint = local_qp_page_hints[idx]
        _page_hint_expected_ids = [
            str(value).strip().lower()
            for value in (_page_hint.get("expected_ids") or [])
            if str(value).strip()
        ]
        _page_hint_printed_number = str(_page_hint.get("printed_page_number") or idx + 1).strip()
        _page_hint_likely_root = str(_page_hint.get("likely_root") or "").strip()
        if not _page_hint_likely_root and _page_hint_expected_ids:
            _page_hint_likely_root = _page_hint_expected_ids[0].split(".", 1)[0]
        _page_hint_expected_roots = {
            value.split(".", 1)[0]
            for value in _page_hint_expected_ids
            if value
        }

        for item in page_result:
            model = item.get("model")
            if model is None:
                flat.append(item)
                continue

            # ── POINT 1 — UNIFIED CONTINUITY GUARD ───────────────────────────
            # The old `if document_type != "marking scheme"` gate has been
            # removed. Orphan detection and the poisoning guard are
            # document-agnostic: an MS page can equally produce a bare "(b)"
            # orphan or a "[4]"-sourced ghost root. The only document-specific
            # logic in this file is diagram extraction, which is already gated
            # correctly in extract_page_with_gemini via `is_ms`.
            # ─────────────────────────────────────────────────────────────────
            raw_id = str(getattr(model, "question_id", "") or "").strip()
            if (
                not is_ms
                and expected_anchor_parts
                and first_expected_qp_hint_index is not None
                and idx < first_expected_qp_hint_index
                and not _page_hint_expected_ids
            ):
                logger.warning(
                    "[GeminiSlicer][MSAnchorPageGate] Page %s: keeping pre-anchor "
                    "QP row raw_id=%r even though saved-MS page hints start at page %s. "
                    "Local page hints are advisory and must not delete extracted rows.",
                    idx,
                    raw_id,
                    first_expected_qp_hint_index,
                )

            # ══════════════════════════════════════════════════════════════════
            # FIX v6-1 — INTENTIONAL DIGIT FLUSH
            # ══════════════════════════════════════════════════════════════════
            # If the extracted question_id (or question_latex) explicitly contains
            # a root digit token — e.g. "7(c)(i)", "8(a)(i)" — we MUST flush
            # last_known_parent_id to that digit BEFORE running either the orphan
            # fix or the hallucination guard.
            #
            # Root cause of the "8(a)(i) → 5.a.i" state drift:
            #   last_known_parent_id is stale at "5". Gemini correctly emits
            #   "8(a)(i)". _validate_extracted_root fires because root(8) ==
            #   cambridge_page_number(8) AND remainder "(a)(i)" starts with "(a)"
            #   … actually Signal 2 only fires for non-(a) subparts, but SIGNAL 1
            #   still fires if "8" is not 5 or 5+1 — replacing "8(a)(i)" with
            #   "5(a)(i)". The tracker was never updated with the explicit digit.
            #
            # Fix: if raw_id starts with an explicit root digit, extract it and
            # flush last_known_parent_id to that digit NOW — before any guard can
            # overwrite the explicit text token with the stale state.
            #
            # The orphan-prepend logic (TASK 2 below) ONLY runs when the text
            # block is completely void of any root digit — this guard guarantees
            # that invariant is enforced here rather than relying solely on the
            # is_orphan check inside _fix_orphan_question_id.
            _page_hint_root_conflict = False
            _anchor_min_root_for_repair = None
            _early_root_match = re.match(r'^(\d+)', raw_id)
            if _early_root_match:
                _explicit_root   = _early_root_match.group(1)
                # Remainder is everything after the leading digits.
                # "8(a)(i)" → remainder="(a)(i)"  (has a subpart suffix)
                # "2"        → remainder=""        (bare integer, no suffix)
                _flush_remainder = raw_id[len(_explicit_root):]
                _has_subpart     = bool(_flush_remainder.strip())
                try:
                    _explicit_root_int  = int(_explicit_root)
                    _last_int_for_flush = int(last_known_parent_id) if last_known_parent_id else -1

                    # ── Plausibility gate (FIX v6-1c — Ghost-N at low Q numbers) ──
                    # Two acceptance tiers:
                    #
                    # TIER A — rooted ID with a subpart suffix, e.g. "8(a)(i)":
                    #   Accept same OR any forward move (>= last).
                    #   Rationale: a genuine question with an explicit subpart like
                    #   "8(a)(i)" can only have come from reading the column-1 label.
                    #   The stale-tracker state-drift bug requires this permissive
                    #   acceptance to fix "8(a)(i) → 5.a.i" misclassification.
                    #
                    # TIER B — bare integer with NO subpart suffix, e.g. "2" or "4":
                    #   Accept same OR strictly +1 only (== last OR == last + 1).
                    #   Rationale: a bare integer without a subpart is the exact
                    #   fingerprint of a mark-bracket artefact ("[2]" after Q1,
                    #   "[4]" after Q12). A legitimate new root question ALWAYS
                    #   appears with at least a "(a)" suffix or preamble text on
                    #   the same line — it is never a lone digit in the MS table.
                    #   Accepting "[2]" after Q1 as a forward flush poisons the
                    #   tracker, turning the next orphan "(c)" into "2(c)" instead
                    #   of "1(c)". This is the Ghost-N-at-low-Q-numbers bug.
                    #
                    # Both tiers still hard-reject backward jumps (< last).

                    if _has_subpart:
                        # TIER A: rooted with suffix — forward or same, except
                        # for page-number hallucinations. Lite can output the
                        # printed page number as a plausible +1 root (e.g.
                        # 9(ii) while the local skeleton says the active root
                        # is 6), so the local skeleton must veto that root.
                        _cambridge_page_number = idx + 1
                        _page_hint_root_conflict = (
                            not is_ms
                            and bool(_page_hint_likely_root)
                            and _explicit_root == _page_hint_printed_number
                            and _page_hint_likely_root != _explicit_root
                        )
                        _is_page_number_leap = (
                            _explicit_root_int == _cambridge_page_number
                            and (
                                (
                                    _last_int_for_flush >= 0
                                    and _explicit_root_int > _last_int_for_flush + 1
                                )
                                or (
                                    _last_int_for_flush == -1
                                    and bool(expected_anchor_parts)
                                    and expected_anchor_parts[0]
                                    and str(expected_anchor_parts[0][0]) != _explicit_root
                                )
                                or _page_hint_root_conflict
                            )
                        )
                        _anchor_min_root_for_repair = _last_int_for_flush if _last_int_for_flush >= 0 else None
                        if _page_hint_root_conflict:
                            try:
                                _anchor_min_root_for_repair = int(_page_hint_likely_root)
                            except (ValueError, TypeError):
                                _anchor_min_root_for_repair = None
                        _flush_accepted = (
                            _explicit_root_int >= _last_int_for_flush
                            and not _is_page_number_leap
                        )
                        if _is_page_number_leap:
                            _reject_reason = (
                                f"page-number leap {_last_int_for_flush} → "
                                f"{_explicit_root_int} on Cambridge page "
                                f"{_cambridge_page_number}"
                            )
                            if _page_hint_root_conflict:
                                _reject_reason += (
                                    f"; local QP skeleton expects root "
                                    f"{_page_hint_likely_root}"
                                )
                        elif not _flush_accepted:
                            _reject_reason = (
                                f"backward jump {_last_int_for_flush} → {_explicit_root_int}"
                            )
                            if (
                                not is_ms
                                and _page_hint_expected_roots
                                and _explicit_root in _page_hint_expected_roots
                            ):
                                _flush_accepted = True
                                _reject_reason = ""
                                logger.warning(
                                    "[GeminiSlicer][MSAnchorPageHintOverride] Page %s: "
                                    "accepting backward root %r because saved-MS local "
                                    "page hint expects root(s) %s. Previous tracker=%r.",
                                    idx,
                                    _explicit_root,
                                    sorted(_page_hint_expected_roots),
                                    last_known_parent_id,
                                )
                        else:
                            _reject_reason = ""
                    else:
                        # TIER B: bare integer — acceptance depends on document type.
                        #
                        # QP: accept same OR strictly +1.
                        #   A QP can have a preamble row with just the root number
                        #   (e.g. "2" introducing Q2's shared context). So Q2 bare
                        #   after Q1 is legitimate.
                        #
                        # MS: accept same ONLY.
                        #   Cambridge MS tables NEVER have a standalone root-only
                        #   row. Every MS row has at least a "(a)" subpart suffix.
                        #   A bare integer in an MS question_id is ALWAYS a mark-
                        #   bracket artefact ("[2]" → "2", "[4]" → "4").
                        #   Accepting it as a +1 step is the Ghost-N-at-low-Q bug
                        #   (e.g. "[2]" after Q1 passes >= check, poisons tracker
                        #   to "2", next orphan "(c)" becomes "2(c)" not "1(c)").
                        if is_ms:
                            _flush_accepted = (
                                _explicit_root_int == _last_int_for_flush  # same Q only
                                or _last_int_for_flush == -1               # first Q
                            )
                        else:
                            _flush_accepted = (
                                _explicit_root_int == _last_int_for_flush       # same Q
                                or _explicit_root_int == _last_int_for_flush + 1  # next Q
                                or _last_int_for_flush == -1                    # first Q
                            )
                        _reject_reason = ""
                        if not _flush_accepted:
                            if _explicit_root_int < _last_int_for_flush:
                                _reject_reason = (
                                    f"backward jump {_last_int_for_flush} → "
                                    f"{_explicit_root_int} (bare integer)"
                                )
                            elif is_ms and _explicit_root_int == _last_int_for_flush + 1:
                                _reject_reason = (
                                    f"MS bare-integer +1 step {_last_int_for_flush} → "
                                    f"{_explicit_root_int} rejected (MS rows never have "
                                    f"standalone root — likely mark-bracket '[{_explicit_root}]')"
                                )
                            else:
                                _reject_reason = (
                                    f"bare-integer leap {_last_int_for_flush} → "
                                    f"{_explicit_root_int} (gap > 1, likely mark-bracket "
                                    f"artefact e.g. '[{_explicit_root}]')"
                                )

                    if _flush_accepted:
                        # ── ACCEPT — flush tracker to explicit root ───────────
                        # FIX v6-2 (STALENESS GATING): explicit image token wins
                        # over historical state, for both tiers.
                        if _explicit_root != last_known_parent_id:
                            logger.info(
                                f"[GeminiSlicer][DigitFlush] Page {idx}: "
                                f"Explicit digit '{_explicit_root}' in "
                                f"raw_id={raw_id!r} flushes stale tracker "
                                f"'{last_known_parent_id}' → '{_explicit_root}' "
                                f"({'rooted+suffix' if _has_subpart else 'bare+1 step'})."
                            )
                        last_known_parent_id = _explicit_root
                        # Refresh full-context tracker so orphan fix (TASK 2)
                        # and downstream code inherit the correct hierarchy.
                        _flush_parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(raw_id)
                        _flush_ctx   = _QUESTION_NUMBER_NORMALIZER.format_parts(_flush_parts)
                        if _flush_ctx:
                            last_known_question_id = _flush_ctx

                    else:
                        # ── REJECT — POISONING GUARD ─────────────────────────
                        # Log and leave tracker unchanged. The downstream
                        # _validate_extracted_root guard handles correction using
                        # the intact tracker value.
                        _anchor_repair_applied = False
                        if (
                            not is_ms
                            and _has_subpart
                            and expected_anchor_parts
                            and "_is_page_number_leap" in locals()
                            and _is_page_number_leap
                        ):
                            _original_raw_id = raw_id
                            _raw_parts_for_anchor = _QUESTION_NUMBER_NORMALIZER.extract_parts(raw_id)
                            _anchor_parts = None
                            # When the local QP skeleton knows this page's expected
                            # IDs, it is the hard boundary for page-number repair.
                            # Do not use a global suffix match here: that caused
                            # printed page "5(a)" to jump to unrelated future IDs
                            # like 8.a just because "(a)" was the next missing suffix.
                            if _page_hint_expected_ids:
                                _anchor_parts = _next_expected_anchor_for_page(
                                    expected_parts=expected_anchor_parts,
                                    used_canonical_ids=used_canonical_ids,
                                    page_expected_ids=_page_hint_expected_ids,
                                )
                                if _anchor_parts and len(_anchor_parts) != len(_raw_parts_for_anchor):
                                    # A rooted subpart like 5(c)(i) cannot safely
                                    # collapse into a page hint that only says 6.
                                    # Leave it for the normal guard/review path.
                                    logger.warning(
                                        "[GeminiSlicer][MSAnchorPageNumberRepair] Page %s: "
                                        "refusing page-hint repair raw_id=%r to %r because "
                                        "the hierarchy depth does not match.",
                                        idx,
                                        _original_raw_id,
                                        _QUESTION_NUMBER_NORMALIZER.format_parts(_anchor_parts),
                                    )
                                    _anchor_parts = None
                            else:
                                _anchor_parts = _next_expected_anchor_for_suffix(
                                    expected_parts=expected_anchor_parts,
                                    used_canonical_ids=used_canonical_ids,
                                    raw_suffix_parts=_raw_parts_for_anchor[1:],
                                    min_root=locals().get("_anchor_min_root_for_repair"),
                                )
                            if _anchor_parts:
                                _anchor_label = _QUESTION_NUMBER_NORMALIZER.format_parts(_anchor_parts)
                                _anchor_canonical = _canonical_from_parts(_anchor_parts)
                                _anchor_model = _rebuild_model_with_anchor_id(
                                    model=model,
                                    corrected_parts=_anchor_parts,
                                    warning=(
                                        f"MS anchor corrected page-number hallucination "
                                        f"{_original_raw_id!r} to {_anchor_label!r}."
                                    ),
                                )
                                if _anchor_model is not None:
                                    item["model"] = _anchor_model
                                    model = _anchor_model
                                    raw_id = str(getattr(model, "question_id", "") or "").strip()
                                    last_known_parent_id = str(_anchor_parts[0])
                                    last_known_question_id = _anchor_label
                                    _anchor_repair_applied = True
                                    logger.warning(
                                        f"[GeminiSlicer][MSAnchorPageNumberRepair] Page {idx}: "
                                        f"raw_id={_original_raw_id!r} matched printed Cambridge "
                                        f"page {_explicit_root_int}; repaired to saved-MS ID "
                                        f"{_anchor_canonical!r}."
                                    )

                        if not _anchor_repair_applied:
                            logger.warning(
                                f"[GeminiSlicer][DigitFlush] Page {idx}: "
                                f"POISONING GUARD — digit flush rejected. "
                                f"raw_id={raw_id!r}, explicit root='{_explicit_root}', "
                                f"reason={_reject_reason}. "
                                f"Tracker stays at '{last_known_parent_id}'."
                            )

                except ValueError:
                    pass  # Non-numeric digit string — leave tracker unchanged.
            # ── END FIX v6-1 / v6-2 / v6-1c ─────────────────────────────────

            # ── TASK 2: Orphan sub-question fix ──────────────────────────────
            # FIX v4-2: _fix_orphan_question_id returns the corrected model
            # (or None). We MUST update item["model"] and the local `model`
            # variable here — the old bool-return + in-place mutation was a
            # silent no-op on frozen Pydantic models.
            #
            # Post-v6-1: this block only executes when raw_id has NO root digit
            # (the digit-flush above does not set _early_root_match for orphans
            # like "(b)" or "c"), preserving the original orphan-detection contract.
            if raw_id and last_known_parent_id and not _early_root_match:
                corrected_orphan = _fix_orphan_question_id(
                    model, last_known_question_id or last_known_parent_id, idx
                )
                if corrected_orphan is not None:
                    item["model"] = corrected_orphan
                    model         = corrected_orphan
                    # Re-read corrected ID so the hallucination guard below
                    # sees "3(b)" rather than the bare orphan "(b)".
                    raw_id = str(
                        getattr(model, "question_id", "") or ""
                    ).strip()
            # ── END TASK 2 ────────────────────────────────────────────────────

            # ── Post-extraction page-number hallucination guard ───────────────
            # Runs for both QP and MS (gate removed — see POINT 1 above).
            #
            # FIX v6-2 (STALENESS GATING — defense layer):
            # After the digit flush above, last_known_parent_id already reflects
            # the explicit root. _validate_extracted_root uses last_known_parent_id
            # as the "expected" anchor for its SIGNAL 1 plausibility check.
            # Because the tracker was already flushed to the explicit root, the
            # guard will now evaluate "8(a)(i)" against parent "8" (not stale "5"),
            # so Signal 1's continuity test (8 == 8 → same question) will correctly
            # classify it as non-hallucinatory.
            if raw_id and last_known_parent_id:
                corrected_id, was_corrected = _validate_extracted_root(
                    raw_id, last_known_parent_id, idx
                )
                if was_corrected:
                    # SAFE RE-INSTANTIATION: Prevents Pydantic silent assignment failures
                    try:
                        model_dict = model.model_dump() if hasattr(model, 'model_dump') else model.dict()
                        model_dict["question_id"] = corrected_id
                        model_dict["needs_review"] = True
                        item["model"] = type(model)(**model_dict)
                        model = item["model"]
                    except Exception as e:
                        logger.error(f"[GeminiSlicer] Failed to re-instantiate model for corrected ID: {e}")

            _apply_backward_root_guard(
                item=item,
                previous_model=previous_model,
                used_canonical_ids=used_canonical_ids,
                last_known_parent_id=last_known_parent_id,
                page_num=idx,
            )

            if not is_ms:
                _align_qp_model_to_visible_label(item, used_canonical_ids)

            _apply_sequential_duplicate_guard(
                item=item,
                previous_model=previous_model,
                used_canonical_ids=used_canonical_ids,
                page_num=idx,
                is_ms=is_ms,
            )

            # ══════════════════════════════════════════════════════════════════
            # FIX v6-3 — FLAT-ARRAY PROGRESSIVE CONSTRAINTS
            # ══════════════════════════════════════════════════════════════════
            # When parsing nested subparts, if a parent boundary (e.g. "5.c")
            # has just been committed to the flat array, the NEXT child without
            # explicit indicators must logically transition to "5.c.i" rather
            # than stacking (emitting "5.c" again) or colliding with an already-
            # used key.
            #
            # Mechanism:
            #   1. After each item is appended, compute the committed model's
            #      canonical parts.
            #   2. If the committed item is a multi-depth node (len(parts) >= 2),
            #      record it as last_written_parent_canonical.
            #   3. If the NEXT item in the same page produces a canonical that is
            #      already in used_canonical_ids AND its parts match the last
            #      written parent exactly, pre-seed used_canonical_ids with the
            #      parent itself so _apply_sequential_duplicate_guard correctly
            #      advances to the first child (.i) rather than emitting a sib.
            #
            # This is an index-check gate, not a content rewrite — it only
            # constrains which canonical IDs are already "occupied" before the
            # duplicate guard runs on the next iteration.
            _committed_model = item.get("model")
            if _committed_model is not None:
                _committed_parts = _question_parts_from_model(_committed_model)
                _committed_canonical = _canonical_from_parts(_committed_parts)
                if _committed_canonical and len(_committed_parts) >= 2:
                    # This is a nested node — record it as the last written
                    # parent boundary for the progressive constraint.
                    last_written_parent_canonical = _committed_canonical
                    # Pre-occupy the parent canonical so any subsequent emission
                    # of the SAME key is immediately treated as a duplicate by
                    # _apply_sequential_duplicate_guard and advanced to the next
                    # child (e.g. "5.c.i") instead of colliding at "5.c".
                    used_canonical_ids.add(_committed_canonical)
            # ── END FIX v6-3 ─────────────────────────────────────────────────

            flat.append(item)
            previous_model = item.get("model")

            # ── Update last_known_parent_id with POISONING GUARD ─────────────
            # FIX v4-4: Only accept a root digit that is a PLAUSIBLE continuation
            # of the sequence — same number (multi-page question) or any forward
            # step. REJECT backward jumps unconditionally.
            #
            # Root cause of the "Ghost 4" bug:
            #   The LLM reads a mark-bracket "[4]" at line-end and emits
            #   question_id="4". After Q12, this is a backward jump (4 < 12).
            #   OLD code accepted it, poisoning last_known_parent_id = "4".
            #   Every subsequent orphan "(iii)" became "4(iii)".
            #   NEW code rejects the backward jump — tracker stays at "12" —
            #   so the next orphan correctly becomes "12(iii)".
            current_model = item.get("model")
            q_id = str(getattr(current_model, "question_id", "") or "").strip()
            if q_id:
                q_parts = _QUESTION_NUMBER_NORMALIZER.extract_parts(q_id)
                q_full_context = _QUESTION_NUMBER_NORMALIZER.format_parts(q_parts)
                root_match = re.match(r"^(\d+)", q_id)
                if root_match:
                    candidate_root = root_match.group(1)
                    if not last_known_parent_id:
                        # First question seen — always accept.
                        last_known_parent_id = candidate_root
                        if q_full_context:
                            last_known_question_id = q_full_context
                    else:
                        try:
                            candidate_int = int(candidate_root)
                            last_int      = int(last_known_parent_id)
                            cambridge_page_number = idx + 1
                            final_page_hint_root_conflict = (
                                not is_ms
                                and bool(_page_hint_likely_root)
                                and candidate_root == _page_hint_printed_number
                                and _page_hint_likely_root != candidate_root
                            )
                            is_page_number_leap = (
                                candidate_int == cambridge_page_number
                                and (
                                    candidate_int > last_int + 1
                                    or final_page_hint_root_conflict
                                )
                            )
                            if (
                                candidate_int >= last_int
                                and not is_page_number_leap
                            ) or (
                                not is_ms
                                and _page_hint_expected_roots
                                and candidate_root in _page_hint_expected_roots
                            ):
                                # Forward or same — plausible continuation, unless
                                # it is a large jump to the printed page number.
                                # Saved-MS local page hints can also reset a
                                # tracker poisoned by an earlier hallucinated row.
                                last_known_parent_id = candidate_root
                                if q_full_context:
                                    last_known_question_id = q_full_context
                            else:
                                # Backward jump — highly suspicious; reject.
                                reason = (
                                    f"page-number leap {last_int} → {candidate_int}"
                                    if is_page_number_leap
                                    else f"backward jump {last_int} → {candidate_int}"
                                )
                                if final_page_hint_root_conflict:
                                    reason += f"; local QP skeleton expects root {_page_hint_likely_root}"
                                logger.warning(
                                    f"[GeminiSlicer] Page {idx}: "
                                    f"POISONING GUARD — rejected "
                                    f"last_known_parent_id update from "
                                    f"'{last_known_parent_id}' to "
                                    f"'{candidate_root}' ({reason}). "
                                    f"Tracker unchanged."
                                )
                        except ValueError:
                            # Non-numeric root — don't update tracker.
                            pass

    if is_ms:
        before_merge_count = len(flat)
        flat = _repair_and_merge_ms_rowspan_continuations(flat)
        if len(flat) != before_merge_count:
            logger.info(
                f"[GeminiSlicer][MSRowspan] Merged {before_merge_count - len(flat)} "
                f"row-span continuation row(s) into their visible MS labels."
            )

    logger.info(
        f"[GeminiSlicer] Batch complete: {len(page_jpeg_b64_list)} page(s), "
        f"{len(flat)} total question(s) extracted."
    )
    if not is_ms and expected_anchor_parts:
        final_ids: List[str] = []
        for item in flat:
            model = item.get("model") if isinstance(item, dict) else None
            if model is None:
                continue
            parts = _question_parts_from_model(model)
            canonical = _canonical_from_parts(parts)
            if canonical:
                final_ids.append(canonical)
        expected_ids = [_canonical_from_parts(parts) for parts in expected_anchor_parts]
        expected_ids = [value for value in expected_ids if value]
        final_set = set(final_ids)
        expected_set = set(expected_ids)
        missing = [value for value in expected_ids if value not in final_set]
        extras = [value for value in final_ids if value not in expected_set]
        logger.warning(
            "[MSAnchorTrace][SlicerExit] rows=%s unique=%s expected=%s "
            "exact=%s missing=%s extras=%s first_ids=%s last_ids=%s",
            len(final_ids),
            len(final_set),
            len(expected_ids),
            len(final_set & expected_set),
            missing[:40],
            extras[:40],
            final_ids[:12],
            final_ids[-8:],
        )
    return flat


# ===========================================================================
# SECTION 8 — Strict JSON output builder
# ===========================================================================

def build_strict_json_output(
    extraction_results: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build STRICT JSON output:
    {
      "metadata": { "subject_code": "...", "paper_number": "...", "session": "...", "year": "..." },
      "questions_array": [
        { "canonical_id": "...", "diagram_regions": [{ "y_start": ..., "y_end": ... }] }
      ]
    }
    """
    strict_metadata = {
        "subject_code": "N/A",
        "paper_number": "N/A",
        "session": "N/A",
        "year": "N/A",
    }

    if metadata:
        strict_metadata.update({
            "subject_code": metadata.get("subject_code", "N/A"),
            "paper_number": metadata.get("paper_number", "N/A"),
            "session": metadata.get("session", "N/A"),
            "year": metadata.get("year", "N/A"),
        })

    questions_array = []

    for result in extraction_results:
        if not isinstance(result, dict):
            continue

        model = result.get("model")
        diagram_regions = result.get("diagram_regions", [])

        if model is None:
            continue

        canonical_id = (
            str(model.question_id or model.question_latex or "unknown")
        ).strip()

        strict_regions = []
        for region in diagram_regions:
            strict_region = {
                "y_start": region.get("y_start_pct"),
                "y_end": region.get("y_end_pct"),
            }
            if region.get("clipping_risk"):
                strict_region["clipping_risk"] = True
            if region.get("question_number"):
                strict_region["question_number"] = region["question_number"]
            if region.get("visual_description"):
                strict_region["visual_description"] = region["visual_description"]

            strict_regions.append(strict_region)

        questions_array.append({
            "canonical_id": canonical_id,
            "diagram_regions": strict_regions,
        })

    return {
        "metadata": strict_metadata,
        "questions_array": questions_array,
    }


__all__ = ["extract_page_with_gemini", "extract_pages_batch", "build_strict_json_output"]
