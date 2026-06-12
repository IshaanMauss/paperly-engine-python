import time
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _sanitize_filename_for_parsing(raw: str) -> str:
    """
    Pre-processing sanitization pipeline. Mirrors JS _sanitizeFileNameForParsing.

    Transformation example:
      "0580_s18_ms_4 (Extended)2 (1).pdf"
      → lowercase           → "0580_s18_ms_4 (extended)2 (1).pdf"
      → strip tier tags     → "0580_s18_ms_42 (1).pdf"
      → strip dupe markers  → "0580_s18_ms_42 .pdf"
      → collapse whitespace → "0580_s18_ms_42.pdf"
    """
    s = raw.lower()
    s = re.sub(r'\s*\((extended|core|higher|foundation|standard)\)\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*\(\d+\)\s*', '', s)
    s = re.sub(r'\s+', '', s)
    return s

def _extract_session_from_content(content_str: str) -> str | None:
    """
    Extract session code ('m', 's', 'w') from a content string.

    Handles two input forms:

    1. Single-letter authoritative code — "m", "s", or "w" exactly (case-insensitive).
       These are emitted by gemini_slicer after its authoritative metadata sync and must
       pass through immediately.  The previous implementation treated these as unknown
       strings and returned None, causing generate_paper_keys to fall back to the
       filename session ("s" from "s18") and silently overwrite the correct AI session.

    2. Full-word / phrase strings — "February/March 2018", "may/june", "November", etc.
       Matched with word-boundary regex to prevent substring false positives
       (e.g. "Maximum Mark 130" must NOT match "mar" inside "Mark").

    Safe against false positives:
      "m18"    → None  (single-letter check requires content_lower == "m" exactly)
      "ms"     → None
      "sm"     → None
      "mark"   → None  (word-boundary regex, no match)
    """
    if not content_str:
        return None
    content_lower = str(content_str).lower().strip()

    # ── TIER 1: Single-letter authoritative code passthrough ─────────────────
    # Must be EXACTLY one character to avoid matching "m18", "ms", "sm" etc.
    if content_lower in ("m", "s", "w"):
        return content_lower

    # ── TIER 2: Full-word / phrase extraction ────────────────────────────────
    # Word boundaries prevent substring matches (e.g. "mar" inside "Mark").
    if re.search(r'\b(march|february|feb)\b', content_lower):
        return "m"
    elif re.search(r'\b(may|june|jun|summer)\b', content_lower):
        return "s"
    elif re.search(r'\b(october|november|oct|nov|winter)\b', content_lower):
        return "w"
    return None

def _extract_session_from_content(content_str: str) -> str | None:
    """
    Metadata-aware Cambridge session mapper.

    Priority:
      1. exact code: m / s / w
      2. explicit printed phrases: Feb/March, May/June/July, Oct/Nov
      3. month words near a printed year

    This intentionally ignores generic prose like "marks may be awarded".
    """
    if not content_str:
        return None

    content_lower = str(content_str).lower().strip()
    if content_lower in ("m", "s", "w"):
        return content_lower

    if re.search(r'\b(feb(?:ruary)?|mar(?:ch)?)\s*/\s*(mar(?:ch)?|feb(?:ruary)?)\b', content_lower):
        return "m"
    if re.search(r'\b(may|jun(?:e)?|jul(?:y)?)\s*/\s*(jun(?:e)?|jul(?:y)?)\b', content_lower):
        return "s"
    if re.search(r'\b(oct(?:ober)?|nov(?:ember)?)\s*/\s*(nov(?:ember)?|oct(?:ober)?)\b', content_lower):
        return "w"

    if re.search(r'\b(february|march|feb|mar)\b.{0,24}\b20\d{2}\b|\b20\d{2}\b.{0,24}\b(february|march|feb|mar)\b', content_lower):
        return "m"
    if re.search(r'\b(may|june|july|jun|jul|summer)\b.{0,24}\b20\d{2}\b|\b20\d{2}\b.{0,24}\b(may|june|july|jun|jul|summer)\b', content_lower):
        return "s"
    if re.search(r'\b(october|november|oct|nov|winter)\b.{0,24}\b20\d{2}\b|\b20\d{2}\b.{0,24}\b(october|november|oct|nov|winter)\b', content_lower):
        return "w"
    return None


# ===========================================================================
# PUBLIC API
# ===========================================================================

def generate_paper_keys(
    filename: str,
    board_hint: Optional[str] = None,
    content_session: Optional[str] = None,
    ib_ref_code_base: Optional[str] = None,
    ib_meta: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generates paper_reference_key and unified_paper_key based on filename, board,
    and optionally content-derived session and IB metadata.

    Args:
        filename (str): The original filename of the paper.
        board_hint (Optional[str]): A hint for the board (e.g., 'IGCSE', 'IB').
        content_session (Optional[str]): Session ('m', 's', 'w') derived from document content.
        ib_ref_code_base (Optional[str]): IB specific reference code base.
        ib_meta (Optional[Dict[str, str]]): IB specific metadata (subject, level, paper_number, timezone, year, document_type).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'board': Normalized board (IGCSE, IB, or UNKNOWN).
            - 'subject_code': Extracted subject code (e.g., '0580').
            - 'year_suffix': Two-digit year suffix (e.g., '18').
            - 'session': Normalized session ('m', 's', 'w', or 'UNKNOWN').
            - 'paper_number': Paper number (e.g., '4').
            - 'variant': Variant number (e.g., '2').
            - 'document_type_code': Document type code (e.g., 'ms', 'qp').
            - 'paper_reference_key': The full, unique key for the paper.
            - 'unified_paper_key': The key for pairing (strips document type).
            - 'needs_review': True if key generation encountered issues.
    """
    board = "UNKNOWN"
    if board_hint:
        board_upper = board_hint.upper()
        if "IGCSE" in board_upper or "CAMBRIDGE" in board_upper:
            board = "IGCSE"
        elif "IB" in board_upper or "INTERNATIONAL BACCALAUREATE" in board_upper:
            board = "IB"
    
    result: Dict[str, Any] = {
        "board": board,
        "subject_code": "UNKNOWN",
        "year_suffix": "UNKNOWN",
        "session": "UNKNOWN",
        "paper_number": "UNKNOWN",
        "variant": "UNKNOWN",
        "document_type_code": "UNKNOWN",
        "paper_reference_key": "",
        "unified_paper_key": "",
        "needs_review": False,
    }

    clean_filename = _sanitize_filename_for_parsing(filename)

    # Prioritize content-derived session if available
    effective_session = _extract_session_from_content(content_session) if content_session else None

    if board == "IGCSE":
        # IGCSE key generation (filename-based)
        match = re.search(
            r"(\d{4})_([smw])(\d{2})_((?:ms|qp|er|gt))_?(\d)?(\d)?",
            clean_filename, re.IGNORECASE,
        )
        if match:
            subject_code_f, season_f, year_suffix_f, doc_type_f, paper_number_f, variant_f = match.groups()

            result["subject_code"]       = subject_code_f
            result["year_suffix"]        = year_suffix_f
            result["document_type_code"] = (doc_type_f or "").lower()

            # Session priority: content > filename
            result["session"] = (effective_session or season_f).lower()

            if paper_number_f and variant_f:
                result["paper_number"] = paper_number_f
                result["variant"]      = variant_f
                # Full key: igcse_0580_s18_qp_42
                result["paper_reference_key"] = f"igcse_{subject_code_f}_{result['session']}{year_suffix_f}_{doc_type_f.lower()}_{paper_number_f}{variant_f}"
                result["unified_paper_key"]   = f"igcse_{subject_code_f}_{result['session']}{year_suffix_f}_{paper_number_f}{variant_f}"
            else:
                # Partial key: igcse_0580_s18
                result["paper_reference_key"] = f"igcse_{subject_code_f}_{result['session']}{year_suffix_f}_{doc_type_f.lower()}"
                result["unified_paper_key"]   = f"igcse_{subject_code_f}_{result['session']}{year_suffix_f}"
        else:
            # Fallback for IGCSE if initial regex fails (e.g., just subject_code and year in filename)
            match2 = re.search(r"(\d{4})_([smw])?(\d{2})", clean_filename, re.IGNORECASE)
            if match2:
                subject_code_f, season_f, year_suffix_f = match2.groups()
                result["subject_code"] = subject_code_f
                result["year_suffix"] = year_suffix_f
                result["session"] = (effective_session or season_f or "UNKNOWN").lower()
                result["paper_reference_key"] = f"igcse_{subject_code_f}_{result['session']}{year_suffix_f}"
                result["unified_paper_key"] = f"igcse_{subject_code_f}_{result['session']}{year_suffix_f}"
            else:
                logger.warning(f"[KeyBuilder] IGCSE filename regex failed for: {filename}")
                result["needs_review"] = True

    elif board == "IB":
        # IB key generation (metadata and ref_code based)
        if ib_ref_code_base:
            result["subject_code"] = ib_meta.get("subject_code", "UNKNOWN") if ib_meta else "UNKNOWN"
            result["year_suffix"] = str(ib_meta.get("year", "UNKNOWN"))[-2:] if ib_meta and ib_meta.get("year") else "UNKNOWN"
            result["session"] = (effective_session or ib_meta.get("session", "UNKNOWN") if ib_meta else "UNKNOWN").lower()
            
            # IB ref codes already contain some session info if available, but ensure 'm' priority
            if effective_session == 'm' and result['session'] != 'm':
                result['session'] = 'm'
                logger.info(f"[KeyBuilder] IB session forced to 'm' due to content priority for {filename}")

            # Format: ib_refcodebase_sessionyear_doctype
            # Example: ib_mathaa_hl_m24_qp_tz1
            paper_num_part = f"_{ib_meta.get('paper_number', '1')}" if ib_meta and ib_meta.get('paper_number') else ""
            timezone_part = f"_tz{ib_meta.get('timezone', '1')}" if ib_meta and ib_meta.get('timezone') else ""
            doc_type_code = "qp" if ib_meta and "question" in str(ib_meta.get("document_type", "")).lower() else "ms"
            
            result["paper_reference_key"] = f"ib_{ib_ref_code_base}_{result['session']}{result['year_suffix'][-2:]}_{doc_type_code}{paper_num_part}{timezone_part}"
            result["unified_paper_key"] = f"ib_{ib_ref_code_base}_{result['session']}{result['year_suffix'][-2:]}{paper_num_part}{timezone_part}"
        else:
            # Fallback for IB without ref_code
            logger.warning(f"[KeyBuilder] IB key generation failed - no ref_code_base for: {filename}")
            result["needs_review"] = True
    else:
        # Unknown board fallback
        logger.warning(f"[KeyBuilder] Unknown board '{board}' for filename: {filename}")
        result["needs_review"] = True

    return result

def generate_igcse_key(filename: str, content_detected_session: str = None) -> str:
    """
    SIMPLIFIED IGCSE key generator - prioritizes content-detected session over filename.
    
    Args:
        filename (str): The original filename
        content_detected_session (str): Session ('m', 's', 'w') detected from PDF content
        
    Returns:
        str: Generated IGCSE key (e.g., 'igcse_0580_m18_qp_42')
    """
    result = generate_paper_keys(
        filename=filename,
        board_hint="IGCSE", 
        content_session=content_detected_session
    )
    return result.get("paper_reference_key", "")

def generate_ib_key(
    subject: str,
    level: str,
    paper_number: str,
    session: str,
    year: str,
    document_type: str,
    timezone: str = None,
    content_detected_session: str = None
) -> str:
    """
    IB key generator - prioritizes content-detected session over other sources.
    
    Args:
        subject (str): Subject name
        level (str): Level (SL/HL)
        paper_number (str): Paper number
        session (str): Session string (may be verbose like "May/June")
        year (str): 4-digit year
        document_type (str): "Question Paper" or "Marking Scheme"
        timezone (str): Timezone identifier
        content_detected_session (str): Session ('m', 's', 'w') detected from PDF content
        
    Returns:
        str: Generated IB key
    """
    # Map subject to ref code base (simplified mapping)
    subject_map = {
        "mathematics": "mathaa",
        "math": "mathaa",
        "mathematics analysis": "mathaa",
        "mathematics approaches": "mathaa"
    }
    
    subject_code = subject_map.get(subject.lower(), "mathaa")
    level_code = level.lower() if level and level.lower() in ["sl", "hl"] else "hl"
    
    # Prioritize content-detected session
    effective_session = content_detected_session or _extract_session_from_content(session) or "s"
    
    year_suffix = str(year)[-2:] if year else "24"
    doc_type_code = "qp" if "question" in document_type.lower() else "ms"
    
    # Build key parts
    parts = [
        "ib",
        f"{subject_code}_{level_code}",
        f"{effective_session}{year_suffix}",
        doc_type_code
    ]
    
    if paper_number:
        parts.append(f"p{paper_number}")
    if timezone:
        parts.append(f"tz{timezone}")
    
    return "_".join(parts)

def generate_unified_paper_key(paper_reference_key: str) -> str:
    """
    Generate unified key for QP/MS pairing by removing document type markers.
    
    Args:
        paper_reference_key (str): Full paper reference key
        
    Returns:
        str: Unified key without document type (e.g., 'igcse_0580_m18_42')
    """
    if not paper_reference_key:
        return ""
    
    # Remove document type markers (qp, ms, er, gt) to create pairing key
    unified = re.sub(r'_(qp|ms|er|gt)(?=_|$)', '', paper_reference_key, flags=re.IGNORECASE)
    return unified.strip()

# ===========================================================================
# PRIVATE HELPERS (used by both IGCSE and IB key generation)
# ===========================================================================

# Make this accessible to gemini_pdf_service.py for session extraction
generate_ib_key._extract_session_from_content = _extract_session_from_content
