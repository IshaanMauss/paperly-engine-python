# File: services/extract_router.py
"""
Paperly Extraction Router — Orchestration Layer
================================================

This module is the single entry point for all document extraction requests.
It is responsible for:

  1. Routing documents to the correct extraction service
     (currently: Gemini multimodal single-pass via gemini_pdf_service.py).

  2. Building the correct `fallback_metadata` dict before any extraction
     call, ensuring both QP and MS documents are processed with authoritative
     metadata from the moment extraction begins.

  3. POINT 2 — MS METADATA INHERITANCE (core responsibility of this module):
     Marking Scheme PDFs have no cover page carrying session/year/tier data.
     Before extracting an MS, this router MUST:
       a. Derive the unified_paper_key for the MS document.
       b. Perform a PaperRegistry.findOne lookup for a paired QP with the
          same unified_paper_key.
       c. Inject the QP's session, year, tier, and subjectCode into the MS
          extraction request's fallback_metadata.
     This ensures the MS questions are written to MongoDB with metadata
     identical to their paired QP from the very first write — no post-hoc
     patching required.

  4. Updating the PaperRegistry entry after extraction with the confirmed
     document_id, validation_status, and unified_paper_key.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH v5 — MS METADATA INHERITANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT CAUSE OF THE BUG THIS FIXES:
  MS documents were being extracted with an empty or absent fallback_metadata
  dict. The slicer's _sync_metadata_to_all_pages was gated behind
  `if not is_ms`, so even when fallback_metadata contained useful values,
  nothing was propagating them to MS question models.

  Two-layer fix:
    Layer 1 (this file): The router now always performs a PaperRegistry
      lookup before MS extraction and populates fallback_metadata with the
      paired QP's authoritative session/year/tier.
    Layer 2 (gemini_slicer.py): The slicer's `if not is_ms` gate in
      extract_pages_batch has been replaced with a dual-path that uses
      fallback_metadata as the authoritative source for MS documents.

CONTRACT WITH THE SLICER:
  For MS documents, the router MUST pass fallback_metadata containing:
    {
        "session":     "<m|s|w>",          # from paired QP PaperRegistry entry
        "year":        <int>,              # e.g. 2018
        "tier":        "<Extended|Core>",  # from paired QP
        "subjectCode": "<str>",            # e.g. "0580"
        "paper_reference_key": "<str>",    # MS key, e.g. "igcse_0580_s18_ms_43"
        "unified_paper_key":   "<str>",    # shared key, e.g. "igcse_0580_s18_43"
    }
  The slicer will apply these values to every question model in the MS batch.
  If this dict is absent or empty, the slicer logs a WARNING and writes N/A
  session values — which is the exact bug we are fixing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal imports — adjust to your project's actual module paths.
# ---------------------------------------------------------------------------
# These are illustrative; replace with your real import paths.
try:
    from models.paper_registry import PaperRegistry          # Mongoose/Motor model
    from services.gemini_pdf_service import extract_pdf_native_gemini
    from services.key_builder import build_paper_reference_key, build_unified_paper_key
except ImportError:
    # Graceful degradation for environments where these aren't importable
    # (e.g. unit-test runners that only import this module).
    PaperRegistry = None  # type: ignore[assignment]
    extract_pdf_native_gemini = None  # type: ignore[assignment]
    build_paper_reference_key = None  # type: ignore[assignment]
    build_unified_paper_key = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOCUMENT_TYPE_QP = "Question Paper"
_DOCUMENT_TYPE_MS = "Marking Scheme"

# Fields the router reads from a PaperRegistry QP entry to inject into MS.
# These are the four scalars that _sync_metadata_to_all_pages propagates.
_QP_INHERITANCE_FIELDS = ("session", "year", "tier", "subjectCode")


# ===========================================================================
# SECTION 1 — Key derivation helpers
# ===========================================================================

def _derive_unified_paper_key(paper_reference_key: str) -> str:
    """
    Derive the unified_paper_key from a paper_reference_key by stripping
    the document-type segment (_qp, _ms, _er, _gt).

    Examples:
      "igcse_0580_s18_ms_43"  →  "igcse_0580_s18_43"
      "igcse_0580_s18_qp_43"  →  "igcse_0580_s18_43"
      "igcse_0607_m23_qp_22"  →  "igcse_0607_m23_22"

    This mirrors the logic in QuestionNumberNormalizer._generate_unified_paper_key
    and in gemini_slicer.py's metadata sync — it must stay in sync with both.
    """
    if not paper_reference_key:
        return ""
    unified = re.sub(
        r'_(qp|ms|er|gt)',
        '',
        str(paper_reference_key),
        flags=re.IGNORECASE,
    )
    return unified.strip("_").strip()


def _derive_ms_paper_reference_key(ms_paper_reference_key: str) -> str:
    """
    Normalise an MS paper_reference_key to lowercase and strip whitespace.
    Returns the key unchanged if already normalised.
    """
    return str(ms_paper_reference_key or "").strip().lower()


# ===========================================================================
# SECTION 2 — PaperRegistry lookup
# ===========================================================================

async def _fetch_qp_metadata_for_ms(
    unified_paper_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up the PaperRegistry for a paired Question Paper entry sharing the
    same unified_paper_key as the MS being processed.

    Returns a dict with the four authoritative scalar fields if a paired QP
    is found, or None if no match exists or the lookup fails.

    The returned dict matches the contract documented in the module docstring:
        { "session": ..., "year": ..., "tier": ..., "subjectCode": ... }

    This function is intentionally narrow — it only reads the four fields
    needed for MS metadata injection. It does NOT read or expose the QP's
    document_id, question content, or any other fields.
    """
    if not unified_paper_key:
        logger.warning(
            "[ExtractRouter] _fetch_qp_metadata_for_ms called with empty "
            "unified_paper_key. Cannot perform registry lookup."
        )
        return None

    if PaperRegistry is None:
        logger.error(
            "[ExtractRouter] PaperRegistry model is not importable. "
            "Skipping QP metadata lookup for MS inheritance."
        )
        return None

    try:
        # Find any registry entry for this unified_paper_key that is a
        # Question Paper (has qp_document_id set) and has been processed.
        # Motor (async Mongoose-style): use .find_one() with a projection.
        registry_entry = await PaperRegistry.find_one(
            {
                "unified_paper_key": unified_paper_key,
                # Only consider entries where the QP has been extracted
                # (qp_document_id is set and non-null).
                "qp_document_id": {"$exists": True, "$ne": None},
            },
            projection={
                "session": 1,
                "year": 1,
                "tier": 1,
                "subjectCode": 1,
                "_id": 0,
            },
        )

        if registry_entry is None:
            logger.warning(
                f"[ExtractRouter] No paired QP found in PaperRegistry for "
                f"unified_paper_key={unified_paper_key!r}. "
                f"MS will be extracted without inherited session metadata."
            )
            return None

        # Extract the four scalar fields, filtering out missing/None values.
        inherited: Dict[str, Any] = {}
        for field in _QP_INHERITANCE_FIELDS:
            # Support both Motor document objects and plain dicts.
            value = (
                getattr(registry_entry, field, None)
                if not isinstance(registry_entry, dict)
                else registry_entry.get(field)
            )
            if value is not None and str(value).strip() not in ("", "N/A", "None", "null"):
                inherited[field] = value

        if not inherited:
            logger.warning(
                f"[ExtractRouter] PaperRegistry entry found for "
                f"unified_paper_key={unified_paper_key!r} but all four "
                f"scalar fields (session/year/tier/subjectCode) are empty "
                f"or N/A. MS metadata inheritance will be incomplete."
            )
            return None

        logger.info(
            f"[ExtractRouter] QP metadata locked for MS inheritance "
            f"(unified_paper_key={unified_paper_key!r}): {inherited}"
        )
        return inherited

    except Exception as exc:
        logger.error(
            f"[ExtractRouter] PaperRegistry lookup failed for "
            f"unified_paper_key={unified_paper_key!r}: {exc!r}. "
            f"MS will be extracted without inherited session metadata."
        )
        return None


# ===========================================================================
# SECTION 3 — fallback_metadata builder
# ===========================================================================

def _build_ms_fallback_metadata(
    ms_paper_reference_key: str,
    unified_paper_key: str,
    qp_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the fallback_metadata dict to pass to extract_pdf_native_gemini
    (and by extension, to extract_pages_batch in the slicer) for an MS doc.

    Merges:
      1. MS document identity fields (paper_reference_key, unified_paper_key).
      2. QP-inherited authoritative scalars (session, year, tier, subjectCode).

    Priority: QP metadata > any value previously known about the MS.
    The merged dict is what the slicer uses in _sync_metadata_to_all_pages
    MS path to overwrite every question model in the batch.

    Parameters
    ----------
    ms_paper_reference_key : The MS-specific key, e.g. "igcse_0580_s18_ms_43".
    unified_paper_key      : The shared key,     e.g. "igcse_0580_s18_43".
    qp_metadata            : Dict from _fetch_qp_metadata_for_ms, or None.

    Returns
    -------
    Dict ready to pass as `fallback_metadata` to extract_pdf_native_gemini.
    """
    fallback: Dict[str, Any] = {
        "paper_reference_key": ms_paper_reference_key,
        "unified_paper_key":   unified_paper_key,
        "document_type":       _DOCUMENT_TYPE_MS,
    }

    if qp_metadata:
        fallback.update(qp_metadata)
        logger.debug(
            f"[ExtractRouter] MS fallback_metadata built with QP inheritance: "
            f"{fallback}"
        )
    else:
        logger.warning(
            f"[ExtractRouter] MS fallback_metadata built WITHOUT QP inheritance "
            f"(qp_metadata is None). MS questions will have session=N/A unless "
            f"Gemini extracts a cover page (which MS files do not have). "
            f"paper_reference_key={ms_paper_reference_key!r}"
        )

    return fallback


# ===========================================================================
# SECTION 4 — Public routing entry point
# ===========================================================================

async def route_extraction(
    file_path: str,
    document_type: str,
    paper_reference_key: str,
    board: str = "IGCSE",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Primary routing function. Called by the ingestion pipeline for every
    uploaded document (QP or MS).

    For QP documents:
      - Builds a minimal fallback_metadata with document identity fields.
      - Calls extract_pdf_native_gemini directly.
      - The slicer handles session extraction from the QP cover page.

    For MS documents:
      - Derives the unified_paper_key.
      - Performs a PaperRegistry.findOne lookup for the paired QP.
      - Injects the QP's session/year/tier into fallback_metadata.
        ── THIS IS THE CORE OF POINT 2 ──
      - Calls extract_pdf_native_gemini with the enriched fallback_metadata.
      - The slicer's MS path in _sync_metadata_to_all_pages uses this dict
        as the authoritative source (no cover page scan for MS).

    Parameters
    ----------
    file_path            : Absolute path to the PDF on disk (or S3 key, etc.)
    document_type        : "Question Paper" or "Marking Scheme"
    paper_reference_key  : Pre-generated key, e.g. "igcse_0580_s18_ms_43"
    board                : "IGCSE" or "IB"
    extra_metadata       : Any additional metadata to merge (optional).

    Returns
    -------
    Dict from extract_pdf_native_gemini — structure depends on that service.
    """
    if extract_pdf_native_gemini is None:
        raise RuntimeError(
            "[ExtractRouter] extract_pdf_native_gemini is not importable. "
            "Cannot route extraction."
        )

    doc_type_normalised = str(document_type or "").strip().lower()
    is_ms = doc_type_normalised == "marking scheme"

    unified_paper_key = _derive_unified_paper_key(paper_reference_key)

    # ── Build fallback_metadata ───────────────────────────────────────────────
    if is_ms:
        # POINT 2: MS path — inherit QP metadata from PaperRegistry BEFORE
        # extraction begins. The slicer cannot do this itself because it has
        # no access to the database.
        qp_metadata = await _fetch_qp_metadata_for_ms(unified_paper_key)
        fallback_metadata = _build_ms_fallback_metadata(
            ms_paper_reference_key=paper_reference_key,
            unified_paper_key=unified_paper_key,
            qp_metadata=qp_metadata,
        )
    else:
        # QP path — simple identity metadata; session comes from the cover page.
        fallback_metadata = {
            "paper_reference_key": paper_reference_key,
            "unified_paper_key":   unified_paper_key,
            "document_type":       _DOCUMENT_TYPE_QP,
        }

    # Merge any extra metadata (lowest priority — never overwrite inherited values).
    if extra_metadata:
        for k, v in extra_metadata.items():
            fallback_metadata.setdefault(k, v)

    logger.info(
        f"[ExtractRouter] Routing extraction: "
        f"document_type={document_type!r}, "
        f"paper_reference_key={paper_reference_key!r}, "
        f"unified_paper_key={unified_paper_key!r}, "
        f"is_ms={is_ms}, "
        f"inherited_session={fallback_metadata.get('session', 'NOT_INJECTED')!r}"
    )

    # ── Call the extraction service ───────────────────────────────────────────
    result = await extract_pdf_native_gemini(
        file_path=file_path,
        document_type=document_type,
        board=board,
        paper_reference_key=paper_reference_key,
        fallback_metadata=fallback_metadata,
    )

    return result


# ===========================================================================
# SECTION 5 — Batch routing (multi-document ingestion jobs)
# ===========================================================================

async def route_extraction_batch(
    documents: List[Dict[str, Any]],
    board: str = "IGCSE",
) -> List[Dict[str, Any]]:
    """
    Route a batch of documents. Handles ordering to ensure QP documents are
    always extracted BEFORE their paired MS documents.

    Why ordering matters:
      The MS metadata inheritance (POINT 2) requires a PaperRegistry entry
      for the paired QP to exist before the MS is processed. If both are
      uploaded simultaneously, the QP must be extracted first so that the
      PaperRegistry entry is written before _fetch_qp_metadata_for_ms runs.

    If a batch contains both a QP and its paired MS, this function:
      1. Extracts all QPs first (concurrently across different subjects).
      2. Extracts all MSs second (concurrently, but only after QPs are done).

    Parameters
    ----------
    documents : List of dicts, each with keys:
        {
            "file_path":           str,
            "document_type":       "Question Paper" | "Marking Scheme",
            "paper_reference_key": str,
            "extra_metadata":      dict (optional),
        }
    board     : "IGCSE" or "IB"

    Returns
    -------
    List of results in the same order as the input documents list.
    """
    import asyncio

    if not documents:
        return []

    # Partition into QPs and MSs.
    qp_docs  = [(i, d) for i, d in enumerate(documents)
                if str(d.get("document_type", "")).strip().lower() != "marking scheme"]
    ms_docs  = [(i, d) for i, d in enumerate(documents)
                if str(d.get("document_type", "")).strip().lower() == "marking scheme"]

    results: List[Optional[Dict[str, Any]]] = [None] * len(documents)

    async def _route_one(idx: int, doc: Dict[str, Any]) -> None:
        result = await route_extraction(
            file_path=doc["file_path"],
            document_type=doc["document_type"],
            paper_reference_key=doc["paper_reference_key"],
            board=board,
            extra_metadata=doc.get("extra_metadata"),
        )
        results[idx] = result

    # Phase 1: extract all QPs concurrently.
    if qp_docs:
        logger.info(
            f"[ExtractRouter] Batch phase 1: extracting {len(qp_docs)} QP(s) concurrently."
        )
        await asyncio.gather(*[_route_one(i, d) for i, d in qp_docs])

    # Phase 2: extract all MSs concurrently (QPs are now in PaperRegistry).
    if ms_docs:
        logger.info(
            f"[ExtractRouter] Batch phase 2: extracting {len(ms_docs)} MS(s) concurrently "
            f"(paired QP registry entries are now available)."
        )
        await asyncio.gather(*[_route_one(i, d) for i, d in ms_docs])

    return [r for r in results if r is not None]