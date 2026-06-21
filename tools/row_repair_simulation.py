import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.extract_router import (
    _canonical_parts,
    _display_label_from_canonical,
    _fallback_pdf_repair_context_crop,
    _pdf_page_text_matches_target,
    _repair_candidate_pages,
    _repair_row_from_pdf_context,
)


def _pdf_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _dummy_rows(target_id: str) -> List[Dict[str, Any]]:
    parts = _canonical_parts(target_id)
    root = parts[0] if parts else target_id.split(".", 1)[0]
    parent = ".".join(parts[:-1]) if len(parts) > 1 else root
    return [
        {"document_type": "Question Paper", "canonical_question_id": root, "question_latex": f"{root} previous/root context"},
        {
            "document_type": "Question Paper",
            "canonical_question_id": target_id,
            "question_latex": f"{_display_label_from_canonical(target_id)} BROKEN ROW PLACEHOLDER",
            "diagram_urls": [],
        },
        {
            "document_type": "Question Paper",
            "canonical_question_id": f"{parent}.ii" if not target_id.endswith(".ii") else f"{parent}.iii",
            "question_latex": "near sibling placeholder",
            "diagram_urls": [],
        },
    ]


async def _run_case(pdf_path: Path, target_id: str, ai: bool) -> Dict[str, Any]:
    pdf_base64 = _pdf_b64(pdf_path)
    rows = _dummy_rows(target_id)
    pages = _repair_candidate_pages(pdf_base64, target_id, rows)
    local_pages = [
        {"page": page + 1, "text_match": _pdf_page_text_matches_target(pdf_base64, page, target_id)}
        for page in pages
    ]
    fallback_crop = None
    if pages:
        fallback_crop = await _fallback_pdf_repair_context_crop(pdf_base64, pages[0], target_id)
    result: Dict[str, Any] = {
        "pdf": str(pdf_path),
        "target_id": target_id,
        "candidate_pages": local_pages,
        "fallback_crop_found": bool(fallback_crop),
        "fallback_crop_bytes": len((fallback_crop or "").split(",", 1)[-1]) if fallback_crop else 0,
    }
    if ai:
        context = {
            "title": "Diagram/Image Missing",
            "problem": "This row mentions a diagram/graph/grid but has no image attached.",
            "solution": "Repair from the original PDF and attach only a connected diagram.",
            "expected": "Target row text includes required parent stem; image only if connected.",
        }
        repair = await _repair_row_from_pdf_context(
            row=rows[1],
            rows=rows,
            row_index=1,
            metadata={"document_type": "Question Paper"},
            pdf_base64=pdf_base64,
            mime_type="application/pdf",
            file_name=pdf_path.name,
            board="IGCSE",
            repair_context=context,
        )
        proposal = repair.get("proposal") or {}
        result.update(
            {
                "applied": repair.get("applied"),
                "reason": repair.get("reason"),
                "trace": repair.get("repair_trace"),
                "text": proposal.get("question_latex"),
                "diagram_count": len(proposal.get("diagram_urls") or []),
            }
        )
    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="PDF_PATH::canonical_id")
    parser.add_argument("--ai", action="store_true", help="Run Gemini repair, not just local crop/page simulation.")
    args = parser.parse_args()
    results = []
    for case in args.case:
        pdf_raw, target_id = case.split("::", 1)
        results.append(await _run_case(Path(pdf_raw), target_id, args.ai))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
