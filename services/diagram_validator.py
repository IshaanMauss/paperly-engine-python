
import asyncio
import base64
from typing import Any, Dict, List, Optional
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

from services.gemini_slicer import extract_pages_batch
from services.pdf_processor import pdf_base64_to_jpeg_pages_async, crop_and_compress_diagram_async
from services.pipeline_errors import PipelineServiceError
from services.gemini_runtime import run_gemini_async

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 4.0
_TRANSIENT_ERROR_CODES = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "Too Many Requests")
_GEMINI_SEMAPHORE = asyncio.Semaphore(3)

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise PipelineServiceError(
            stage="diagram_validator",
            message="GEMINI_API_KEY is not configured.",
            details={"provider": "gemini"},
        )
    return genai.Client(api_key=api_key)


async def _validate_diagram(diagram_jpeg_b64: str) -> float:
    """
    Validate a single diagram using Gemini 2.5 Flash and return a confidence score.
    """
    if not diagram_jpeg_b64:
        return 0.0

    system_prompt = """
You are a PIXEL-PRECISION DIAGRAM VALIDATOR.
Your goal is to determine if an image contains a genuine, standalone mathematical diagram and provide a confidence score.

Analyze the image based on the following criteria:

1.  **Geometric Content (Weight: 60%)**:
    *   Does the image contain closed shapes (polygons, circles, triangles)?
    *   Does the image contain plotted curves or data points?
    *   Images with only text, tables, or blank grid lines have 0 geometric content.

2.  **Completeness (Weight: 40%)**:
    *   Does the image appear to contain a full and complete diagram?
    *   Are there any signs of cropping that cut off essential parts of the diagram (e.g., axes, labels, vertices)?

Based on your analysis, provide a confidence score between 0.0 and 1.0, where:
*   **1.0**: High confidence that the image is a genuine, complete mathematical diagram.
*   **0.0**: High confidence that the image is not a valid diagram (e.g., it's just text, a partial diagram, or a table).

Return only the confidence score as a float.
"""

    try:
        image_bytes = base64.b64decode(diagram_jpeg_b64)
    except Exception:
        return 0.0

    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            async with _GEMINI_SEMAPHORE:
                client = _get_client()
                # Update inside _validate_diagram function
                response = await run_gemini_async(
                    lambda: asyncio.to_thread(
                        client.models.generate_content,
                        model=_MODEL_NAME,
                        contents=[
                            system_prompt,
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        ],
                        config={
                            "thinking_config": {"thinking_budget": 1024} # Gemini ko visual sochne ke liye budget do
                        }
                    )
                )

            raw_text = (getattr(response, "text", "") or "").strip()
            return float(raw_text)

        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            is_transient = any(code in err_str for code in _TRANSIENT_ERROR_CODES)
            if is_transient and attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY_S * (2 ** attempt)
                await asyncio.sleep(wait)
            else:
                break
    return 0.0

async def validate_diagrams_from_pdf(
    pdf_base64: str,
    document_type: str = "Question Paper",
    board: str = "IGCSE",
    paper_reference_key: str = "",
    fallback_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts and validates mathematical diagrams from a PDF.
    """
    jpeg_pages = await pdf_base64_to_jpeg_pages_async(pdf_base64)
    extracted_data = await extract_pages_batch(
        page_jpeg_b64_list=jpeg_pages,
        document_type=document_type,
        board=board,
        paper_reference_key=paper_reference_key,
        fallback_metadata=fallback_metadata,
    )

    validated_diagrams = []
    for item in extracted_data:
        for region in item.get("diagram_regions", []):
            page_num = region.get("page_num")
            y_start_pct = region.get("y_start_pct")
            y_end_pct = region.get("y_end_pct")

            if page_num is None or y_start_pct is None or y_end_pct is None:
                continue
            
            # Initial crop
            diagram_jpeg_b64 = await crop_and_compress_diagram_async(
                pdf_base64=pdf_base64,
                page_num=page_num,
                y_start_pct=y_start_pct,
                y_end_pct=y_end_pct,
            )

            if not diagram_jpeg_b64:
                continue

            # FIX: Indentation fixed here
            confidence_score = await _validate_diagram(diagram_jpeg_b64)
            
            # Verification Layer: Adjust coordinates if necessary
            if confidence_score < 0.9:
                # Expand the crop slightly (5% margin) and re-validate
                y_start_pct_expanded = max(0, y_start_pct - 5)
                y_end_pct_expanded = min(100, y_end_pct + 5)
                
                diagram_jpeg_b64_expanded = await crop_and_compress_diagram_async(
                    pdf_base64=pdf_base64,
                    page_num=page_num,
                    y_start_pct=y_start_pct_expanded,
                    y_end_pct=y_end_pct_expanded,
                )
                
                if diagram_jpeg_b64_expanded:
                    confidence_score_expanded = await _validate_diagram(diagram_jpeg_b64_expanded)
                    if confidence_score_expanded >= 0.9:
                        y_start_pct = y_start_pct_expanded
                        y_end_pct = y_end_pct_expanded
                        confidence_score = confidence_score_expanded

            if confidence_score >= 0.9:
                validated_diagrams.append(
                    {
                        "question_number": region.get("question_number"),
                        "y_start_pct": y_start_pct,
                        "y_end_pct": y_end_pct,
                    }
                )

    return validated_diagrams
