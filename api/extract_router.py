# File: api/extract_router.py
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas.ingestion_schema import SlicedQuestionsResponse
from services.gemini_pdf_service import extract_pdf_native_gemini
from services.groq_slicer import slice_and_format_questions
from services.pipeline_errors import PipelineServiceError, build_error_detail
from services.pix2text_ocr import extract_latex_from_image

router = APIRouter(prefix="/extract", tags=["extract"])
BATCH_SIZE = 10

# Schema match karega Node.js ke JSON.stringify({ image: base64 }) se
class ExtractRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Base64-encoded image payload")
    mime_type: str = Field(default="image/png", description="Incoming payload mime type")
    document_type: str = Field(default="Question Paper", description="Question Paper or Marking Scheme")


async def _process_single_page(page_image: str, page_number: int, document_type: str):
    raw_latex = await extract_latex_from_image(page_image)
    print(f"\n====== 🟢 DEBUG: RAW LATEX FROM GEMINI (PAGE {page_number}) ======")
    print(raw_latex)
    print("===============================================================\n")

    page_questions = await slice_and_format_questions(raw_latex, document_type)
    print(f"\n====== 🔵 DEBUG: SLICED ARRAY FROM GROQ (PAGE {page_number}) ======")
    print(page_questions)
    print("================================================================\n")
    return page_questions


@router.post("", response_model=SlicedQuestionsResponse)
async def process_image(request: ExtractRequest) -> SlicedQuestionsResponse:
    try:
        print("📥 [API] Processing pipeline starting...")
        mime_type = (request.mime_type or "").lower()

        if mime_type == "application/pdf":
            # Native Gemini file ingestion can take longer for larger PDFs.
            questions_array = await asyncio.wait_for(
                extract_pdf_native_gemini(request.image, request.document_type),
                timeout=540,
            )
            return SlicedQuestionsResponse(questions_array=questions_array)

        page_images = [request.image]
        questions_array = []
        for batch_start in range(0, len(page_images), BATCH_SIZE):
            batch = page_images[batch_start : batch_start + BATCH_SIZE]
            tasks = [
                _process_single_page(page_image, batch_start + index + 1, request.document_type)
                for index, page_image in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            for page_questions in batch_results:
                questions_array.extend(page_questions)

        return SlicedQuestionsResponse(questions_array=questions_array)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "type": "timeout_error",
                    "stage": "pdf_native_gemini",
                    "message": "PDF processing timed out before Gemini returned a response.",
                }
            },
        ) from exc
    except PipelineServiceError as exc:
        print(f"❌ [Pipeline Error:{exc.stage}] {exc.message}")
        raise HTTPException(status_code=exc.status_code, detail=build_error_detail(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
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