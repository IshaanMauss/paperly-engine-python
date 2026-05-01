# File: api/extract_router.py
import asyncio
import time
import hashlib
from functools import lru_cache
import concurrent.futures

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from schemas.ingestion_schema import SlicedQuestionsResponse
from services.gemini_pdf_service import extract_pdf_native_gemini
from services.groq_slicer import slice_and_format_questions
from services.pipeline_errors import PipelineServiceError, build_error_detail
from services.pix2text_ocr import extract_latex_from_image

router = APIRouter(prefix="/extract", tags=["extract"])
BATCH_SIZE = 10

# Cache for storing extraction results
# Using a simple TTL cache with a 4-hour expiration
EXTRACTION_CACHE = {}
CACHE_TTL = 14400  # 4 hours in seconds

# Queue system for managing concurrent requests
MAX_CONCURRENT_REQUESTS = 5
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

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


# Cache results of latex extraction for 4 hours
@lru_cache(maxsize=100)
def _hash_image(image_data: str) -> str:
    """Create a hash of the image data for caching"""
    return hashlib.md5(image_data.encode()).hexdigest()

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
        document_hash = _hash_image(request.image[:2000])  # Use first 2000 chars for PDF hash
        cache_key = f"{document_hash}_{request.document_type}_{request.board}_{mime_type}"
        
        current_time = time.time()
        if cache_key in EXTRACTION_CACHE and current_time - EXTRACTION_CACHE[cache_key]["timestamp"] < CACHE_TTL:
            print(f"🔄 [Cache Hit] Using cached result for document")
            JOBS_STATUS[job_id]["status"] = "completed"
            return EXTRACTION_CACHE[cache_key]["result"]

        if mime_type == "application/pdf":
            # Process within a semaphore to limit concurrent API calls
            async with REQUEST_SEMAPHORE:
                # Use a shorter timeout to improve user experience
                # The extract_pdf_native_gemini is already an async function that uses asyncio.to_thread internally
                # so we don't need to wrap it again with asyncio.to_thread
                questions_array = await asyncio.wait_for(
                    extract_pdf_native_gemini(
                        pdf_base64=request.image,
                        document_type=request.document_type,
                        filename=file_name,
                        board=request.board,
                        page1_base64=request.page1_image
                    ),
                    # Reduced timeout for better user experience
                    timeout=300,
                )
                # Cache the result
                EXTRACTION_CACHE[cache_key] = {"result": questions_array, "timestamp": current_time}
                
                # extract_pdf_native_gemini returns a SlicedQuestionsResponse object directly
                import json
                print(f"\n====== 🟢 DEBUG: OUTGOING PAYLOAD (PDF) ======")
                print(questions_array.model_dump_json(indent=2))
                print("===============================================================\n")
                
                JOBS_STATUS[job_id]["status"] = "completed"
                return questions_array

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
    return {"message": "Cache cleared", "status": "success"}
