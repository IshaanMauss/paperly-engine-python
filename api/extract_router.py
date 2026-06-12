# File: api/extract_router.py
import asyncio
import time
import hashlib
import json
import os
import re
import concurrent.futures
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from schemas.ingestion_schema import SlicedQuestionsResponse
from services.extraction_cost import reset_cost_ledger, start_cost_ledger, summarize_cost_ledger
from services.gemini_pdf_service import extract_pdf_native_gemini, rescue_missing_qp_questions
from services.groq_slicer import slice_and_format_questions
from services.pipeline_errors import PipelineServiceError, build_error_detail
from services.pix2text_ocr import extract_latex_from_image

router = APIRouter(prefix="/extract", tags=["extract"])
BATCH_SIZE = 10

# Cache for storing extraction results
# Using a simple TTL cache with a 4-hour expiration
EXTRACTION_CACHE = {}
CACHE_TTL = 14400  # 4 hours in seconds
PERSISTENT_CACHE_DIR = Path(
    os.getenv("EXTRACTION_CACHE_DIR", ".extraction_cache")
)
PERSISTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = os.getenv("EXTRACTION_CACHE_VERSION", "v1")

# Queue system for managing concurrent requests.
# Keep general request concurrency modest, but gate full-PDF extraction more
# tightly. A QP can render 20 pages and schedule many Gemini tasks; letting five
# such documents start at once can spike memory and provider-side 503s even
# though Gemini calls are rate-limited deeper in the stack.
MAX_CONCURRENT_REQUESTS = int(os.getenv("PAPERLY_MAX_CONCURRENT_REQUESTS", "5"))
MAX_CONCURRENT_PDF_EXTRACTIONS = int(os.getenv("PAPERLY_MAX_CONCURRENT_PDF_EXTRACTIONS", "2"))
REQUEST_SEMAPHORE = asyncio.Semaphore(max(1, MAX_CONCURRENT_REQUESTS))
PDF_EXTRACTION_SEMAPHORE = asyncio.Semaphore(max(1, MAX_CONCURRENT_PDF_EXTRACTIONS))

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
    use_cache: bool = Field(default=False, description="Set to true to use cached results")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional trusted metadata from Node")


class RescueMissingRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Base64-encoded source PDF")
    mime_type: str = Field(default="application/pdf", description="Must be application/pdf")
    document_type: str = Field(default="Question Paper", description="Targeted rescue is for QP rows")
    file_name: Optional[str] = Field(default="", description="Original filename")
    board: str = Field(default="IGCSE", description="Education board")
    missing_ids: List[str] = Field(default_factory=list, description="Canonical IDs missing in this upload")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Trusted metadata and expected IDs")


def _hash_image(image_data: str) -> str:
    """Create a hash of the image data for caching"""
    return hashlib.sha256(str(image_data or "").encode()).hexdigest()


def _cache_path(cache_key: str) -> Path:
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", cache_key)
    return PERSISTENT_CACHE_DIR / f"{safe_key}.json"


def _response_to_cache_payload(result: SlicedQuestionsResponse) -> dict:
    return {
        "timestamp": time.time(),
        "result": result.model_dump(mode="json"),
    }


def _read_persistent_cache(cache_key: str) -> Optional[SlicedQuestionsResponse]:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = float(payload.get("timestamp") or 0)
        if time.time() - timestamp >= CACHE_TTL:
            try:
                path.unlink()
            except Exception:
                pass
            return None
        return SlicedQuestionsResponse.model_validate(payload.get("result") or {})
    except Exception as exc:
        print(f"⚠️ [Persistent Cache] Ignoring corrupt cache file {path.name}: {exc}")
        return None


def _write_persistent_cache(cache_key: str, result: SlicedQuestionsResponse) -> None:
    try:
        path = _cache_path(cache_key)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(_response_to_cache_payload(result), ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception as exc:
        print(f"⚠️ [Persistent Cache] Failed to write cache entry: {exc}")


def _response_log_summary(result: SlicedQuestionsResponse) -> dict:
    questions = list(result.questions_array or [])
    diagram_count = 0
    diagram_bytes_estimate = 0
    for question in questions:
        urls = getattr(question, "diagram_urls", []) or []
        if not isinstance(urls, list):
            continue
        for url in urls:
            if isinstance(url, str) and url:
                diagram_count += 1
                diagram_bytes_estimate += len(url)

    metadata = result.metadata.model_dump(mode="json") if result.metadata else {}
    return {
        "question_count": len(questions),
        "diagram_count": diagram_count,
        "diagram_payload_chars": diagram_bytes_estimate,
        "validation_status": metadata.get("validation_status"),
        "paper_reference_key": metadata.get("paper_reference_key"),
        "unified_paper_key": metadata.get("unified_paper_key"),
    }


def _compact_cost_summary() -> dict:
    summary = summarize_cost_ledger()
    return {
        "gemini_calls": summary.get("gemini_calls", 0),
        "gemini_failures": summary.get("gemini_failures", 0),
        "estimated_inr": summary.get("estimated_inr", 0),
        "by_model": summary.get("by_model", {}),
        "sample_failures": summary.get("failures", []),
    }

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
        # document_hash = _hash_image(request.image[:2000])  # Old hash logic
        document_hash = _hash_image(request.image)  # Use the new hashing logic
        extra_cache_fingerprint = ""
        if request.extra_metadata:
            extra_cache_fingerprint = hashlib.sha256(
                json.dumps(request.extra_metadata, sort_keys=True, ensure_ascii=True).encode("utf-8")
            ).hexdigest()[:16]
        cache_key = (
            f"{CACHE_VERSION}_{document_hash}_{request.document_type}_{request.board}_"
            f"{mime_type}_{file_name}_{extra_cache_fingerprint}"
        )
        print(
            f"[API] Cache policy: use_cache={request.use_cache} "
            f"cache_key_prefix={cache_key[:32]} file={file_name!r}"
        )
        
        current_time = time.time()
        if request.use_cache and cache_key in EXTRACTION_CACHE and current_time - EXTRACTION_CACHE[cache_key]["timestamp"] < CACHE_TTL:
            print(f"🔄 [Cache Hit] Using cached result for document")
            JOBS_STATUS[job_id]["status"] = "completed"
            return EXTRACTION_CACHE[cache_key]["result"]
        if request.use_cache:
            persistent_hit = _read_persistent_cache(cache_key)
            if persistent_hit is not None:
                print(f"Persistent cache hit for {file_name or request.document_type}")
                EXTRACTION_CACHE[cache_key] = {"result": persistent_hit, "timestamp": current_time}
                JOBS_STATUS[job_id]["status"] = "completed"
                return persistent_hit

        if mime_type == "application/pdf":
            # Process within a semaphore to limit concurrent API calls
            async with PDF_EXTRACTION_SEMAPHORE:
                cost_token = start_cost_ledger()
                try:
                    # Use a shorter timeout to improve user experience
                    # The extract_pdf_native_gemini is already an async function that uses asyncio.to_thread internally
                    # so we don't need to wrap it again with asyncio.to_thread
                    questions_array = await asyncio.wait_for(
                        extract_pdf_native_gemini(
                            pdf_base64=request.image,
                            document_type=request.document_type,
                            filename=file_name,
                            board=request.board,
                            page1_base64=request.page1_image,
                            extra_metadata=request.extra_metadata or None,
                        ),
                        timeout=float(os.getenv("PAPERLY_PDF_EXTRACTION_TIMEOUT_SECONDS", "600")),
                    )
                    # Cache the result
                    EXTRACTION_CACHE[cache_key] = {"result": questions_array, "timestamp": current_time}
                    background_tasks.add_task(_write_persistent_cache, cache_key, questions_array)

                    response_summary = _response_log_summary(questions_array)
                    response_summary["gemini_cost"] = _compact_cost_summary()
                    print(f"[API] PDF extraction response summary: {response_summary}")

                    JOBS_STATUS[job_id]["status"] = "completed"
                    return questions_array
                except Exception:
                    print(f"[API] PDF extraction cost summary before failure: {_compact_cost_summary()}")
                    raise
                finally:
                    reset_cost_ledger(cost_token)

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


@router.post("/rescue-missing")
async def rescue_missing_questions(request: RescueMissingRequest) -> Dict[str, Any]:
    mime_type = (request.mime_type or "").lower()
    if mime_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Targeted rescue requires the original PDF payload.")
    if (request.document_type or "").strip().lower() == "marking scheme":
        raise HTTPException(status_code=400, detail="Targeted rescue is only for Question Paper missing rows.")

    async with PDF_EXTRACTION_SEMAPHORE:
        cost_token = start_cost_ledger()
        try:
            result = await asyncio.wait_for(
                rescue_missing_qp_questions(
                    pdf_base64=request.image,
                    missing_ids=request.missing_ids or [],
                    filename=request.file_name or "",
                    board=request.board or "IGCSE",
                    extra_metadata=request.extra_metadata or None,
                ),
                timeout=float(os.getenv("PAPERLY_RESCUE_TIMEOUT_SECONDS", "180")),
            )
            result["gemini_cost"] = _compact_cost_summary()
            print(
                "[API] Rescue missing response summary: "
                f"{result.get('rescue_report')} cost={result.get('gemini_cost')}"
            )
            return result
        except PipelineServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=build_error_detail(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="Targeted rescue timed out. Try full redo only if the missing rows are important.",
            ) from exc
        except Exception as exc:
            print(f"[API] Rescue missing cost summary before failure: {_compact_cost_summary()}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            reset_cost_ledger(cost_token)


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
    for path in PERSISTENT_CACHE_DIR.glob("*.json"):
        try:
            path.unlink()
        except Exception:
            pass
    return {"message": "Cache cleared", "status": "success"}
