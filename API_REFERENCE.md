# 🔗 API Reference Card

## `services/gemini_slicer.py`

### Main Function (Async)
```python
async def extract_and_structure_page_async(
    base64_image: str,
    document_type: str = "Question Paper",
    page_number: int = 1,
) -> List[Dict]:
    """
    Extract structured questions from a single exam page image.
    
    Args:
        base64_image: Base64-encoded JPEG/PNG (with or without data-URI prefix)
        document_type: "Question Paper" or "Marking Scheme"
        page_number: Page number for logging (1-indexed)
    
    Returns:
        List of extraction bundles:
        [
            {
                "model": ExtractedQuestion,
                "diagram_regions": [
                    {
                        "y_start_pct": 25.0,
                        "y_end_pct": 50.0,
                        "x_start_pct": 0.0,
                        "x_end_pct": 100.0,
                        "visual_type": "diagram"
                    },
                    ...
                ]
            },
            ...
        ]
    
    Raises:
        PipelineServiceError: If extraction fails after all retries
    
    Example:
        bundles = await extract_and_structure_page_async(
            base64_image=page_jpeg,
            document_type="Question Paper",
            page_number=1,
        )
        for bundle in bundles:
            question_model = bundle["model"]
            diagram_regions = bundle["diagram_regions"]
    """
```

### Main Function (Sync)
```python
def extract_and_structure_page(
    base64_image: str,
    document_type: str = "Question Paper",
    page_number: int = 1,
) -> List[Dict]:
    """Synchronous version. Same signature as above."""
```

### Internal: JSON Sanitization
```python
def _parse_json_payload(content: str) -> dict:
    """
    Parse Gemini response JSON with aggressive backslash healing.
    
    Handles:
    - Markdown fence removal
    - Invalid \\u escape sequence fixing
    - Illegal backslash double-escaping
    - Iterative JSON parsing with auto-heal (10 attempts)
    
    Returns: dict (empty {"questions_array": []} on failure)
    """
```

---

## `services/gemini_pdf_service.py`

### Main Function (Async)
```python
async def extract_pdf_native_gemini(
    pdf_base64: str,
    document_type: str,
    filename: str,
    board: str = "IGCSE",
    page1_base64: str = None,
) -> SlicedQuestionsResponse:
    """
    Extract structured questions from a PDF using Gemini 2.5 Flash.
    
    Single-pass multimodal architecture:
    1. Convert PDF → page images (250 DPI JPEG)
    2. Loop through pages with semaphore batching (max 2 concurrent)
    3. Call gemini_slicer.extract_and_structure_page for each page
    4. Crop any detected diagrams and append to question.diagram_urls
    5. Aggregate and return SlicedQuestionsResponse
    
    Args:
        pdf_base64: Base64-encoded PDF (with or without data-URI prefix)
        document_type: "Question Paper" or "Marking Scheme"
        filename: Original filename (for paper key generation)
        board: "IGCSE" or "IB"
        page1_base64: (Optional) Base64 image of first page for IB metadata
    
    Returns:
        SlicedQuestionsResponse(
            metadata=ExtractedPaperMetadata(...),
            questions_array=[ExtractedQuestion(...), ...],
            validation_report=ValidationReport(...),
        )
    
    Raises:
        PipelineServiceError: If rendering or extraction fails catastrophically
    
    Example:
        response = await extract_pdf_native_gemini(
            pdf_base64=pdf_b64,
            document_type="Question Paper",
            filename="0580_s24_qp_1.pdf",
            board="IGCSE",
        )
        
        for question in response.questions_array:
            print(f"Q{question.question_id}: {len(question.diagram_urls)} diagrams")
        
        print(f"Validation: {response.validation_report.status}")
    """
```

---

## Data Models (from `schemas/ingestion_schema.py`)

### Input: Extraction Bundle
```python
# Returned by gemini_slicer.extract_and_structure_page_async()
bundle = {
    "model": ExtractedQuestion(
        document_type="Question Paper",
        question_latex="1 The diagram shows...",
        diagram_urls=[],
        cognitive_demand="MEDIUM",
        # ... other fields ...
    ),
    "diagram_regions": [
        {
            "y_start_pct": 25.0,
            "y_end_pct": 50.0,
            "x_start_pct": 0.0,
            "x_end_pct": 100.0,
            "visual_type": "diagram"
        }
    ]
}
```

### Output: SlicedQuestionsResponse
```python
response = SlicedQuestionsResponse(
    metadata=ExtractedPaperMetadata(
        curriculum="IGCSE",
        program=None,
        subjectCode="0580",
        tier="Extended",
        paperNumber=1,
        session="May/June",
        year=2024,
        paper_reference_key="igcse_0580_s24_qp_1",
        validation_status="ok",
        validation_warnings=[],
        ref_code_base="",
        ref_code_full="",
    ),
    questions_array=[
        ExtractedQuestion(
            document_type="Question Paper",
            curriculum="IGCSE",
            program=None,
            subjectCode="0580",
            tier="Extended",
            paperNumber=1,
            session="May/June",
            year=2024,
            paper_reference_key="igcse_0580_s24_qp_1",
            unified_paper_key="",
            canonical_question_id="1",
            parent_canonical_id=None,
            question_number_metadata=QuestionNumberMetadata(
                parent=1,
                subparts=[],
                depth=1,
                is_orphaned=False,
            ),
            validation_status="ok",
            validation_warnings=[],
            ref_code_base="",
            ref_code_full="",
            isTemplatizable=False,
            variables=[],
            question_latex="1 The diagram shows a triangle ABC...",
            official_marking_scheme_latex=None,
            diagram_urls=[
                "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
            ],
            needs_review=False,
            cognitive_demand="MEDIUM",
            difficulty_override=None,
            # MS-specific fields (null for QP):
            question_id="",
            final_answer="",
            total_marks=0,
            method_steps=[],
        ),
        # ... more questions ...
    ],
    validation_report=ValidationReport(
        status="ok",
        recommendation="proceed",
        message="Extraction complete.",
        checks={"sequence_check": False},
    ),
)
```

---

## Error Handling

### PipelineServiceError
```python
from services.pipeline_errors import PipelineServiceError

try:
    response = await extract_pdf_native_gemini(...)
except PipelineServiceError as e:
    print(f"Stage: {e.stage}")           # "pdf_rendering", "gemini_slicer", etc.
    print(f"Message: {e.message}")       # Human-readable error
    print(f"Details: {e.details}")       # Dict with provider, reason, exception_type
```

---

## Configuration

### Environment Variables
```bash
# Required
export GEMINI_API_KEY="sk-..."

# Optional (defaults shown)
# HYBRID_TASK_A_ENABLED=true  # (deprecated, not used in new arch)
```

### Rate Limiting Tuning
Edit these constants in `gemini_pdf_service.py`:
```python
# Max concurrent Gemini API calls
_PAGE_EXTRACTION_SEMAPHORE = asyncio.Semaphore(2)  # Default: 2

# Delay between page extractions (seconds)
_INTER_PAGE_DELAY = 0.5  # Default: 0.5s

# Adjust if you hit rate limits:
# - Increase delays: _INTER_PAGE_DELAY = 1.0
# - Decrease concurrency: Semaphore(1)
```

---

## Integration Example

```python
import asyncio
from services.gemini_pdf_service import extract_pdf_native_gemini

async def process_exam_paper(pdf_b64: str, filename: str):
    """Complete workflow."""
    
    # Extract with full logging
    response = await extract_pdf_native_gemini(
        pdf_base64=pdf_b64,
        document_type="Question Paper",
        filename=filename,
        board="IGCSE",
    )
    
    # Check validation
    if response.validation_report.status == "error":
        print(f"⚠️ Validation failed: {response.validation_report.message}")
        # Handle error...
    
    # Process questions
    for idx, question in enumerate(response.questions_array):
        print(f"\n📝 Question {idx+1}: {question.question_id}")
        print(f"   LaTeX: {question.question_latex[:100]}...")
        print(f"   Diagrams: {len(question.diagram_urls)}")
        print(f"   Cognitive Demand: {question.cognitive_demand}")
        
        # For marking schemes:
        if question.document_type == "Marking Scheme":
            print(f"   Total Marks: {question.total_marks}")
            for step in question.method_steps:
                print(f"   - {step.type}: {step.description}")
    
    return response

# Usage
if __name__ == "__main__":
    with open("exam_paper.pdf", "rb") as f:
        pdf_bytes = f.read()
    
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    
    response = asyncio.run(process_exam_paper(pdf_b64, "0580_s24_qp_1.pdf"))
    print(f"\n✅ Extraction complete: {len(response.questions_array)} questions")
```

---

## Logging Output Examples

```
[GeminiSlicer] Extracting page 1 (Question Paper)...
[GeminiSlicer] Page 1: attempt 1/3
[GeminiSlicer] ✅ Page 1: 5 question(s), 2 diagram region(s)

[PDF Service] Starting extraction: 0580_s24_qp_1.pdf (IGCSE, Question Paper)
[PDF Service] Rendering PDF pages to JPEG (250 DPI)…
[PDF Service] Rendered 10 page(s)
[Metadata] IGCSE key: igcse_0580_s24_qp_1
[Extraction] Processing page 1…
[Extraction] ✅ Page 1: 5 question(s)
[Cropping] Processing 50 question(s) for diagram cropping…
[Cropping] ✅ Crop verified for q=1 (page=0, y=25.0%-50.0%, height_pct=25.0%, px=800×400, aspect=2.00W:1H, size=45.3KB)
[Cropping] ✅ 50 question(s) ready with diagram URLs
[Response] ✅ SlicedQuestionsResponse ready: 50 question(s), status=ok
```

---

**Last Updated:** May 20, 2026  
**Architecture:** Single-Pass Multimodal with Gemini 2.5 Flash  
**Status:** ✅ Production Ready
