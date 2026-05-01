# File: schemas/ingestion_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any


# ---------------------------------------------------------------------------
# MS Training Schema — method_steps item
# ---------------------------------------------------------------------------

class MethodStep(BaseModel):
    """A single mark-point in a marking scheme answer."""
    type: str = Field("", description="Mark code: M1, A1, B1, ft, oe, dep, allow, accept, etc.")
    description: str = Field("", description="What earns this mark, in LaTeX if it contains math.")


# ---------------------------------------------------------------------------
# Single Question / MS Entry
# ---------------------------------------------------------------------------

class ExtractedQuestion(BaseModel):
    # ── Document classification ──────────────────────────────────────────────
    document_type: Optional[str] = Field(
        "Question Paper",
        description="'Question Paper' or 'Marking Scheme' — set by the pipeline, not the LLM.",
    )

    # ── Paper metadata (duplicated on every row for denormalised storage) ────
    curriculum: Optional[str] = Field("", description="Educational curriculum, e.g. 'IGCSE', 'IB'")
    program: Optional[str] = Field("", description="Program, e.g. 'MYP', 'DP', 'Secondary'")
    subjectCode: Optional[str] = Field("", description="Subject code, e.g. '0580', 'AA'")
    tier: Optional[str] = Field("", description="Academic tier, e.g. 'Core', 'Extended', 'SL', 'HL'")
    paperNumber: Optional[int] = Field(0, description="Paper number, e.g. 1, 2, 3")
    session: Optional[str] = Field("", description="Examination session, e.g. 'May/June'")
    year: Optional[int] = Field(0, description="Examination year, e.g. 2023")

    # ── Fingerprint — links QP ↔ MS in the database ──────────────────────────
    paper_reference_key: Optional[str] = Field(
        "",
        description="Slug that links a QP to its MS. Format: YEAR_SUBJECT_PAPER_VARIANT. "
                    "Generated from the filename by the pipeline.",
    )
    ref_code_base: Optional[str] = Field("", description="Base reference code without suffix (e.g., 2225-7106)")
    ref_code_full: Optional[str] = Field("", description="Full reference code with suffix (e.g., 2225-7106M)")

    # ── QP fields ────────────────────────────────────────────────────────────
    isTemplatizable: Optional[bool] = Field(
        False,
        description="True if the question contains standard numerical/algebraic values that can be templated.",
    )
    variables: Optional[List[str]] = Field(
        default_factory=list,
        description="Array of string variables extracted from templatizable questions.",
    )
    question_latex: Optional[str] = Field(
        "",
        description="Full LaTeX for the question text (QP) or the question number/label (MS).",
    )
    official_marking_scheme_latex: Optional[str] = Field(
        "",
        description="Full raw marking scheme text in LaTeX (MS only; null for QP).",
    )
    diagram_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="URLs for diagrams associated with the question.",
    )
    needs_review: Optional[bool] = Field(
        False,
        description="Flag if the question needs human review.",
    )

    # ── MS Training Schema fields ─────────────────────────────────────────────
    question_id: Optional[str] = Field(
        "",
        description="Question number/label (MS only). Mirrors question_latex. "
                    "e.g. '3(a)(i)'. Used as the ML training identifier.",
    )
    final_answer: Optional[str] = Field(
        "",
        description="Concise final answer for this question/sub-part (MS only). "
                    "Plain text or LaTeX. e.g. 'x = 3' or '$\\\\frac{1}{2}$'.",
    )
    total_marks: Optional[int] = Field(
        0,
        description="Total marks awarded for this question/sub-part (MS only). "
                    "Integer sum of all mark codes.",
    )
    method_steps: Optional[List[MethodStep]] = Field(
        default_factory=list,
        description="Ordered list of mark-point objects (MS only). "
                    "Each step has 'type' (mark code) and 'description' (what earns the mark).",
    )


# ---------------------------------------------------------------------------
# Paper-level metadata
# ---------------------------------------------------------------------------

class ExtractedPaperMetadata(BaseModel):
    curriculum: Optional[str] = Field("", description="Extracted curriculum from the paper (e.g., IGCSE, IB)")
    program: Optional[str] = Field("", description="Extracted program (e.g., MYP, DP, Secondary)")
    subjectCode: Optional[str] = Field("", description="Extracted subject code (e.g., 0580)")
    tier: Optional[str] = Field("", description="Extracted tier (e.g., Core, Extended, SL, HL)")
    paperNumber: Optional[int] = Field(0, description="Extracted paper number (e.g., 1, 2)")
    session: Optional[str] = Field("", description="Extracted session (e.g., May/June)")
    year: Optional[int] = Field(0, description="Extracted year (e.g., 2023)")
    paper_reference_key: Optional[str] = Field(
        "",
        description="Slug that links QP ↔ MS. Generated from filename. Format: YEAR_SUBJECT_PAPER_VARIANT.",
    )
    ref_code_base: Optional[str] = Field("", description="Base reference code without suffix (e.g., 2225-7106)")
    ref_code_full: Optional[str] = Field("", description="Full reference code with suffix (e.g., 2225-7106M)")


# ---------------------------------------------------------------------------
# Top-level response envelope
# ---------------------------------------------------------------------------

class SlicedQuestionsResponse(BaseModel):
    metadata: Optional[ExtractedPaperMetadata] = Field(
        default_factory=ExtractedPaperMetadata,
        description="Global metadata extracted from the paper.",
    )
    questions_array: Optional[List[ExtractedQuestion]] = Field(
        default_factory=list,
        description="List of all questions (QP) or marking scheme entries (MS) extracted from the document.",
    )
