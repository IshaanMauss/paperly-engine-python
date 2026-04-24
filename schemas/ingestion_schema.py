# File: schemas/ingestion_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional

# Single Question Definition
class ExtractedQuestion(BaseModel):
    board: str = Field(description="The educational board, e.g., 'IGCSE', 'IB'")
    code: str = Field(description="The subject code, e.g., '0580', 'AA'")
    topic: str = Field(description="The mathematical topic, e.g., 'Algebra', 'Trigonometry'")
    difficulty: str = Field(description="Estimated difficulty: 'Easy', 'Medium', 'Hard'")
    question: str = Field(description="Question text extracted from the source image")
    question_type: str = Field(
        default="SUBJECTIVE",
        description="Question type classification: MCQ or SUBJECTIVE",
    )
    options: List[str] = Field(
        default_factory=list,
        description="MCQ options array in display order; empty for subjective questions",
    )
    latex: str = Field(description="The exact LaTeX code for the question")
    marking_scheme_latex: Optional[str] = Field(
        default="",
        description="Optional LaTeX code for marking scheme steps",
    )
    official_marking_scheme_latex: Optional[str] = Field(
        default="",
        description="Official marking scheme answer and mark points for review table",
    )
    question_number: Optional[str] = Field(
        default="",
        description="Normalized question number key (used heavily for marking schemes)",
    )
    document_type: str = Field(
        default="Question Paper",
        description="Source document type: Question Paper or Marking Scheme",
    )
    diagram_image_base64: Optional[str] = Field(
        default=None,
        description="Optional diagram crop payload from frontend proofreader",
    )

# The Array that Groq will return
class SlicedQuestionsResponse(BaseModel):
    questions_array: List[ExtractedQuestion] = Field(description="List of all questions extracted from the page")