# File: services/groq_slicer.py
import json
import os
import re
from typing import List

from dotenv import load_dotenv
from groq import Groq

from schemas.ingestion_schema import ExtractedQuestion
from services.pipeline_errors import PipelineServiceError

load_dotenv()

def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=2,
    )

def _build_prompt(raw_latex: str, document_type: str) -> str:
    return f"""
You are a highly strict math data extraction parser.

CRITICAL RULES FOR EXTRACTION:
1. STRICT JSON ESCAPING (VITAL): You MUST double-escape all LaTeX backslashes to output valid JSON. Write \\\\frac instead of \\frac, \\\\sin instead of \\sin. Failure to do this will break the JSON parser.
2. ONLY NUMBERED QUESTIONS: You MUST ONLY extract real questions that start with an Arabic numeral (e.g., "1.", "2.", "3.", "10.").
3. BLOCK-LEVEL EXTRACTION ONLY (CRITICAL): One numbered parent question block must map to one JSON object. Do NOT split sub-parts like (a), (b), (i), (ii), 1.1, 1.2 into separate objects.
4. FULL BLOCK RETENTION (CRITICAL): Capture the ENTIRE numbered block exactly as shown, including stem, diagrams/graph references, and all sub-parts until the next numbered parent question starts.
5. PRESERVE VISUAL HIERARCHY (CRITICAL): Keep original line breaks and spacing structure inside the "question" string using explicit escaped newlines (\\\\n and \\\\n\\\\n). This is required to preserve layout in the frontend.
6. VERBATIM CONTENT: Do NOT paraphrase, summarize, or invent content. Keep symbols, options, and wording exactly as extracted.
7. QUESTION TYPE CLASSIFICATION (CRITICAL): Detect whether a block is MCQ or SUBJECTIVE.
8. MCQ EXTRACTION RULE (CRITICAL): If options A/B/C/D exist, set "question_type" to "MCQ", keep only the stem in "question", and put the exact 4 option texts (with original LaTeX/text) into "options" in order.
9. SUBJECTIVE EXTRACTION RULE (CRITICAL): For normal/multiline questions with no A/B/C/D options, set "question_type" to "SUBJECTIVE" and set "options" to [].
10. OPTION INTEGRITY: Do not paraphrase option text. Preserve symbols, spacing intent, and escaped newlines where needed.
11. NO CONTEXT PREPENDING: Do NOT prepend parent instructions to child lines. Keep the block exactly once to avoid redundancy.
12. NO HALLUCINATED LATEX: Do not add math or LaTeX that is not present in the OCR text.
13. EMPTY PAGE RULE (CRITICAL): If a page has no numbered mathematical questions (e.g., cover page, formula sheet, instructions-only page, blank page), return exactly "questions_array": [].
14. DO NOT SPLIT SUBJECTIVE MULTILINE BLOCKS: Keep multiline subjective blocks exactly as visually shown, preserving \\\\n where needed.

Output exactly in this JSON format:
{{
  "questions_array": [
    {{
      "board": "",
      "code": "",
      "topic": "",
      "difficulty": "Medium",
      "question": "Entire numbered question block with preserved line breaks using escaped newlines, e.g. The following diagram shows...\\\\n\\\\n(a) Use the graph to...\\\\n(i) a;\\\\n(ii) c;\\\\n\\\\n(b) Show that...",
      "question_type": "SUBJECTIVE",
      "options": [],
      "latex": "Double-escaped LaTeX here",
      "document_type": "{document_type}"
    }}
  ]
}}

OCR Text to Process:
{raw_latex}
""".strip()

def _parse_json_payload(content: str) -> dict:
    if not content or not content.strip():
        return {"questions_array": []}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Defensive recovery: some model responses include extra wrappers/text
        # around a valid JSON object. Try extracting the outer-most object.
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise

def _split_numbered_blocks(raw_latex: str) -> List[str]:
    pattern = re.compile(r"(?m)^\s*\d+\s*[\.\)]\s+")
    matches = list(pattern.finditer(raw_latex))
    if not matches:
        return [raw_latex.strip()]

    blocks: List[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_latex)
        block = raw_latex[start:end].strip()
        if block:
            blocks.append(block)
    return blocks or [raw_latex.strip()]

def _has_numbered_questions(raw_latex: str) -> bool:
    return bool(re.search(r"(?m)^\s*\d+\s*[\.\)]\s+", raw_latex or ""))

def _chunk_blocks(blocks: List[str], max_chars: int = 7500, max_questions: int = 8) -> List[str]:
    chunks: List[str] = []
    current_blocks: List[str] = []
    current_size = 0

    for block in blocks:
        block_size = len(block)
        would_exceed = (
            current_blocks
            and (current_size + block_size > max_chars or len(current_blocks) >= max_questions)
        )
        if would_exceed:
            chunks.append("\n\n".join(current_blocks))
            current_blocks = [block]
            current_size = block_size
        else:
            current_blocks.append(block)
            current_size += block_size

    if current_blocks:
        chunks.append("\n\n".join(current_blocks))

    return chunks or ["\n\n".join(blocks)]

def _extract_chunk(
    client: Groq,
    raw_chunk: str,
    document_type: str,
) -> List[ExtractedQuestion]:
    last_error = None
    for attempt in range(2):
        retry_suffix = ""
        if attempt == 1:
            retry_suffix = (
                "\n\nRETRY MODE: Your previous response failed JSON parsing. "
                "Return ONLY one valid JSON object, no prose, no markdown, no trailing text."
            )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You convert OCR math content into strict JSON for an API.",
                },
                {
                    "role": "user",
                    "content": _build_prompt(raw_chunk, document_type) + retry_suffix,
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=8192,
        )
        content = completion.choices[0].message.content or '{"questions_array": []}'

        try:
            parsed = _parse_json_payload(content)
            questions = parsed.get("questions_array", [])
            normalized: List[ExtractedQuestion] = []
            for question in questions:
                payload = dict(question)
                payload["document_type"] = payload.get("document_type") or document_type
                payload["question_type"] = payload.get("question_type") or "SUBJECTIVE"
                payload["options"] = payload.get("options") if isinstance(payload.get("options"), list) else []
                normalized.append(ExtractedQuestion(**payload))
            return normalized
        except Exception as parse_exc:
            last_error = parse_exc

    raise last_error or RuntimeError("Unknown slicer parse failure")

def slice_and_format_questions(raw_latex: str, document_type: str = "Question Paper") -> List[ExtractedQuestion]:
    """
    Slice and format questions from OCR text.
    
    This is a synchronous function that will be called from asyncio.to_thread
    """
    if not raw_latex or not raw_latex.strip():
        return []
    if not _has_numbered_questions(raw_latex):
        return []

    try:
        client = _get_client()
        blocks = _split_numbered_blocks(raw_latex)
        chunks = _chunk_blocks(blocks)

        all_questions: List[ExtractedQuestion] = []
        for chunk in chunks:
            chunk_questions = _extract_chunk(client, chunk, document_type)
            all_questions.extend(chunk_questions)

        return all_questions
    except Exception as exc:
        print(f"❌ [Groq Slicer Error]: {exc}")
        raise PipelineServiceError(
            stage="slicer",
            message="Failed to structure extracted questions.",
            details={"provider": "groq", "reason": str(exc)},
        ) from exc

__all__ = ["slice_and_format_questions"]