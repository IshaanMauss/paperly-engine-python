import asyncio
import base64
import json
import os
import tempfile
import time

from google import genai

from schemas.ingestion_schema import ExtractedQuestion
from services.pipeline_errors import PipelineServiceError


def _build_pdf_system_prompt(document_type: str) -> str:
    if (document_type or "").strip().lower() == "marking scheme":
        return f"""
You are an IGCSE/IB mathematics MARKING SCHEME extraction engine with strict JSON output behavior.

CONTEXT
- This is a Marking Scheme in a tabular format (typically Question No, Answer, Marks).
- You must extract each meaningful row/group and map by question number.

OUTPUT FORMAT (STRICT)
- Return ONLY a single JSON object (no markdown, no prose):
  {{
    "questions_array": [ ... ]
  }}
- Each item in questions_array must follow:
  {{
    "board": "",
    "code": "",
    "topic": "",
    "difficulty": "Medium",
    "question": "",
    "question_type": "SUBJECTIVE",
    "options": [],
    "latex": "",
    "marking_scheme_latex": "",
    "official_marking_scheme_latex": "",
    "document_type": "{document_type}"
  }}

MARKING SCHEME EXTRACTION RULES (CRITICAL)
- Extract each row/group from the table where question mapping is possible.
- Use Question Number as the mapping key in "question" (e.g., "Q1(a)(i)", "Question 5").
- CRITICAL: You MUST extract the main top-level question number (1, 2, 3, etc.). Do not drop the main integer. If a question is '3 (a) (i)', the question_number MUST be '3(a)(i)'. NEVER extract just the sub-part like '(i)' or '(a)' without its parent number.
- Put mathematical answer content and marks (e.g., M1, A1, B1, ft, oe) in
  "official_marking_scheme_latex" and mirror the same value in "marking_scheme_latex".
- Set "question_type" to "SUBJECTIVE" for all marking-scheme rows.
- Keep "options" as [].
- Preserve hierarchy/new lines with escaped newlines (\\n, \\n\\n) when rows contain subparts.

STRICT LATEX ESCAPING (CRITICAL)
- You MUST extract ALL mathematical formulas, equations, fractions, variables, and symbols STRICTLY as raw LaTeX code.
- DO NOT use unicode math characters (like ², ³, ±, α, ∫). Instead, use their exact LaTeX equivalents (e.g., ^2, ^3, \\\\pm, \\\\alpha, \\\\int).
- Wrap inline math with single dollar signs (e.g., $x^2 + y^2 = 0$) and block math with double dollar signs ($$...$$).
- The output must be the raw LaTeX string, ready to be saved into a database and parsed by KaTeX later.
- Double-escape all LaTeX backslashes for JSON validity.
- Example: write \\\\frac not \\frac.

SPECIAL RULE
- Do NOT apply a "no numbered questions => empty array" rule for Marking Schemes.
- If content is sparse, still extract all mappable rows you can identify.
""".strip()

    return f"""
You are an IGCSE/IB mathematics paper extraction engine with strict JSON output behavior.

TARGET
- Extract all mathematical questions from the provided PDF.
- Ignore blank spaces, cover pages, irrelevant headers/footers, and purely decorative content.

OUTPUT FORMAT (STRICT)
- Return ONLY a single JSON object (no markdown, no prose) with this top-level key:
  {{
    "questions_array": [ ... ]
  }}
- Each item in questions_array must follow:
  {{
    "board": "",
    "code": "",
    "topic": "",
    "difficulty": "Medium",
    "question": "",
    "question_type": "SUBJECTIVE",
    "options": [],
    "latex": "",
    "marking_scheme_latex": "",
    "document_type": "{document_type}"
  }}

QUESTION TYPE RULES
- If a question has options A/B/C/D, set "question_type" to "MCQ".
- Put options in "options" array in display order.
- For MCQ, keep "question" as the stem and keep full math text in "latex" as needed.
- If not MCQ, set "question_type" to "SUBJECTIVE" and "options": [].

NESTED IGCSE/IB QUESTION GROUPING (CRITICAL)
- For SUBJECTIVE questions, DO NOT split sub-parts like (a), (b), (i), (ii) into separate objects.
- Group each parent numbered question (e.g., 1, 2, 3...) into ONE JSON object.
- Preserve exact visual hierarchy inside "question" using explicit escaped newlines (\\n and \\n\\n).
- CRITICAL: You MUST extract the main top-level question number (1, 2, 3, etc.). Do not drop the main integer. If a question is '3 (a) (i)', the question_number MUST be '3(a)(i)'. NEVER extract just the sub-part like '(i)' or '(a)' without its parent number.

MARKING SCHEME TABLES (CRITICAL)
- If document type is "Marking Scheme", parse marking-scheme table rows and map them into the same schema.
- Put the scheme/table detail in "marking_scheme_latex" with valid escaped text/LaTeX.
- Keep each parent numbered question aligned to its corresponding marking scheme content.
- If the uploaded document is a "Question Paper", you MUST leave "official_marking_scheme_latex" empty ("" or null).
- NEVER copy question text into "official_marking_scheme_latex" for Question Paper uploads.

STRICT LATEX ESCAPING (CRITICAL)
- You MUST extract ALL mathematical formulas, equations, fractions, variables, and symbols STRICTLY as raw LaTeX code.
- DO NOT use unicode math characters (like ², ³, ±, α, ∫). Instead, use their exact LaTeX equivalents (e.g., ^2, ^3, \\\\pm, \\\\alpha, \\\\int).
- Wrap inline math with single dollar signs (e.g., $x^2 + y^2 = 0$) and block math with double dollar signs ($$...$$).
- The output must be the raw LaTeX string, ready to be saved into a database and parsed by KaTeX later.
- Double-escape all LaTeX backslashes for JSON validity.
- Example: write \\\\frac not \\frac.

EMPTY CONTENT RULE
- If no extractable mathematical questions exist, return:
  {{ "questions_array": [] }}
""".strip()


def _parse_json_payload(content: str) -> dict:
    if not content or not content.strip():
        return {"questions_array": []}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _wait_for_file_ready(client: genai.Client, file_name: str, timeout_seconds: int = 90):
    deadline = time.time() + timeout_seconds
    last_state = "UNKNOWN"

    # google-generativeai may expose file state as enum name or integer code.
    # Observed codes include ACTIVE=2 and FAILED=3.
    state_code_map = {
        0: "STATE_UNSPECIFIED",
        1: "PROCESSING",
        2: "ACTIVE",
        3: "FAILED",
    }

    def _normalize_state(state_value) -> str:
        if state_value is None:
            return "UNKNOWN"
        name = getattr(state_value, "name", None)
        if isinstance(name, str) and name:
            return name.upper()
        if isinstance(state_value, int):
            return state_code_map.get(state_value, str(state_value))
        try:
            numeric = int(state_value)
            return state_code_map.get(numeric, str(numeric))
        except Exception:
            return str(state_value).upper() or "UNKNOWN"

    while time.time() < deadline:
        remote_file = client.files.get(name=file_name)
        state = _normalize_state(getattr(remote_file, "state", None))
        last_state = state

        if "ACTIVE" in last_state:
            return remote_file
        if "FAILED" in last_state:
            raise RuntimeError(f"Uploaded file entered FAILED state: {last_state}")

        time.sleep(1.5)

    raise TimeoutError(f"Uploaded file was not ACTIVE before timeout. Last state: {last_state}")


def _pick_available_model(client: genai.Client) -> str:
    preferred = [
        "models/gemini-1.5-pro-latest",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
    ]
    try:
        available = {m.name for m in client.models.list()}
    except Exception:
        # If model listing fails, keep best-effort default.
        return "gemini-1.5-flash-latest"

    for model_name in preferred:
        if model_name in available:
            return model_name.replace("models/", "")

    # Last resort, use any Gemini text model visible to this API key.
    for model_name in available:
        if model_name.startswith("models/gemini-"):
            return model_name.replace("models/", "")

    return "gemini-1.5-flash-latest"


def _extract_pdf_native_sync(pdf_base64: str, document_type: str):
    if not pdf_base64 or not pdf_base64.strip():
        return []

    normalized = pdf_base64.strip()
    if "," in normalized:
        normalized = normalized.split(",", 1)[1]

    uploaded_file = None
    temp_file_path = None
    client = None

    try:
        client = _get_client()
        pdf_bytes = base64.b64decode(normalized)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name

        uploaded_file = client.files.upload(file=temp_file_path)
        _wait_for_file_ready(client, uploaded_file.name, timeout_seconds=240)
        selected_model_name = "gemini-1.5-flash-latest"
        print(f"ℹ️ [Gemini Native PDF] Using model: {selected_model_name}")
        try:
            response = client.models.generate_content(
                model=selected_model_name,
                contents=[_build_pdf_system_prompt(document_type), uploaded_file],
                config={"response_mime_type": "application/json"},
            )
        except Exception as model_not_found_exc:
            print(
                "❌ [Gemini Native PDF Error] Model not found. "
                "Please ensure your API key has access to Gemini 1.5 and your "
                "google-generativeai package is updated."
            )
            # Optional fallback to a second stable alias.
            try:
                fallback_model_name = _pick_available_model(client)
                print(f"ℹ️ [Gemini Native PDF] Fallback model: {fallback_model_name}")
                response = client.models.generate_content(
                    model=fallback_model_name,
                    contents=[_build_pdf_system_prompt(document_type), uploaded_file],
                    config={"response_mime_type": "application/json"},
                )
            except NotFound as fallback_not_found_exc:
                raise PipelineServiceError(
                    stage="pdf_native_gemini",
                    message=(
                        "Model not found. Please ensure your API key has access to Gemini 1.5 "
                        "and your google-generativeai package is updated."
                    ),
                    details={
                        "provider": "gemini",
                        "reason": str(fallback_not_found_exc),
                        "exception_type": type(fallback_not_found_exc).__name__,
                        "models_tried": [selected_model_name, fallback_model_name],
                    },
                ) from fallback_not_found_exc

        parsed = _parse_json_payload(getattr(response, "text", "") or "")
        questions = parsed.get("questions_array", [])
        parsed_json_data = questions

        # Force wipe marking scheme if it's a Question Paper
        if document_type.lower() == "question paper":
            for q in parsed_json_data:
                q["official_marking_scheme_latex"] = ""

        normalized_questions = []
        for question in questions:
            payload = dict(question)
            payload["document_type"] = payload.get("document_type") or document_type
            payload["question_type"] = payload.get("question_type") or "SUBJECTIVE"
            payload["options"] = payload.get("options") if isinstance(payload.get("options"), list) else []
            payload["question_number"] = payload.get("question_number") or payload.get("question") or ""
            if (document_type or "").strip().lower() == "marking scheme":
                payload["official_marking_scheme_latex"] = (
                    payload.get("official_marking_scheme_latex")
                    or payload.get("marking_scheme_latex")
                    or ""
                )
            else:
                payload["official_marking_scheme_latex"] = ""
            payload["marking_scheme_latex"] = (
                payload.get("marking_scheme_latex")
                or payload.get("official_marking_scheme_latex")
                or ""
            )
            normalized_questions.append(ExtractedQuestion(**payload))

        return normalized_questions
    except Exception as exc:
        print(f"❌ [Gemini Native PDF Error] {type(exc).__name__}: {exc!r}")
        raise PipelineServiceError(
            stage="pdf_native_gemini",
            message="Failed to extract structured questions from PDF using Gemini File API.",
            details={"provider": "gemini", "reason": str(exc), "exception_type": type(exc).__name__},
        ) from exc
    finally:
        if client is not None and uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


async def extract_pdf_native_gemini(pdf_base64: str, document_type: str):
    return await asyncio.to_thread(_extract_pdf_native_sync, pdf_base64, document_type)


__all__ = ["extract_pdf_native_gemini"]
