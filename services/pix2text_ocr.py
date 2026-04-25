# File: services/pix2text_ocr.py
import os
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
from services.pipeline_errors import PipelineServiceError

load_dotenv()

def _get_client() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY"):
        raise PipelineServiceError(
            stage="ocr",
            message="GEMINI_API_KEY is not configured.",
            details={"provider": "gemini"},
        )
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def extract_latex_from_image(base64_image: str) -> str:
    print("🧠 [Production] Auto-detecting and extracting via Gemini...")
    
    try:
        client = _get_client()
        # Base64 string ko clean karna
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_image)

        # 1. Fetch available models dynamically
        models = client.models.list()
        
        # 2. Filter for vision-capable models (1.5-flash or 2.0-flash)
        vision_model = next((m for m in models if "flash" in m.name and "vision" not in m.name), None)

        if not vision_model:
            raise PipelineServiceError(
                stage="ocr",
                message="No compatible Gemini flash model is available.",
                details={"provider": "gemini"},
            )

        # 3. Use the dynamically found model name with the STRICT Prompt
       # Purana logic (jo list kar raha tha) hata do aur yeh use karo:
        
        # 3. Use the stable model directly
        model_name = "gemini-2.0-flash" 
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                """
                You are a STRICT and DETERMINISTIC Optical Character Recognition (OCR) engine.
                Your ONLY job is to transcribe the ENTIRE image exactly as it appears, line-by-line, top-to-bottom.
                
                CRITICAL RULES:
                1. DO NOT summarize, rephrase, or analyze. 
                2. DO NOT SKIP ANYTHING.
                3. Convert all math symbols, equations, and expressions into exact LaTeX wrapped in $ symbols.
                4. Return ONLY the raw text. No greetings, no explanations.
                """,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"❌ [Gemini SDK Error]: {str(e)}")
        if isinstance(e, PipelineServiceError):
            raise
        raise PipelineServiceError(
            stage="ocr",
            message="Failed to extract OCR output from image.",
            details={"provider": "gemini", "reason": str(e)},
        ) from e