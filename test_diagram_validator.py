
import asyncio
import base64
import json

from services.diagram_validator import validate_diagrams_from_pdf


async def main():
    """
    Test the diagram validator service.
    """
    pdf_path = "c:/Users/ishaa/OneDrive/Desktop/0580_s23_ms_41.pdf"
    with open(pdf_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    validated_diagrams = await validate_diagrams_from_pdf(pdf_base64)
    print(json.dumps(validated_diagrams, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
