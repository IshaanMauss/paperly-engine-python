import base64
import asyncio
from typing import List

import fitz


def pdf_base64_to_jpeg_pages(pdf_base64: str, dpi: int = 300) -> List[str]:
    if not pdf_base64 or not pdf_base64.strip():
        return []

    normalized = pdf_base64.strip()
    if "," in normalized:
        normalized = normalized.split(",", 1)[1]

    pdf_bytes = base64.b64decode(normalized)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        rendered_pages: List[str] = []

        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            rendered_pages.append(base64.b64encode(pix.tobytes("jpeg")).decode("utf-8"))

        return rendered_pages
    finally:
        doc.close()


async def pdf_base64_to_jpeg_pages_async(pdf_base64: str, dpi: int = 300) -> List[str]:
    return await asyncio.to_thread(pdf_base64_to_jpeg_pages, pdf_base64, dpi)


__all__ = ["pdf_base64_to_jpeg_pages", "pdf_base64_to_jpeg_pages_async"]
