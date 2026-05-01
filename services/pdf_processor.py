import base64
import asyncio
import logging
from typing import List

import fitz

logger = logging.getLogger(__name__)


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
            try:
                # Get the pixmap with alpha=False to ensure it's in the right format
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Check if we need to convert color space to RGB
                if pix.colorspace and pix.colorspace.n != 3:  # Not RGB
                    try:
                        # Convert to RGB format to ensure compatibility
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    except Exception as e:
                        logger.warning(f"Failed to convert pixmap to RGB: {e}")
                
                # Convert to bytes with error handling
                try:
                    rendered_pages.append(base64.b64encode(pix.tobytes("jpeg")).decode("utf-8"))
                except Exception as e:
                    logger.warning(f"Failed to convert pixmap to bytes: {e}")
            except Exception as e:
                logger.warning(f"Error processing page pixmap: {e}")
                # Continue processing other pages instead of failing completely
                continue

        return rendered_pages
    finally:
        doc.close()


async def pdf_base64_to_jpeg_pages_async(pdf_base64: str, dpi: int = 300) -> List[str]:
    return await asyncio.to_thread(pdf_base64_to_jpeg_pages, pdf_base64, dpi)


__all__ = ["pdf_base64_to_jpeg_pages", "pdf_base64_to_jpeg_pages_async"]
