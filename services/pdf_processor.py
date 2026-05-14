"""
pdf_processor.py
================
Utilities for converting PDF pages to images and mathematically cropping
diagram regions for the Decoupled Vision Pattern.

Dependencies: PyMuPDF (fitz), Pillow (PIL)
"""

import asyncio
import base64
import io
import logging
from typing import List, Optional, Tuple

import fitz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Safety padding added on all sides during diagram crop to avoid clipping.
_CROP_PADDING_PCT: float = 0.03  # 3% of page height/width

# JPEG compression quality for cropped diagrams.
# 88 gives excellent sharpness at ~60% of PNG size.
_DIAGRAM_JPEG_QUALITY: int = 88

# DPI used when rasterising full pages for the Vision Engine.
_PAGE_RENDER_DPI: int = 250  # Increased DPI for better readability of small question numbers


# ---------------------------------------------------------------------------
# Public: full-page rasterisation (existing, untouched in signature)
# ---------------------------------------------------------------------------

def pdf_base64_to_jpeg_pages(pdf_base64: str, dpi: int = 300) -> List[str]:
    """
    Decode a base64 PDF and render every page as a JPEG.

    Returns a list of base64-encoded JPEG strings, one per page.
    """
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
                pix = page.get_pixmap(matrix=matrix, alpha=False)

                # Ensure RGB colour space for universal JPEG compatibility.
                if pix.colorspace and pix.colorspace.n != 3:
                    try:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    except Exception as cs_err:
                        logger.warning(f"Colorspace conversion failed on page {page.number}: {cs_err}")

                try:
                    rendered_pages.append(
                        base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
                    )
                except Exception as enc_err:
                    logger.warning(f"JPEG encode failed on page {page.number}: {enc_err}")

            except Exception as page_err:
                logger.warning(f"Error rasterising page {page.number}: {page_err}")
                continue

        return rendered_pages
    finally:
        doc.close()


async def pdf_base64_to_jpeg_pages_async(pdf_base64: str, dpi: int = 300) -> List[str]:
    """Async wrapper around :func:`pdf_base64_to_jpeg_pages`."""
    return await asyncio.to_thread(pdf_base64_to_jpeg_pages, pdf_base64, dpi)


# ---------------------------------------------------------------------------
# NEW: Low-DPI page renders for Vision Engine (faster, cheaper API calls)
# ---------------------------------------------------------------------------

def pdf_base64_to_vision_pages(pdf_base64: str, dpi: int = _PAGE_RENDER_DPI) -> List[str]:
    """
    Render pages at a reduced DPI specifically for the Vision Engine.
    The Vision Engine only needs to *locate* diagrams, not read fine detail,
    so 150 DPI cuts token usage roughly in half vs. 300 DPI.
    """
    return pdf_base64_to_jpeg_pages(pdf_base64, dpi=dpi)


async def pdf_base64_to_vision_pages_async(pdf_base64: str, dpi: int = _PAGE_RENDER_DPI) -> List[str]:
    """Async wrapper around :func:`pdf_base64_to_vision_pages`."""
    return await asyncio.to_thread(pdf_base64_to_vision_pages, pdf_base64, dpi)


# ---------------------------------------------------------------------------
# NEW: crop_and_compress_diagram — the core of the Decoupled Vision Pattern
# ---------------------------------------------------------------------------

def crop_and_compress_diagram(
    pdf_base64: str,
    page_num: int,
    y_start_pct: float,
    y_end_pct: float,
    x_start_pct: float = 0.0,
    x_end_pct: float = 100.0,
    dpi: int = 220,
    jpeg_quality: int = _DIAGRAM_JPEG_QUALITY,
    padding_pct: float = _CROP_PADDING_PCT,
) -> Optional[str]:
    """
    Mathematically crop a rectangular region of a PDF page and return a
    base64-encoded JPEG string suitable for storage / embedding.

    Parameters
    ----------
    pdf_base64     : Base64-encoded PDF (with or without data-URI prefix).
    page_num       : 0-indexed page number.
    y_start_pct    : Top boundary of the crop region, as a percentage of
                     page height (0.0 = top, 100.0 = bottom).
    y_end_pct      : Bottom boundary of the crop region (same scale).
    x_start_pct    : Left boundary (default 0 = full width).
    x_end_pct      : Right boundary (default 100 = full width).
    dpi            : Render resolution. 220 DPI gives sharp diagrams at ~40 KB.
    jpeg_quality   : Pillow JPEG quality (0-95).
    padding_pct    : Safety fringe added on all 4 sides (% of page dimension).

    Returns
    -------
    Base64 JPEG string, or ``None`` on failure.
    """
    if not pdf_base64 or not pdf_base64.strip():
        logger.error("crop_and_compress_diagram: empty pdf_base64 received.")
        return None

    # ------------------------------------------------------------------
    # 1. Decode PDF
    # ------------------------------------------------------------------
    normalized = pdf_base64.strip()
    if "," in normalized:
        normalized = normalized.split(",", 1)[1]

    try:
        pdf_bytes = base64.b64decode(normalized)
    except Exception as e:
        logger.error(f"crop_and_compress_diagram: base64 decode failed: {e}")
        return None

    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # ------------------------------------------------------------------
        # 2. Page bounds check
        # ------------------------------------------------------------------
        num_pages = len(doc)
        if not (0 <= page_num < num_pages):
            logger.warning(
                f"crop_and_compress_diagram: page_num={page_num} out of range "
                f"(doc has {num_pages} pages). Clamping to last page."
            )
            page_num = max(0, min(page_num, num_pages - 1))

        page = doc[page_num]
        pw = page.rect.width   # page width  in PDF points
        ph = page.rect.height  # page height in PDF points

        # ------------------------------------------------------------------
        # 3. Convert percentage coordinates → PDF points, with padding
        # ------------------------------------------------------------------
        def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, val))

        y0_pct = _clamp((y_start_pct / 100.0) - padding_pct)
        y1_pct = _clamp((y_end_pct   / 100.0) + padding_pct)
        x0_pct = _clamp((x_start_pct / 100.0) - padding_pct)
        x1_pct = _clamp((x_end_pct   / 100.0) + padding_pct)

        clip = fitz.Rect(
            x0_pct * pw,   # left
            y0_pct * ph,   # top
            x1_pct * pw,   # right
            y1_pct * ph,   # bottom
        )

        # Guard against degenerate clip rectangles (e.g. Vision Engine hallucination)
        if clip.width < 10 or clip.height < 10:
            logger.warning(
                f"crop_and_compress_diagram: degenerate clip rect {clip} on page {page_num}. "
                "Falling back to full page."
            )
            clip = page.rect

        # ------------------------------------------------------------------
        # 4. Rasterise the clipped region at high DPI
        # ------------------------------------------------------------------
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, clip=clip, colorspace=fitz.csRGB, alpha=False)

        # ------------------------------------------------------------------
        # 5. Compress with Pillow (better JPEG quality control than fitz)
        # ------------------------------------------------------------------
        try:
            from PIL import Image  # Lazy import — Pillow is optional at module level

            img_bytes = pix.tobytes("ppm")  # PPM is lossless, fast bridge to PIL
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        except ImportError:
            # Pillow not installed — fall back to fitz's built-in JPEG encoder
            logger.warning("Pillow not available; using fitz JPEG encoder (no quality control).")
            return base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

        except Exception as pil_err:
            logger.warning(f"Pillow encode failed ({pil_err}); retrying with fitz.")
            try:
                return base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
            except Exception as fitz_err:
                logger.error(f"crop_and_compress_diagram: all encode paths failed: {fitz_err}")
                return None

    except Exception as e:
        logger.error(f"crop_and_compress_diagram: unexpected error: {type(e).__name__}: {e}")
        return None

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


async def crop_and_compress_diagram_async(
    pdf_base64: str,
    page_num: int,
    y_start_pct: float,
    y_end_pct: float,
    x_start_pct: float = 0.0,
    x_end_pct: float = 100.0,
    dpi: int = 220,
    jpeg_quality: int = _DIAGRAM_JPEG_QUALITY,
    padding_pct: float = _CROP_PADDING_PCT,
) -> Optional[str]:
    """Async wrapper around :func:`crop_and_compress_diagram`."""
    return await asyncio.to_thread(
        crop_and_compress_diagram,
        pdf_base64, page_num, y_start_pct, y_end_pct,
        x_start_pct, x_end_pct, dpi, jpeg_quality, padding_pct,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_percentage_range(value: float, name: str) -> Tuple[float, bool]:
    """
    Clamp a percentage value to [0, 100] and return (clamped_value, was_valid).
    Logs a warning if the original value was out of range.
    """
    if 0.0 <= value <= 100.0:
        return value, True
    clamped = max(0.0, min(100.0, value))
    logger.warning(f"{name}={value} out of [0, 100]; clamped to {clamped}")
    return clamped, False


__all__ = [
    "pdf_base64_to_jpeg_pages",
    "pdf_base64_to_jpeg_pages_async",
    "pdf_base64_to_vision_pages",
    "pdf_base64_to_vision_pages_async",
    "crop_and_compress_diagram",
    "crop_and_compress_diagram_async",
]