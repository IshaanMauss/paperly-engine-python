import re
import fitz
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

log = logging.getLogger(__name__)

HEADER_CROP_RATIO = 0.12

# Regex: \b(\d{4}[-–—]\d{4,5})([A-Z]?)\b.
# Support standard hyphens (U+002D), en-dash (U+2013), and em-dash (U+2014).
IB_REF_CODE_PATTERN = re.compile(
    r"""\b(\d{4}[-–—]\d{4,5})([A-Z]?)\b""",
    re.VERBOSE,
)

@dataclass
class IBReferenceCode:
    raw: str
    base: str
    suffix: str
    session_prefix: str
    paper_code: str

    @property
    def is_mark_scheme(self) -> bool:
        return self.suffix.upper() == "M"

    @classmethod
    def from_match(cls, m: re.Match) -> "IBReferenceCode":
        # Always normalize to U+002D (-)
        base = m.group(1).replace("–", "-").replace("—", "-")
        suffix = m.group(2) or ""
        parts = base.split("-", 1)
        return cls(
            raw=base + suffix,
            base=base,
            suffix=suffix,
            session_prefix=parts[0],
            paper_code=parts[1] if len(parts) > 1 else "",
        )

def _extract_header_text(pdf_path: str, max_pages: int = 2) -> str:
    texts: list[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_idx in range(min(max_pages, len(doc))):
                page = doc[page_idx]
                rect = page.rect
                header = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * HEADER_CROP_RATIO)
                texts.append(page.get_text("text", clip=header))
    except Exception as e:
        log.warning(f"Failed to extract header from {pdf_path}: {e}")
    return "\n".join(texts)

def _extract_fullpage_text(pdf_path: str, max_pages: int = 2) -> str:
    texts: list[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_idx in range(min(max_pages, len(doc))):
                texts.append(doc[page_idx].get_text("text"))
    except Exception as e:
        log.warning(f"Failed to extract full text from {pdf_path}: {e}")
    return "\n".join(texts)

def regex_extract_ref_code(pdf_path: str) -> Tuple[Optional[IBReferenceCode], str]:
    """
    3-pass strategy to extract IB Reference Code.
    Pass 1: Scan top 12% of pages 1-2 (Header)
    Pass 2: Scan full text of pages 1-2
    """
    # Pass 1: Header scan
    header_text = _extract_header_text(pdf_path)
    matches = list(IB_REF_CODE_PATTERN.finditer(header_text))
    if matches:
        chosen = next((m for m in matches if m.group(2).upper() == "M"), matches[0])
        return IBReferenceCode.from_match(chosen), "regex_header"

    # Pass 2: Full text scan
    full_text = _extract_fullpage_text(pdf_path)
    matches = list(IB_REF_CODE_PATTERN.finditer(full_text))
    if matches:
        chosen = next((m for m in matches if m.group(2).upper() == "M"), matches[0])
        return IBReferenceCode.from_match(chosen), "regex_fulltext"

    return None, "regex_failed"

def normalize_reference_key(raw_code):
    """
    Normalize IB paper reference key:
    - Strip any suffixes (like 'M')
    - Remove slashes
    - Standardize to format like '2225-7106'
    """
    # Remove any session/tier information
    code = re.sub(r'[/_].*', '', raw_code)
    
    # Remove any suffix like 'M'
    code = re.sub(r'[A-Z]$', '', code)
    
    return code.strip()

def extract_reference_code(header_text):
    """
    Extract standard IB reference code from document header
    """
    # Regex patterns to match IB reference codes
    patterns = [
        r'\b(\d{4}-\d{4})\b',  # Standard format like 2225-7106
        r'\b(\d{4}/\d+)\b',    # Alternative format like 7106/1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, header_text)
        if match:
            return normalize_reference_key(match.group(1))
    
    return None