import time
import re
import logging
from extractors.ref_code_extractor import normalize_reference_key

logger = logging.getLogger(__name__)

def build_paper_reference_key(document_metadata):
    """
    Build a normalized paper reference key based on document metadata
    
    If all extraction passes fail to find a reference code, generate a fallback key
    using the format: UNKNOWN_REF_<unix_timestamp>_<sanitized first 10 chars of filename>
    
    Any document saved with an UNKNOWN_REF base will have needs_review: true set automatically.
    """
    raw_code = document_metadata.get('reference_code', '')
    
    if not raw_code:
        # Generate fallback key using timestamp and filename
        timestamp = int(time.time())
        filename = document_metadata.get('filename', 'unknown')
        
        # Sanitize filename - remove special chars and take first 10 chars
        sanitized_name = re.sub(r'[^a-zA-Z0-9]', '', filename)[:10]
        
        fallback_key = f"UNKNOWN_REF_{timestamp}_{sanitized_name}"
        
        logger.warning(f"Empty reference key detected. Using fallback: {fallback_key}")
        
        # Set needs_review flag to true in document_metadata
        document_metadata['needs_review'] = True
        
        return fallback_key
    
    return normalize_reference_key(raw_code)
