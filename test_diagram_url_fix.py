"""
Test script to verify diagram URL array formatting in the Python engine
"""
import json
from typing import List, Any, Dict, Union


def sanitize_diagram_urls(diagram_urls: Any) -> List[str]:
    """
    Sanitize diagram URLs to ensure they are a flat array of valid strings
    
    Args:
        diagram_urls: Any type of input that should be converted to a list of strings
        
    Returns:
        List[str]: A flat array of valid diagram URL strings
    """
    # Handle None case
    if diagram_urls is None:
        return []
    
    # Handle single string
    if isinstance(diagram_urls, str):
        return [diagram_urls] if diagram_urls.strip() else []
    
    # Handle non-iterables
    if not hasattr(diagram_urls, '__iter__') or isinstance(diagram_urls, dict):
        return []
    
    # Recursively flatten nested lists
    def flatten_deep(items):
        result = []
        for item in items:
            if isinstance(item, list) or (hasattr(item, '__iter__') and not isinstance(item, (str, dict))):
                result.extend(flatten_deep(item))
            elif isinstance(item, str) and item.strip():
                result.append(item)
        return result
    
    flat_urls = flatten_deep(diagram_urls)
    
    # Ensure all items are valid strings
    return [url for url in flat_urls if isinstance(url, str) and url.strip()]


def is_valid_url(url: Any) -> bool:
    """
    Check if a string looks like a valid URL
    
    Args:
        url: Any value to check
        
    Returns:
        bool: True if valid URL string, False otherwise
    """
    if not isinstance(url, str):
        return False
    
    # Basic URL validation - protocol required
    return url.startswith(('http://', 'https://', 'data:'))


# Test cases for sanitize_diagram_urls
def run_tests():
    """Run test cases for diagram URL sanitization"""
    print("TEST CASES FOR DIAGRAM URL SANITIZATION")
    print("=======================================")
    
    # Test case 1: Already valid array of strings
    test1 = ['https://example.com/img1.jpg', 'https://example.com/img2.jpg']
    print(f"Test 1 (Valid array of strings): {sanitize_diagram_urls(test1)}")
    
    # Test case 2: Nested array (the bug case)
    test2 = [[]]
    print(f"Test 2 (Nested empty array): {sanitize_diagram_urls(test2)}")
    
    # Test case 3: Deeply nested array
    test3 = [[['https://example.com/img1.jpg']]]
    print(f"Test 3 (Deeply nested array): {sanitize_diagram_urls(test3)}")
    
    # Test case 4: Mixed types
    test4 = ['https://example.com/img1.jpg', None, '', 123, {}, []]
    print(f"Test 4 (Mixed types): {sanitize_diagram_urls(test4)}")
    
    # Test case 5: Null/None
    print(f"Test 5 (None): {sanitize_diagram_urls(None)}")
    
    # Test case 6: Non-iterable
    print(f"Test 7 (Dict): {sanitize_diagram_urls({'url': 'https://example.com/img1.jpg'})}")
    
    # Test case 8: Single string
    print(f"Test 8 (Single string): {sanitize_diagram_urls('https://example.com/img1.jpg')}")
    

    # Test cases for is_valid_url
    print("\nTEST CASES FOR URL VALIDATION")
    print("============================")
    
    # Valid URLs
    print(f"http://example.com: {is_valid_url('http://example.com')}")
    print(f"https://example.com: {is_valid_url('https://example.com')}")
    print(f"data:image/png;base64,...: {is_valid_url('data:image/png;base64,abc123')}")
    
    # Invalid URLs
    print(f"ftp://example.com: {is_valid_url('ftp://example.com')}")
    print(f"example.com (no protocol): {is_valid_url('example.com')}")
    print(f"Empty string: {is_valid_url('')}")
    print(f"Non-string (number): {is_valid_url(123)}")
    print(f"Non-string (dict): {is_valid_url({})}")
    print(f"Non-string (None): {is_valid_url(None)}")


# Run the tests
if __name__ == "__main__":
    run_tests()

# Fix recommendation for gemini_pdf_service.py line ~750:
"""
# Before:
diagram_urls = self.extract_diagrams(...)

# After:
diagram_urls = self.extract_diagrams(...)
# Ensure diagram_urls is a flat array of strings
if diagram_urls:
    # Apply sanitization function
    diagram_urls = sanitize_diagram_urls(diagram_urls)
"""