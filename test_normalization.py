import re

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

# Test cases
test_keys = [
    '2225-7106M',         # Should normalize to 2225-7106
    '7106/1_HL_May_2025', # Should normalize to 7106
    '2225-7106',          # Already normalized
    '7106M',              # Should normalize to 7106
    '7106/1',             # Should normalize to 7106
]

print("Testing reference key normalization:")
for key in test_keys:
    normalized = normalize_reference_key(key)
    print(f"'{key}' -> '{normalized}'")