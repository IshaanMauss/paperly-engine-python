#!/usr/bin/env python3
"""
Test script to verify the substring collision bug fix.
Tests both Python builders/key_builder.py and ensures regex word boundaries work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from builders.key_builder import _extract_session_from_content

def test_substring_collision_fix():
    """Test that the regex word boundary fix prevents false matches."""
    
    print("🧪 Testing Substring Collision Bug Fix")
    print("=" * 50)
    
    # Test cases that should NOT match (these were the false positives)
    # These test the core bug: substring matching within words
    false_positive_cases = [
        ("Maximum Mark 130", None, "Should NOT match 'mar' from 'Mark'"),
        ("MaRketing scheme document", None, "Should NOT match 'mar' from 'Marketing'"),  
        ("Maritime studies examination", None, "Should NOT match 'mar' from 'Maritime'"),
        ("Remarkable results achieved", None, "Should NOT match 'mar' from 'remarkable'"),
        ("This is a remarkable exam", None, "Should NOT match 'mar' from 'remarkable'"),
        ("Smart solutions for problems", None, "Should NOT match 'mar' from 'Smart'"),
        ("Summary of the document", None, "Should NOT match 'sum' from 'Summary'"),
        ("Summarize the content", None, "Should NOT match 'sum' from 'Summarize'"),
    ]
    
    # Test cases showing that legitimate keywords still work alongside the fix
    mixed_cases = [
        ("May/June Maximum Mark 130", "s", "Should match 'May/June' but ignore 'mar' from 'Mark'"),
        ("October November Maritime studies", "w", "Should match 'October November' but ignore 'mar' from 'Maritime'"),
        ("Summer session with remarkable results", "s", "Should match 'Summer' but ignore 'mar' from 'remarkable'"),
        ("March towards remarkable success", "m", "Should match 'March' but ignore 'mar' from 'remarkable'"),
    ]
    
    # Test cases that SHOULD match (legitimate matches)
    true_positive_cases = [
        ("March 2024 examination", "m", "Should match standalone 'March'"),
        ("February/March session", "m", "Should match standalone 'March'"),
        ("feb mar session", "m", "Should match standalone 'mar'"),
        ("May June session", "s", "Should match 'May' and 'June'"),
        ("Summer examination", "s", "Should match 'Summer'"),
        ("October November session", "w", "Should match 'October' and 'November'"),
        ("october nov examination", "w", "Should match 'october' and 'nov'"),
        ("Winter examination period", "w", "Should match 'Winter'"),
    ]
    
    all_passed = True
    
    print("\n🔴 Testing False Positive Prevention:")
    for test_input, expected, description in false_positive_cases:
        result = _extract_session_from_content(test_input)
        success = result == expected
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | '{test_input}' -> {result} | {description}")
        if not success:
            all_passed = False
    
    print("\n🟡 Testing Mixed Cases (Should match legitimate keywords, ignore substring matches):")
    for test_input, expected, description in mixed_cases:
        result = _extract_session_from_content(test_input)
        success = result == expected
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | '{test_input}' -> {result} | {description}")
        if not success:
            all_passed = False

    print("\n🟢 Testing True Positive Detection:")
    for test_input, expected, description in true_positive_cases:
        result = _extract_session_from_content(test_input)
        success = result == expected
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | '{test_input}' -> {result} | {description}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Substring collision bug is fixed.")
        return True
    else:
        print("❌ SOME TESTS FAILED! Review the regex implementation.")
        return False

if __name__ == "__main__":
    test_substring_collision_fix()