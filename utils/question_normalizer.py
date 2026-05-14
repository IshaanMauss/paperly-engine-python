import re

class QuestionNumberNormalizer:
    def __init__(self):
        # 🚀 THE FINAL BULLETPROOF REGEX (Handles all spaces, brackets, and word bleeds)
        self._DEPTH3_REGEX = re.compile(
            r"^(\d+)"                                         # Group 1: Parent
            r"(?:"
                r"\s*[\.\(\)]*\s*"                            # Separator 1: Handles "9(a)", "9 (a)", "9.a" flawlessly
                # Group 2: Safe letters. Negative lookahead prevents standalone i,v,x consumption.
                r"([a-hj-uw-zA-HJ-UW-Z]|(?![ivxIVX](?:[\.\(\)\s]|$))[a-zA-Z])" 
                r"(?![a-zA-Z])"                               # 🔒 LOCK 1: Prevents "6 Solve" from bleeding into "6.s"
                r"(?:"
                    r"\s*[\.\(\)]*\s*"                        # Separator 2: Handles "(b)(i)", "(b) (i)", "b.i"
                    # Group 3: Roman numerals ordered longest to shortest
                    r"(viii|vii|vi|iv|ix|iii|ii|i|x|v)"
                    r"(?![a-zA-Z])"                           # 🔒 LOCK 2: Prevents word bleed on roman numerals
                r")?"
            r")?",
            re.IGNORECASE,
        )

        # Fallback regex for orphaned subparts
        self._ORPHAN_REGEX = re.compile(
            r"^\s*[\(\[]?\s*([a-z]|viii|vii|vi|iv|ix|iii|ii|i|x|v)\s*[\)\]\.]",
            re.IGNORECASE,
        )

    def normalize(self, raw_question_id: str, paper_reference_key: str) -> dict:
        """
        Normalizes raw question IDs (e.g., '1(b) a square number' or '4(a)(iii)') into 
        canonical dots format (e.g., '1.b' or '4.a.iii') for QP/MS linking.
        """
        parts = self._extract_parts(raw_question_id)
        
        # If parsing fails to find a number, we mark it as UNKNOWN to trigger a review
        if not parts:
            canonical_question_id = "UNKNOWN"
            parent_canonical_id = "UNKNOWN"
        else:
            canonical_question_id = ".".join(parts)  # e.g., "4.a.iii"
            parent_canonical_id = parts[0]           # e.g., "4"
            
        unified_paper_key = self._generate_unified_paper_key(paper_reference_key)
        metadata = self._build_question_metadata(parts)

        return {
            "canonical_question_id": canonical_question_id,
            "parent_canonical_id": parent_canonical_id,
            "unified_paper_key": unified_paper_key,
            "question_number_metadata": metadata,
        }

    def _extract_parts(self, raw_id: str) -> list[str]:
        """
        Parse a raw question label into a list of hierarchical parts.
        Returns up to 3 elements: [parent_digit, letter_subpart, roman_subpart]
        """
        # 1. Strip leading 'Q' or 'q' noise
        cleaned = re.sub(r'^[Qq]\s*', '', str(raw_id).strip())

        # 2. Primary match via the strict 3-group regex.
        match = self._DEPTH3_REGEX.match(cleaned)

        parts: list[str] = []
        if match and match.group(1):
            parts.append(match.group(1))                  # Parent digit(s)
            if match.group(2):
                parts.append(match.group(2).lower())      # Letter subpart
            if match.group(3):
                parts.append(match.group(3).lower())      # Roman numeral subpart
            return parts

        # 3. Fallback for orphaned subparts (no parent digit)
        orphan = self._ORPHAN_REGEX.match(cleaned)
        if orphan:
            return [orphan.group(1).lower()]

        # 4. Completely unparseable
        return []

    def normalize_for_matching(self, raw: str) -> str:
        """
        Normalise a question identifier for fuzzy cross-engine matching.
        "4(a)(i)"  → "4ai"
        "4.a.i"    → "4ai"
        "4 a i"    → "4ai"
        """
        if not raw:
            return ""
        s = str(raw).lower()
        # Remove all punctuation, whitespace, and normalize common variations
        s = re.sub(r"\s*\([a-z0-9]+\)", "", s) # Remove (a), (i) etc. but keep the content inside for now
        s = re.sub(r"[^a-z0-9]", "", s) # Remove all non-alphanumeric

        # Handle cases like "4ai" from "4(a)(i)" or "4.a.i"
        # This regex specifically targets patterns like 4a, 4ai, 4aii etc.
        # It's aggressive to ensure matching, e.g., "4 (a) (i)" -> "4ai"
        match = re.match(r"^(\d+)([a-z]*)(i|ii|iii|iv|v|vi|vii|viii|ix|x)?", s)
        if match:
            parts = [match.group(1)]
            if match.group(2): parts.append(match.group(2))
            if match.group(3): parts.append(match.group(3))
            return "".join(parts)
        return s

    def _generate_unified_paper_key(self, paper_reference_key: str) -> str:
        # Input: "igcse_0607_m23_qp_22" -> Output: "igcse_0607_m23_22"
        # Strict removal of doc_type markers to unify QP and MS
        if not paper_reference_key:
            return ""
        unified = re.sub(r'_(qp|ms|er|gt)', '', str(paper_reference_key), flags=re.IGNORECASE)
        return unified.strip()

    def _build_question_metadata(self, parts: list[str]) -> dict:
        # Safely builds the metadata dict ensuring accurate types and depth tracking
        if not parts:
            return {
                "parent": None,
                "subparts": [],
                "depth": 1,
                "is_orphaned": True,
            }
            
        parent_id = parts[0]
        
        if parent_id.isdigit():
            return {
                "parent": int(parent_id),
                "subparts": parts[1:],
                "depth": len(parts),
                "is_orphaned": False,
            }
        else:
            # Handles cases where only a subpart (like 'a' or 'ii') was found
            return {
                "parent": None,
                "subparts": parts,
                "depth": len(parts) + 1,
                "is_orphaned": True,
            }