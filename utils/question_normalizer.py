import re

class QuestionNumberNormalizer:
    def __init__(self):
        pass

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
        # 1. Clean basic noise like leading 'Q'
        cleaned = re.sub(r'^[Qq]\s*', '', str(raw_id).strip())
        
        # 2. ULTIMATE STRICT REGEX: Captures exactly Parent, Subpart 1, Subpart 2
        # It strictly avoids trailing texts like " a square number" by explicitly 
        # looking for boundaries and restricting groups.
        regex = re.compile(
            r"^(\d+)"                                         # Group 1: Parent (e.g., 4)
            r"(?:"
                r"[\.\(\)]*"                                  # Optional separators like '(' or '.'
                r"\s*"
                r"([a-z])"                                    # Group 2: Single letter (e.g., a)
                r"[\.\(\)]*"                                  # Optional closing separators like ')'
                r"(?:"
                    r"[\.\(\)]*"
                    r"\s*"
                    r"(i{1,3}|iv|v|vi{1,3}|ix|x)"             # Group 3: Roman numeral (e.g., iii)
                    r"[\.\(\)]*"
                r")?"
            r")?",
            re.IGNORECASE
        )
        
        match = regex.match(cleaned)
        
        parts = []
        if match and match.group(1):
            parts.append(match.group(1))                         # Append Parent
            if match.group(2):
                parts.append(match.group(2).lower())             # Append 'a', 'b', etc.
            if match.group(3):
                parts.append(match.group(3).lower())             # Append 'i', 'ii', etc.
            return parts
        
        # 3. Fallback for orphaned subparts like "(a)" or "ii)"
        sub_match = re.match(r'^\s*[\(\[]?([a-z]|[ivx]+)[\)\]\.]', cleaned, re.IGNORECASE)
        if sub_match:
            return [sub_match.group(1).lower()]

        return []

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