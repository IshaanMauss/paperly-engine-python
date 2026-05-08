import re

class QuestionNumberNormalizer:
    def __init__(self):
        pass

    def normalize(self, raw_question_id: str, paper_reference_key: str) -> dict:
        """
        Normalizes raw question IDs (e.g., '1(b) a square number') into 
        canonical dots format ('1.b') for QP/MS linking.
        """
        parts = self._extract_parts(raw_question_id)
        
        # If parsing fails to find a number, we mark it as UNKNOWN to trigger a review
        if not parts:
            canonical_question_id = "UNKNOWN"
            parent_canonical_id = "UNKNOWN"
        else:
            canonical_question_id = ".".join(parts)  # e.g., "5.b.ii"
            parent_canonical_id = parts[0]           # e.g., "5"
            
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
        
        # 2. STRICT REGEX: Only capture the leading numbering pattern
        # Stops at the first space, backslash, or LaTeX character
        # Matches: "1", "1(a)", "3.b.ii", "22)"
        pattern = r'^(\d+(?:[\.\(\)][a-zA-Z0-9]+)*)'
        match = re.match(pattern, cleaned)
        
        if match:
            numbering = match.group(1)
            # Split the numbering part into clean tokens
            parts = [p.lower() for p in re.split(r'[\(\)\.]+', numbering) if p]
            return parts
        
        # 3. Fallback for standalone subparts like "(a)" or "b)"
        # Regex looks for a single letter or roman numeral at the start
        sub_match = re.match(r'^\s*[\(\[]?([a-z]|[ivx]+)[\)\]\.]', cleaned, re.IGNORECASE)
        if sub_match:
            return [sub_match.group(1).lower()]

        return []

    def _generate_unified_paper_key(self, paper_reference_key: str) -> str:
        # Input: "igcse_0607_m23_qp_22" -> Output: "igcse_0607_m23_22"
        # Strict removal of doc_type markers to unify QP and MS
        unified = re.sub(r'_(qp|ms|er|gt)', '', str(paper_reference_key), flags=re.IGNORECASE)
        return unified.strip()

    def _build_question_metadata(self, parts: list[str]) -> dict:
        if not parts:
            return {
                "parent": None,
                "subparts": [],
                "depth": 1,
                "is_orphaned": False,
            }
            
        parent_id = parts[0]
        subparts = parts[1:]
        
        return {
            "parent": int(parent_id) if parent_id.isdigit() else None,
            "subparts": subparts,
            "depth": len(parts),
            "is_orphaned": False,
        }