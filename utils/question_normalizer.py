import re
from typing import Optional


class QuestionNumberNormalizer:
    """
    Parse and format exam question identifiers without mutating the question text.

    The important production rule is that a question label and the question body
    are separate concepts. For example, in "10 (a) f(x) = 2x - 3", the label is
    "10(a)" and the payload begins with "f(x) = 2x - 3". The parser must never
    consume the "f(x)" token as a subpart.
    """

    _ROMAN_ORDER = [
        "i", "ii", "iii", "iv", "v",
        "vi", "vii", "viii", "ix", "x",
        "xi", "xii", "xiii", "xiv", "xv",
        "xvi", "xvii", "xviii", "xix", "xx",
    ]
    _ROMAN_SET = set(_ROMAN_ORDER)
    # In maths papers, "(x)" is far more often a function argument than a
    # tenth-level roman subpart. Treat it as payload, not hierarchy.
    _LABEL_ROMAN_SET = _ROMAN_SET - {"x"}
    _TOKEN_ALT = r"(?:xviii|xvii|xvi|xiv|xiii|xii|xi|viii|vii|vi|iv|ix|iii|ii|i|xx|xix|xv|x|v|[a-z])"

    def __init__(self):
        self._ROOT_REGEX = re.compile(r"^\s*[Qq]?\s*(\d+)")
        self._WRAPPED_TOKEN_REGEX = re.compile(
            rf"\s*[\(\[]\s*({self._TOKEN_ALT})\s*[\)\]]",
            re.IGNORECASE,
        )
        self._DOTTED_TOKEN_REGEX = re.compile(
            rf"\s*[\.\-]\s*({self._TOKEN_ALT})(?=$|[^a-zA-Z])",
            re.IGNORECASE,
        )
        self._BARE_TOKEN_REGEX = re.compile(
            rf"\s+({self._TOKEN_ALT})(?=$|[^a-zA-Z])",
            re.IGNORECASE,
        )

        self._ORPHAN_REGEX = re.compile(
            rf"^\s*[\(\[]?\s*({self._TOKEN_ALT})\s*[\)\]\.]",
            re.IGNORECASE,
        )

    def normalize(self, raw_question_id: str, paper_reference_key: str) -> dict:
        """
        Normalize raw question IDs such as "1(b)" or "4(a)(iii)" into dot
        format such as "1.b" or "4.a.iii" for QP/MS linking.
        """
        parts = self._extract_parts(raw_question_id)

        if not parts:
            canonical_question_id = "UNKNOWN"
            parent_canonical_id = "UNKNOWN"
        else:
            canonical_question_id = self.canonical_from_parts(parts)
            parent_canonical_id = self.parent_from_parts(parts)

        unified_paper_key = self._generate_unified_paper_key(paper_reference_key)
        metadata = self.build_question_metadata(parts)

        return {
            "canonical_question_id": canonical_question_id,
            "parent_canonical_id": parent_canonical_id,
            "unified_paper_key": unified_paper_key,
            "question_number_metadata": metadata,
        }

    def extract_parts(self, raw_id: str) -> list[str]:
        """Public wrapper used by slicer/orchestration guards."""
        return self._extract_parts(raw_id)

    def split_label_and_remainder(self, raw_text: str) -> tuple[str, str]:
        """
        Return (formatted_label, untouched_remainder) from the leading label.

        The remainder is sliced only at the end of the parsed visual label. This
        is what preserves inline math like "f(x)" after "10(a)".
        """
        parts, span_end = self._extract_parts_with_span(raw_text)
        if not parts or span_end <= 0:
            return "", str(raw_text or "")
        return self.format_parts(parts), str(raw_text or "")[span_end:].lstrip()

    def extract_leading_label(self, raw_text: str) -> str:
        parts, _ = self._extract_parts_with_span(raw_text)
        return self.format_parts(parts) if parts else ""

    def canonicalize(self, raw_id: str) -> str:
        parts = self._extract_parts(raw_id)
        return self.canonical_from_parts(parts) if parts else ""

    def canonical_from_parts(self, parts: list[str]) -> str:
        return ".".join(p.lower() for p in parts if p)

    def parent_from_parts(self, parts: list[str]) -> str:
        if not parts:
            return "UNKNOWN"
        # Root parent is preserved for compatibility with existing sequence checks.
        return parts[0]

    def immediate_parent_from_parts(self, parts: list[str]) -> str:
        if len(parts) <= 1:
            return parts[0] if parts else "UNKNOWN"
        return self.canonical_from_parts(parts[:-1])

    def build_question_metadata(self, parts: list[str]) -> dict:
        return self._build_question_metadata(parts)

    def format_parts(self, parts: list[str]) -> str:
        if not parts:
            return ""
        root = str(parts[0]).strip()
        if not root:
            return ""
        suffix = "".join(f"({str(part).strip().lower()})" for part in parts[1:] if part)
        return f"{root}{suffix}"

    def increment_terminal_part(self, parts: list[str]) -> Optional[list[str]]:
        """
        Increment the final hierarchy token structurally.

        Examples:
            ["5", "c", "i"] -> ["5", "c", "ii"]
            ["5", "c"]      -> ["5", "d"]

        Root-only IDs are intentionally not incremented because turning a
        duplicate "10" into "11" is more likely to hide a real extraction split.
        """
        if len(parts) < 2:
            return None

        updated = [str(p).lower() for p in parts]
        terminal = updated[-1]

        if terminal in self._ROMAN_SET:
            idx = self._ROMAN_ORDER.index(terminal)
            if idx + 1 < len(self._ROMAN_ORDER):
                updated[-1] = self._ROMAN_ORDER[idx + 1]
                return updated
            return None

        if len(terminal) == 1 and "a" <= terminal <= "y":
            updated[-1] = chr(ord(terminal) + 1)
            return updated

        if terminal.isdigit():
            updated[-1] = str(int(terminal) + 1)
            return updated

        return None

    def normalize_for_matching(self, raw: str) -> str:
        """
        Normalize a question identifier for fuzzy cross-engine matching.

        "4(a)(i)" -> "4ai"
        "4.a.i"   -> "4ai"
        "4 a i"   -> "4ai"
        """
        if not raw:
            return ""
        parts = self._extract_parts(raw)
        if parts:
            return "".join(parts).lower()
        return re.sub(r"[^a-z0-9]", "", str(raw).lower())

    def _extract_parts(self, raw_id: str) -> list[str]:
        parts, _ = self._extract_parts_with_span(raw_id)
        return parts

    def _extract_parts_with_span(self, raw_id: str) -> tuple[list[str], int]:
        text = str(raw_id or "")
        if not text.strip():
            return [], 0

        root_match = self._ROOT_REGEX.match(text)
        if not root_match:
            orphan = self._ORPHAN_REGEX.match(text)
            if orphan:
                return [orphan.group(1).lower()], orphan.end()
            return [], 0

        parts: list[str] = [root_match.group(1)]
        pos = root_match.end()

        # Parse several visible subpart levels. Cambridge MS/QP labels can be
        # deeper than root + letter + roman, e.g. "1(a)(iv)(a)". Earlier we
        # capped this at two subparts, which silently truncated valid MS labels
        # and caused QP/MS parity noise. The token consumer still guards against
        # math payload such as f(x), so widening this is safe for normal text.
        for depth in range(5):
            consumed = self._consume_subpart_token(text, pos, depth)
            if not consumed:
                break
            token, pos = consumed
            parts.append(token.lower())

        return parts, pos

    def _consume_subpart_token(
        self,
        text: str,
        pos: int,
        depth: int,
    ) -> Optional[tuple[str, int]]:
        wrapped = self._WRAPPED_TOKEN_REGEX.match(text, pos)
        if wrapped:
            token = wrapped.group(1).lower()
            if depth >= 1 and token == "x":
                return None
            return token, wrapped.end()

        dotted = self._DOTTED_TOKEN_REGEX.match(text, pos)
        if dotted:
            return dotted.group(1), dotted.end()

        bare = self._BARE_TOKEN_REGEX.match(text, pos)
        if not bare:
            return None

        token = bare.group(1).lower()
        end = bare.end()
        next_non_space = self._next_non_space_char(text, end)

        # Bare "f(" after a root/subpart is almost always a function token,
        # not a question label: "10(a) f(x) = 2x - 3".
        if next_non_space == "(":
            return None

        # Bare tokens followed by operators are usually math payload, not labels.
        if next_non_space in {"=", "+", "-", "*", "/", "^"}:
            return None

        if depth == 0:
            # First bare subpart should be a letter, not a bare roman variable.
            if token in {"i", "v", "x"}:
                return None
            if next_non_space.isalpha():
                return None
            if len(token) != 1 or not token.isalpha():
                return None
        else:
            # Third-level bare labels should be roman. This prevents words or
            # variables after "(a)" from being consumed.
            if token not in self._LABEL_ROMAN_SET:
                return None

        return token, end

    @staticmethod
    def _next_non_space_char(text: str, pos: int) -> str:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return text[pos] if pos < len(text) else ""

    def _generate_unified_paper_key(self, paper_reference_key: str) -> str:
        if not paper_reference_key:
            return ""
        unified = re.sub(r'_(qp|ms|er|gt)', '', str(paper_reference_key), flags=re.IGNORECASE)
        return unified.strip()

    def _build_question_metadata(self, parts: list[str]) -> dict:
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

        return {
            "parent": None,
            "subparts": parts,
            "depth": len(parts) + 1,
            "is_orphaned": True,
        }
