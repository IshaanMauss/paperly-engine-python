"""
MS-authoritative QP numbering reconciler.

This module is deliberately local and deterministic. It does not call Gemini or
any external service. It uses the saved Marking Scheme canonical IDs as the
numbering authority for Question Paper rows after initial extraction.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from utils.question_normalizer import QuestionNumberNormalizer

logger = logging.getLogger(__name__)


def _is_single_letter_part(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 1 and "a" <= text <= "z"


def _is_roman_part(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in QuestionNumberNormalizer._LABEL_ROMAN_SET


def _raw_parts(raw: dict, normalizer: QuestionNumberNormalizer) -> list[str]:
    label = (
        normalizer.extract_leading_label(raw.get("question_id", ""))
        or normalizer.extract_leading_label(raw.get("question_latex", ""))
        or str(raw.get("canonical_question_id") or "")
    )
    return normalizer.extract_parts(label)


def _canonical(raw: dict, normalizer: QuestionNumberNormalizer) -> str:
    parts = _raw_parts(raw, normalizer)
    return normalizer.canonical_from_parts(parts) if parts else ""


class _MSTree:
    """Index exact and implied MS hierarchy without treating root-only IDs as safe."""

    def __init__(self, expected_ids: list[str], normalizer: QuestionNumberNormalizer):
        self.normalizer = normalizer
        self.exact_ids: set[str] = set()
        self.exact_parts: list[list[str]] = []

        for value in expected_ids or []:
            parts = normalizer.extract_parts(str(value or ""))
            canonical = normalizer.canonical_from_parts(parts) if parts else ""
            if canonical and canonical not in self.exact_ids:
                self.exact_ids.add(canonical)
                self.exact_parts.append([str(part).lower() for part in parts])

        self.implied_ids: set[str] = set()
        for parts in self.exact_parts:
            # Do not add root-only implied IDs. A standalone QP root row should
            # be handled by the root-row repair, not hidden as hierarchy-covered.
            for depth in range(2, len(parts)):
                prefix = normalizer.canonical_from_parts(parts[:depth])
                if prefix and prefix not in self.exact_ids:
                    self.implied_ids.add(prefix)

        self.all_known = self.exact_ids | self.implied_ids
        self.roots_with_children = {
            parts[0] for parts in self.exact_parts if len(parts) >= 2
        }

        self.direct_letter_parents_by_root: dict[str, list[list[str]]] = defaultdict(list)
        seen_direct: set[str] = set()
        for parts in self.exact_parts:
            if len(parts) >= 2 and _is_single_letter_part(parts[1]):
                direct = parts[:2]
                direct_canonical = normalizer.canonical_from_parts(direct)
                if direct_canonical and direct_canonical not in seen_direct:
                    self.direct_letter_parents_by_root[parts[0]].append(direct)
                    seen_direct.add(direct_canonical)

    def is_hierarchy_covered(self, canonical: str) -> bool:
        canonical = str(canonical or "").strip().lower()
        if not canonical:
            return False
        if canonical in self.all_known:
            return True
        parts = canonical.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            if ".".join(parts[:depth]) in self.all_known:
                return True
        return False


def _set_parts(
    raw: dict,
    new_parts: list[str],
    normalizer: QuestionNumberNormalizer,
    *,
    old_canonical: str,
    reason: str,
    source: str,
    suppress_warning: bool = True,
) -> dict:
    updated = dict(raw)
    new_label = normalizer.format_parts(new_parts)
    new_canonical = normalizer.canonical_from_parts(new_parts)
    if not new_label or not new_canonical:
        return updated

    updated["question_id"] = new_label
    updated["canonical_question_id"] = new_canonical
    updated["parent_canonical_id"] = normalizer.parent_from_parts(new_parts)

    q_latex = str(updated.get("question_latex") or "")
    _label, remainder = normalizer.split_label_and_remainder(q_latex)
    if remainder or _label:
        updated["question_latex"] = f"{new_label} {remainder}".strip() if remainder else new_label
    elif q_latex.strip():
        updated["question_latex"] = f"{new_label} {q_latex}".strip()
    else:
        updated["question_latex"] = new_label

    if suppress_warning:
        warnings = updated.get("validation_warnings")
        if isinstance(warnings, list):
            stale_prefixes = (
                "QP embedded subpart guard promoted ",
                "QP embedded subpart guard corrected ",
            )
            kept = [warning for warning in warnings if not str(warning).startswith(stale_prefixes)]
            updated["validation_warnings"] = kept
            if not kept:
                updated["needs_review"] = False

    logger.info(
        "[MSReconciler] %r -> %r | reason=%s | source=%s | warning_suppressed=%s",
        old_canonical,
        new_canonical,
        reason,
        source,
        suppress_warning,
    )
    print(
        f"[MSReconciler] old_id={old_canonical!r} new_id={new_canonical!r} "
        f"reason={reason} source={source} warning_suppressed={suppress_warning}"
    )
    return updated


def _question_has_visible_marker(
    raw: dict,
    marker: str,
    normalizer: QuestionNumberNormalizer,
) -> bool:
    marker = str(marker or "").strip().lower()
    if not marker:
        return False
    _label, remainder = normalizer.split_label_and_remainder(str(raw.get("question_latex") or ""))
    return bool(re.search(rf"\(\s*{re.escape(marker)}\s*\)", remainder[:900], flags=re.IGNORECASE))


def _repair_root_only_rows(
    questions_raw: list[dict],
    ms_tree: _MSTree,
    normalizer: QuestionNumberNormalizer,
) -> list[dict]:
    parts_by_index = [
        _raw_parts(raw, normalizer) if isinstance(raw, dict) else []
        for raw in questions_raw
    ]
    original_ids = {
        normalizer.canonical_from_parts(parts)
        for parts in parts_by_index
        if parts
    }

    children_by_root: dict[str, list[int]] = defaultdict(list)
    for idx, parts in enumerate(parts_by_index):
        if len(parts) >= 2 and parts[0].isdigit():
            children_by_root[parts[0]].append(idx)

    promoted: list[dict] = []
    stem_by_root: dict[str, dict] = {}
    promoted_count = 0

    for idx, (raw, parts) in enumerate(zip(questions_raw, parts_by_index)):
        if not isinstance(raw, dict) or len(parts) != 1 or not str(parts[0]).isdigit():
            promoted.append(raw)
            continue

        root = parts[0]
        canonical = normalizer.canonical_from_parts(parts)
        if canonical in ms_tree.exact_ids or root not in ms_tree.roots_with_children:
            promoted.append(raw)
            continue

        direct_candidates = []
        for candidate in ms_tree.direct_letter_parents_by_root.get(root, []):
            candidate_canonical = normalizer.canonical_from_parts(candidate)
            if candidate_canonical in original_ids:
                continue
            if _question_has_visible_marker(raw, candidate[1], normalizer):
                direct_candidates.append(candidate)

        if len(direct_candidates) == 1:
            promoted.append(
                _set_parts(
                    raw,
                    direct_candidates[0],
                    normalizer,
                    old_canonical=canonical,
                    reason="root promoted via visible marker",
                    source="saved MS child + visible QP marker",
                    suppress_warning=ms_tree.is_hierarchy_covered(
                        normalizer.canonical_from_parts(direct_candidates[0])
                    ),
                )
            )
            promoted_count += 1
            continue

        child_indexes = [child_idx for child_idx in children_by_root.get(root, []) if child_idx != idx]
        if child_indexes:
            _label, stem_remainder = normalizer.split_label_and_remainder(str(raw.get("question_latex") or ""))
            if stem_remainder.strip():
                stem_by_root[root] = raw
                print(
                    f"[MSReconciler] old_id={canonical!r} new_id=merged-into-children "
                    "reason=shared parent stem source=saved MS has children warning_suppressed=False"
                )
            continue

        promoted.append(raw)

    if promoted_count:
        print(f"[MSReconciler] root_promotions={promoted_count}")

    if not stem_by_root:
        return promoted

    merged: list[dict] = []
    for raw in promoted:
        if not isinstance(raw, dict):
            merged.append(raw)
            continue

        parts = _raw_parts(raw, normalizer)
        if len(parts) < 2 or parts[0] not in stem_by_root:
            merged.append(raw)
            continue

        parent_raw = stem_by_root.pop(parts[0])
        _parent_label, parent_remainder = normalizer.split_label_and_remainder(
            str(parent_raw.get("question_latex") or "")
        )
        child_label, child_remainder = normalizer.split_label_and_remainder(
            str(raw.get("question_latex") or "")
        )
        updated = dict(raw)
        parent_norm = re.sub(r"\s+", " ", parent_remainder.lower()).strip()
        child_norm = re.sub(r"\s+", " ", child_remainder.lower()).strip()
        if parent_norm and parent_norm[:80] not in child_norm:
            updated["question_latex"] = f"{child_label} {parent_remainder} {child_remainder}".strip()

        parent_diagrams = parent_raw.get("diagram_urls") if isinstance(parent_raw.get("diagram_urls"), list) else []
        child_diagrams = updated.get("diagram_urls") if isinstance(updated.get("diagram_urls"), list) else []
        if parent_diagrams:
            seen = {str(url) for url in child_diagrams}
            updated["diagram_urls"] = child_diagrams + [
                url for url in parent_diagrams if str(url) not in seen
            ]
        merged.append(updated)

    return merged


def _repair_missing_letter_parent(
    questions_raw: list[dict],
    ms_tree: _MSTree,
    normalizer: QuestionNumberNormalizer,
) -> list[dict]:
    original_ids = {
        _canonical(raw, normalizer)
        for raw in questions_raw
        if isinstance(raw, dict)
    }
    original_ids.discard("")

    repaired: list[dict] = []
    used: set[str] = set()
    last_letter_parent_by_root: dict[str, list[str]] = {}

    for raw in questions_raw:
        if not isinstance(raw, dict):
            repaired.append(raw)
            continue

        parts = _raw_parts(raw, normalizer)
        canonical = normalizer.canonical_from_parts(parts) if parts else ""
        if parts and len(parts) >= 2 and _is_single_letter_part(parts[1]):
            last_letter_parent_by_root[parts[0]] = parts[:2]

        if (
            not canonical
            or canonical in ms_tree.all_known
            or len(parts) < 2
            or not _is_roman_part(parts[1])
        ):
            repaired.append(raw)
            if canonical:
                used.add(canonical)
            continue

        root = parts[0]
        tail = [str(part).lower() for part in parts[1:]]
        exact_candidates = [
            candidate
            for candidate in ms_tree.exact_parts
            if (
                len(candidate) == len(parts) + 1
                and candidate[0] == root
                and _is_single_letter_part(candidate[1])
                and candidate[2:] == tail
            )
        ]

        corrected_parts: list[str] = []
        source = ""
        if len(exact_candidates) == 1:
            corrected_parts = exact_candidates[0]
            source = "exact saved-MS leaf"
        else:
            direct_candidates = ms_tree.direct_letter_parents_by_root.get(root, [])
            missing_direct = [
                candidate
                for candidate in direct_candidates
                if normalizer.canonical_from_parts(candidate) not in original_ids
            ]
            last = last_letter_parent_by_root.get(root)
            if len(missing_direct) == 1:
                corrected_parts = missing_direct[0] + tail
                source = "only missing saved-MS parent"
            elif last and any(candidate == last for candidate in direct_candidates):
                corrected_parts = last + tail
                source = "current saved-MS parent context"
            elif len(direct_candidates) == 1:
                corrected_parts = direct_candidates[0] + tail
                source = "unique saved-MS parent"

        if not corrected_parts:
            repaired.append(raw)
            if canonical:
                used.add(canonical)
            continue

        new_canonical = normalizer.canonical_from_parts(corrected_parts)
        if not new_canonical or new_canonical in used:
            repaired.append(raw)
            if canonical:
                used.add(canonical)
            continue

        repaired.append(
            _set_parts(
                raw,
                corrected_parts,
                normalizer,
                old_canonical=canonical,
                reason="inserted missing letter-level parent",
                source=source,
                suppress_warning=ms_tree.is_hierarchy_covered(new_canonical),
            )
        )
        used.add(new_canonical)

    return repaired


def _suppress_confirmed_warnings(
    questions_raw: list[dict],
    ms_tree: _MSTree,
    normalizer: QuestionNumberNormalizer,
) -> list[dict]:
    stale_prefixes = (
        "QP embedded subpart guard promoted ",
        "QP embedded subpart guard corrected ",
    )
    out: list[dict] = []
    suppressed = 0
    for raw in questions_raw:
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        canonical = _canonical(raw, normalizer)
        warnings = raw.get("validation_warnings")
        if not canonical or not ms_tree.is_hierarchy_covered(canonical) or not isinstance(warnings, list):
            out.append(raw)
            continue
        kept = [warning for warning in warnings if not str(warning).startswith(stale_prefixes)]
        if len(kept) == len(warnings):
            out.append(raw)
            continue
        updated = dict(raw)
        updated["validation_warnings"] = kept
        if not kept:
            updated["needs_review"] = False
        out.append(updated)
        suppressed += len(warnings) - len(kept)

    if suppressed:
        print(f"[MSReconciler] suppressed_stale_numbering_warnings={suppressed}")
    return out


def reconcile_qp_against_ms(
    questions_raw: list[dict],
    expected_canonical_ids: list[str],
    question_normalizer: QuestionNumberNormalizer,
) -> list[dict]:
    """Repair QP numbering against saved MS IDs where deterministic."""
    if not isinstance(questions_raw, list) or not expected_canonical_ids:
        return questions_raw

    ms_tree = _MSTree(expected_canonical_ids, question_normalizer)
    if not ms_tree.exact_ids:
        return questions_raw

    print(
        "[MSReconciler] starting "
        f"qp_rows={len(questions_raw)} ms_exact={len(ms_tree.exact_ids)} "
        f"ms_implied={len(ms_tree.implied_ids)}"
    )
    questions_raw = _repair_root_only_rows(questions_raw, ms_tree, question_normalizer)
    questions_raw = _repair_missing_letter_parent(questions_raw, ms_tree, question_normalizer)
    questions_raw = _suppress_confirmed_warnings(questions_raw, ms_tree, question_normalizer)
    return questions_raw


__all__ = ["reconcile_qp_against_ms"]
