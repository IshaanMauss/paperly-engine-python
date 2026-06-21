#!/usr/bin/env python3
"""Focused regressions for MS-authoritative QP numbering reconciliation."""

from services.ms_anchor_reconciler import reconcile_qp_against_ms
from utils.question_normalizer import QuestionNumberNormalizer


N = QuestionNumberNormalizer()


def q(question_id, text=None, warnings=None, review=False, diagrams=None):
    parts = N.extract_parts(question_id)
    return {
        "question_id": question_id,
        "canonical_question_id": N.canonical_from_parts(parts),
        "parent_canonical_id": N.parent_from_parts(parts),
        "question_latex": text or question_id,
        "validation_warnings": warnings or [],
        "needs_review": review,
        "diagram_urls": diagrams or [],
    }


def ids(rows):
    return [row.get("canonical_question_id") for row in rows]


def test_grouped_ms_parent_repairs_5_roman_family():
    rows = reconcile_qp_against_ms(
        [
            q("5(a)", "5(a) Some first part"),
            q("5(i)", "5(i) Describe transformation"),
            q("5(i)(a)", "5(i)(a) Draw image"),
            q("5(i)(b)", "5(i)(b) Draw image"),
            q("5(ii)(a)", "5(ii)(a) Draw image"),
            q("5(ii)(b)", "5(ii)(b) Draw image"),
        ],
        ["5.a", "5.b"],
        N,
    )
    assert ids(rows) == ["5.a", "5.b.i", "5.b.i.a", "5.b.i.b", "5.b.ii.a", "5.b.ii.b"]
    assert rows[2]["question_latex"].startswith("5(b)(i)(a)")
    assert all(not row.get("validation_warnings") for row in rows)


def test_implied_ms_parent_repairs_7_roman_family():
    rows = reconcile_qp_against_ms(
        [
            q("7(i)", "7(i) Describe fully. (ii) Draw images."),
            q("7(i)(a)", "7(i)(a) triangle A onto B"),
            q("7(ii)(a)", "7(ii)(a) Draw reflection"),
            q("7(ii)(b)", "7(ii)(b) Draw enlargement"),
        ],
        ["7.a.i", "7.a.ii", "7.a.ii.a", "7.a.ii.b"],
        N,
    )
    assert ids(rows) == ["7.a.i", "7.a.i.a", "7.a.ii.a", "7.a.ii.b"]
    assert rows[0]["question_latex"].startswith("7(a)(i)")


def test_root_with_visible_marker_promotes_to_ms_child():
    rows = reconcile_qp_against_ms(
        [q("7", "7 Information table. (a) Calculate an estimate.")],
        ["7.a", "7.c.i"],
        N,
    )
    assert ids(rows) == ["7.a"]
    assert rows[0]["question_latex"].startswith("7(a)")


def test_pure_parent_stem_merges_into_first_child_and_carries_diagram():
    rows = reconcile_qp_against_ms(
        [
            q("7", "7 Information about masses.", diagrams=["diagram-1"]),
            q("7(a)", "7(a) Calculate the mean."),
        ],
        ["7.a"],
        N,
    )
    assert ids(rows) == ["7.a"]
    assert "Information about masses" in rows[0]["question_latex"]
    assert rows[0]["diagram_urls"] == ["diagram-1"]


def test_ambiguous_missing_parent_does_not_guess():
    rows = reconcile_qp_against_ms(
        [q("5(i)", "5(i) Ambiguous")],
        ["5.a.i", "5.b.i"],
        N,
    )
    assert ids(rows) == ["5.i"]


def test_no_duplicate_created_when_target_taken():
    rows = reconcile_qp_against_ms(
        [q("5(b)(i)", "5(b)(i) Existing"), q("5(i)", "5(i) Would collide")],
        ["5.b.i", "5.b.ii"],
        N,
    )
    assert ids(rows) == ["5.b.i", "5.i"]
    assert len(ids(rows)) == len(set(ids(rows)))


def test_ms_confirmed_numbering_warning_is_suppressed():
    rows = reconcile_qp_against_ms(
        [
            q(
                "5(i)",
                "5(i) Text",
                warnings=["QP embedded subpart guard corrected 5.a to 5.i."],
                review=True,
            )
        ],
        ["5.b.i"],
        N,
    )
    assert ids(rows) == ["5.b.i"]
    assert rows[0]["validation_warnings"] == []
    assert rows[0]["needs_review"] is False


if __name__ == "__main__":
    test_grouped_ms_parent_repairs_5_roman_family()
    test_implied_ms_parent_repairs_7_roman_family()
    test_root_with_visible_marker_promotes_to_ms_child()
    test_pure_parent_stem_merges_into_first_child_and_carries_diagram()
    test_ambiguous_missing_parent_does_not_guess()
    test_no_duplicate_created_when_target_taken()
    test_ms_confirmed_numbering_warning_is_suppressed()
    print("ms anchor reconciler tests passed")
