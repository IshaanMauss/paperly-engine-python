#!/usr/bin/env python3
"""Focused regressions for structural question numbering guards."""

from services.gemini_pdf_service import _normalize_response
from services.gemini_slicer import (
    _apply_sequential_duplicate_guard,
    _build_extracted_question,
    _fix_orphan_question_id,
)
from utils.question_normalizer import QuestionNumberNormalizer


def test_normalizer_preserves_inline_math_payload():
    normalizer = QuestionNumberNormalizer()
    label, remainder = normalizer.split_label_and_remainder(
        "10 (a) f(x) = 2x - 3"
    )
    assert label == "10(a)"
    assert remainder == "f(x) = 2x - 3"
    assert normalizer.split_label_and_remainder("8 A solid cuboid")[1] == "A solid cuboid"
    assert normalizer.extract_parts("9(d) (ii)") == ["9", "d", "ii"]
    assert normalizer.normalize_for_matching("4(a)(i)") == "4ai"


def test_slicer_repairs_duplicate_nested_sibling():
    meta = {"paper_reference_key": "igcse_0580_s23_qp_22"}
    items = []
    for raw in [
        {"question_latex": "5(c)(i) The diagram shows a curve. (i) Find x."},
        {"question_latex": "5(c)(i) The diagram shows a curve. (ii) Find y."},
    ]:
        items.append(
            {
                "model": _build_extracted_question(raw, "Question Paper", meta),
                "diagram_regions": [],
                "page_num": 0,
            }
        )

    used = set()
    previous = None
    for item in items:
        _apply_sequential_duplicate_guard(item, previous, used, 0)
        previous = item["model"]

    assert items[0]["model"].canonical_question_id == "5.c.i"
    assert items[1]["model"].canonical_question_id == "5.c.ii"
    assert items[1]["model"].question_latex == "5(c)(ii) Find y."


def test_orphan_roman_uses_full_parent_context():
    meta = {"paper_reference_key": "igcse_0580_s23_qp_22"}
    orphan = _build_extracted_question(
        {"question_latex": "(iii) Find z.", "question_id": "(iii)"},
        "Question Paper",
        meta,
    )
    fixed = _fix_orphan_question_id(orphan, "5(c)(ii)", 1)
    assert fixed.question_id == "5(c)(iii)"
    assert fixed.question_latex == "5(c)(iii) Find z."


def test_marking_scheme_group_expands_to_leaf_ids():
    response = _normalize_response(
        {
            "metadata": {
                "paper_reference_key": "igcse_0580_s23_ms_22",
                "curriculum": "IGCSE",
            },
            "questions_array": [
                {
                    "document_type": "Marking Scheme",
                    "question_id": "8(a)",
                    "question_latex": "8(a)",
                    "official_marking_scheme_latex": (
                        "(i) x = 2 [1]\n(ii) y = 3 [1]\n(iii) z = 4 [1]"
                    ),
                    "method_steps": [],
                }
            ],
        },
        "0580_s23_ms_22.pdf",
        "Marking Scheme",
        "IGCSE",
        "igcse_0580_s23_ms_22",
    )
    assert [q.canonical_question_id for q in response.questions_array] == [
        "8.a.i",
        "8.a.ii",
        "8.a.iii",
    ]


def test_qp_backward_root_intrusion_repaired_to_next_sibling():
    response = _normalize_response(
        {
            "metadata": {
                "paper_reference_key": "igcse_0580_s21_qp_43",
                "curriculum": "IGCSE",
            },
            "questions_array": [
                {"document_type": "Question Paper", "question_id": "7(a)", "question_latex": "7(a) ok"},
                {"document_type": "Question Paper", "question_id": "7(b)(i)", "question_latex": "(b) (i) ok"},
                {"document_type": "Question Paper", "question_id": "7(b)(ii)", "question_latex": "(ii) ok"},
                {"document_type": "Question Paper", "question_id": "4(b)(v)", "question_latex": "4(b)(v) bad"},
                {"document_type": "Question Paper", "question_id": "4(b)(vi)", "question_latex": "4(b)(vi) bad2"},
                {"document_type": "Question Paper", "question_id": "8(a)(i)", "question_latex": "8 A solid cuboid (a) (i) Calculate"},
            ],
        },
        "0580_s21_qp_43.pdf",
        "Question Paper",
        "IGCSE",
        "igcse_0580_s21_qp_43",
    )
    assert [q.canonical_question_id for q in response.questions_array] == [
        "7.a",
        "7.b.i",
        "7.b.ii",
        "4.b.v",
        "4.b.vi",
        "8.a.i",
    ]
    assert response.questions_array[1].question_latex == "7(b)(i) ok"
    assert "A solid cuboid" in response.questions_array[5].question_latex


if __name__ == "__main__":
    test_normalizer_preserves_inline_math_payload()
    test_slicer_repairs_duplicate_nested_sibling()
    test_orphan_roman_uses_full_parent_context()
    test_marking_scheme_group_expands_to_leaf_ids()
    test_qp_backward_root_intrusion_repaired_to_next_sibling()
    print("structural numbering guard tests passed")
