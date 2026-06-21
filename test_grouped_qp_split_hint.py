from services.gemini_pdf_service import _annotate_grouped_qp_split_candidates
from services.gemini_pdf_service import _match_qp_orphan_label
from services.gemini_pdf_service import _orphan_tokens_to_qp_candidate_parts
from utils.question_normalizer import QuestionNumberNormalizer


def test_split_hint_for_grouped_sibling_row():
    normalizer = QuestionNumberNormalizer()
    rows = [
        {
            "question_id": "2(a)(ii)",
            "question_latex": (
                "2(a)(ii) The heights, h metres, of the boys are recorded. "
                "Write down the modal class. "
                "2(a)(ii) Calculate an estimate of the mean height. "
                "(b)(i) One boy is chosen at random."
            ),
            "validation_warnings": [],
            "needs_review": False,
        },
        {
            "question_id": "2(b)(i)",
            "question_latex": "2(b)(i) One boy is chosen at random.",
            "validation_warnings": [],
            "needs_review": False,
        },
    ]

    annotated = _annotate_grouped_qp_split_candidates(rows, ["2.a.i"], normalizer)

    warnings = annotated[0]["validation_warnings"]
    assert any("REPAIR_HINT split_grouped_row" in warning for warning in warnings)
    assert any("missing_id=2.a.i" in warning for warning in warnings)
    assert any("source_id=2.a.ii" in warning for warning in warnings)
    assert annotated[1]["validation_warnings"] == []


def test_split_hint_does_not_fire_when_missing_list_empty():
    normalizer = QuestionNumberNormalizer()
    rows = [
        {
            "question_id": "2(a)(i)",
            "question_latex": "2(a)(i) Write down the modal class.",
            "validation_warnings": [],
            "needs_review": False,
        },
        {
            "question_id": "2(a)(ii)",
            "question_latex": "2(a)(ii) Calculate an estimate of the mean height.",
            "validation_warnings": [],
            "needs_review": False,
        },
    ]

    annotated = _annotate_grouped_qp_split_candidates(rows, [], normalizer)

    assert annotated == rows


def test_multi_token_qp_orphan_label_maps_to_leaf_id():
    normalizer = QuestionNumberNormalizer()

    tokens, tail = _match_qp_orphan_label("(a) (i) Find an expression, in terms of m.")
    parts = _orphan_tokens_to_qp_candidate_parts(tokens, ["9"])

    assert tokens == ["a", "i"]
    assert normalizer.canonical_from_parts(parts) == "9.a.i"
    assert tail == "Find an expression, in terms of m."


def test_roman_qp_orphan_label_maps_under_current_parent():
    normalizer = QuestionNumberNormalizer()

    tokens, tail = _match_qp_orphan_label("(ii) Find the number of sacks.")
    parts = _orphan_tokens_to_qp_candidate_parts(tokens, ["9", "c", "i"])

    assert tokens == ["ii"]
    assert normalizer.canonical_from_parts(parts) == "9.c.ii"
    assert tail == "Find the number of sacks."
