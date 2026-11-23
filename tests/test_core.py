from finding_level_itv.candidates import build_candidate_pool
from finding_level_itv.metrics import (
    calculate_document_metrics,
    calculate_pooled_extraction_metrics,
)
from finding_level_itv.pipeline import run_itv
from finding_level_itv.text import normalise_sentence, repair_numbered_fragments


def test_normalization_removes_leading_numbering_and_newlines():
    assert normalise_sentence("  2.) Example\nFinding ") == "example finding"


def test_candidate_pool_retains_first_surface_form_and_per_generation_votes():
    pool = build_candidate_pool(
        [
            ["1. Finding alpha", "Finding alpha", "Finding beta"],
            ["finding alpha", "Finding gamma"],
        ]
    )
    assert [item.normalized for item in pool] == [
        "finding alpha",
        "finding beta",
        "finding gamma",
    ]
    assert pool[0].text == "1. Finding alpha"
    assert pool[0].occurrences == [1, 2]


def test_pipeline_verifies_unique_findings_independently():
    outputs = iter([["Finding alpha", "Finding beta"], ["Finding alpha"]])
    calls = []

    def generator(_report):
        return next(outputs)

    def verifier(candidate, report):
        calls.append((candidate, report))
        return candidate == "Finding alpha"

    result = run_itv("Synthetic report.", generator, verifier, num_generations=2)
    assert result.findings == ["Finding alpha"]
    assert result.report_label == 1
    assert len(calls) == 2


def test_finding_metrics_pool_counts_before_f1():
    result = calculate_pooled_extraction_metrics(
        [["a"], ["wrong"]],
        [["a", "b"], ["c"]],
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 2)
    assert result["f1"] == 0.4


def test_document_macro_f1_uses_any_output_and_both_classes():
    result = calculate_document_metrics([(1, 1), (0, 1)])
    assert result["fp"] == 1
    assert result["macro_f1"] == 1 / 3


def test_short_numbering_fragments_are_attached_to_next_sentence():
    assert repair_numbered_fragments(["1.", "First item.", "2.", "Second item."]) == [
        "1. First item.",
        "2. Second item.",
    ]

