from finding_level_itv.dataset import (
    AnnotatedReport,
    NegativeConstructionConfig,
    annotated_sentence_indices,
    build_verifier_training_examples,
)
from finding_level_itv.text import normalise_sentence


def synthetic_reports():
    return [
        AnnotatedReport(
            text="Synthetic report one.",
            sentences=("Opening text.", "Finding alpha.", "Middle text.", "Ending text."),
            findings=("Finding alpha.",),
        ),
        AnnotatedReport(
            text="Synthetic report two.",
            sentences=("Opening text.", "Different middle.", "Another sentence.", "Ending two."),
            findings=(),
        ),
    ]


def test_alignment_prefers_exact_then_substring_match():
    indices = annotated_sentence_indices(
        ("prefix target suffix", "target"),
        ("target",),
    )
    assert indices == {1}


def test_global_builder_preserves_ratio_and_unique_normalized_negatives():
    examples = build_verifier_training_examples(
        synthetic_reports(),
        NegativeConstructionConfig(
            strategy="gaussian",
            positive_repetitions=2,
            negative_ratio=1.5,
            seed=42,
        ),
    )
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    assert len(positives) == 2
    assert len(negatives) == 3
    normalized = [normalise_sentence(example.candidate) for example in negatives]
    assert len(normalized) == len(set(normalized))
    assert "finding alpha." not in normalized


def test_fixed_window_builder_is_reproducible():
    config = NegativeConstructionConfig(
        strategy="central_window",
        positive_repetitions=1,
        negative_ratio=1.0,
        seed=42,
    )
    assert build_verifier_training_examples(synthetic_reports(), config) == (
        build_verifier_training_examples(synthetic_reports(), config)
    )

