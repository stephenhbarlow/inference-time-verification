"""Weakly supervised construction of contextual verifier examples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .sampling import (
    global_report_sentence_weights,
    probability_proportional_sample_without_replacement,
    probability_proportional_unique_group_sample,
)
from .text import normalise_sentence


SamplingStrategy = Literal["uniform", "gaussian", "central_window"]


@dataclass(frozen=True)
class AnnotatedReport:
    """A report, its ordered source sentences and existing finding annotations."""

    text: str
    sentences: tuple[str, ...]
    findings: tuple[str, ...]
    id: str = ""


@dataclass(frozen=True)
class VerifierExample:
    """One report--candidate pair used to adapt the contextual verifier."""

    document: str
    candidate: str
    label: int
    document_id: str = ""

    def as_record(self) -> dict[str, str | int]:
        """Return the public JSONL training-record representation."""
        return {
            "document_id": self.document_id,
            "document": self.document,
            "candidate": self.candidate,
            "label": self.label,
        }


@dataclass(frozen=True)
class NegativeConstructionConfig:
    """Settings used by the paper's global negative-construction analysis."""

    strategy: SamplingStrategy = "gaussian"
    positive_repetitions: int = 4
    negative_ratio: float = 1.5
    sigma_factor: float = 0.2
    central_window_fraction: float = 0.4
    report_prior_power: float = 0.0
    global_normalized_deduplication: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.positive_repetitions < 1:
            raise ValueError("positive_repetitions must be positive")
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio cannot be negative")
        if self.sigma_factor <= 0:
            raise ValueError("sigma_factor must be positive")
        if not 0 < self.central_window_fraction <= 1:
            raise ValueError("central_window_fraction must be in (0, 1]")


def annotated_sentence_indices(
    report_sentences: tuple[str, ...],
    annotated_findings: tuple[str, ...],
) -> set[int]:
    """Locate each annotation by exact match, then by substring containment."""
    available = set(range(len(report_sentences)))
    matched: set[int] = set()
    for finding in annotated_findings:
        selected = next(
            (
                index
                for index in sorted(available)
                if report_sentences[index] == finding
            ),
            None,
        )
        if selected is None:
            selected = next(
                (
                    index
                    for index in sorted(available)
                    if finding in report_sentences[index]
                ),
                None,
            )
        if selected is not None:
            matched.add(selected)
            available.remove(selected)
    return matched


def build_verifier_training_examples(
    reports: list[AnnotatedReport],
    config: NegativeConstructionConfig = NegativeConstructionConfig(),
) -> list[VerifierExample]:
    """Construct repeated positives and globally sampled unlabelled negatives.

    Every non-empty report initially receives mass proportional to its eligible
    sentence count raised to ``report_prior_power``. Position weights are then
    normalized within the report. The required negatives are drawn globally by a
    seeded exponential race, without fixed report quotas. By default, duplicate
    normalized sentence strings across reports form one sampling group.

    An unlabelled sentence is a weakly supervised negative. This construction does
    not assert that it has been independently judged clinically incorrect.
    """
    config.validate()
    if not reports:
        raise ValueError("reports cannot be empty")

    positives = [
        VerifierExample(report.text, finding, 1, report.id)
        for report in reports
        for finding in report.findings
        for _ in range(config.positive_repetitions)
    ]
    if not positives:
        raise ValueError("at least one annotated finding is required")

    candidate_indices_by_report: list[list[int]] = []
    for report in reports:
        annotated = annotated_sentence_indices(report.sentences, report.findings)
        candidate_indices_by_report.append(
            [
                index
                for index, sentence in enumerate(report.sentences)
                if index not in annotated and normalise_sentence(sentence)
            ]
        )

    weighted = global_report_sentence_weights(
        candidate_indices_by_report,
        [len(report.sentences) for report in reports],
        config.strategy,
        sigma_factor=config.sigma_factor,
        window_fraction=config.central_window_fraction,
        report_prior_power=config.report_prior_power,
    )
    target_negatives = int(config.negative_ratio * len(positives))
    weights = [weight for _, _, weight in weighted]
    group_ids = [
        normalise_sentence(reports[report_index].sentences[sentence_index])
        for report_index, sentence_index, _ in weighted
    ]
    if config.global_normalized_deduplication:
        selected = probability_proportional_unique_group_sample(
            weights,
            group_ids,
            target_negatives,
            seed=config.seed,
        )
    else:
        selected = probability_proportional_sample_without_replacement(
            weights,
            target_negatives,
            seed=config.seed,
        )

    negatives = []
    for weighted_index in selected:
        report_index, sentence_index, _ = weighted[weighted_index]
        report = reports[report_index]
        negatives.append(
            VerifierExample(
                report.text,
                report.sentences[sentence_index],
                0,
                report.id,
            )
        )

    examples = positives + negatives
    # Order is not part of the sampling distribution, but a deterministic shuffle
    # prevents class blocks in downstream training.
    random.Random(config.seed).shuffle(examples)
    return examples

