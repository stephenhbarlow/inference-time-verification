"""Finding-level pooled metrics and report-level fixed-class macro-F1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .text import normalise_sentence


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class FindingCounts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: "FindingCounts") -> "FindingCounts":
        return FindingCounts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
        )


def match_findings(predicted: Iterable[str], gold: Iterable[str]) -> FindingCounts:
    """One-to-one exact matching after the paper's normalization step."""
    remaining = Counter(normalise_sentence(item) for item in gold)
    tp = fp = 0
    for finding in predicted:
        normalized = normalise_sentence(finding)
        if remaining[normalized] > 0:
            tp += 1
            remaining[normalized] -= 1
        else:
            fp += 1
    return FindingCounts(tp=tp, fp=fp, fn=sum(remaining.values()))


def calculate_extraction_metrics(counts: FindingCounts) -> dict[str, int | float]:
    """Compute one finding score from dataset-total TP, FP and FN."""
    precision = _safe_divide(counts.tp, counts.tp + counts.fp)
    recall = _safe_divide(counts.tp, counts.tp + counts.fn)
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
    }


def calculate_pooled_extraction_metrics(
    predicted_by_report: Sequence[Iterable[str]],
    gold_by_report: Sequence[Iterable[str]],
) -> dict[str, int | float]:
    """Pool finding counts over reports before calculating precision/recall/F1."""
    if len(predicted_by_report) != len(gold_by_report):
        raise ValueError("predicted and gold report collections must align")
    total = FindingCounts(0, 0, 0)
    for predicted, gold in zip(predicted_by_report, gold_by_report):
        total += match_findings(predicted, gold)
    return calculate_extraction_metrics(total)


def calculate_document_metrics(
    true_and_predicted: Iterable[tuple[int, int]],
) -> dict[str, int | float]:
    """Calculate binary report metrics with both classes fixed in macro-F1."""
    pairs = list(true_and_predicted)
    if any(true not in (0, 1) or pred not in (0, 1) for true, pred in pairs):
        raise ValueError("document labels must be binary integers")
    tp = sum(true == 1 and pred == 1 for true, pred in pairs)
    fp = sum(true == 0 and pred == 1 for true, pred in pairs)
    fn = sum(true == 1 and pred == 0 for true, pred in pairs)
    tn = sum(true == 0 and pred == 0 for true, pred in pairs)

    positive_precision = _safe_divide(tp, tp + fp)
    positive_recall = _safe_divide(tp, tp + fn)
    positive_f1 = _safe_divide(
        2 * positive_precision * positive_recall,
        positive_precision + positive_recall,
    )
    negative_precision = _safe_divide(tn, tn + fn)
    negative_recall = _safe_divide(tn, tn + fp)
    negative_f1 = _safe_divide(
        2 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": _safe_divide(tp + tn, len(pairs)),
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2,
    }

