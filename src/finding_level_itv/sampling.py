"""Pure sentence-position helpers for verifier negative construction.

These functions contain no clinical-data or model dependencies, making the fixed
central-window definition straightforward to test before constructing a dataset.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def normalized_sentence_position(sentence_index: int, sentence_count: int) -> float:
    """Map a zero-based sentence index onto [0, 1] within its own report.

    The first and last sentences of a multi-sentence report map to 0 and 1.
    A one-sentence report has no meaningful endpoints, so its sole sentence maps
    to the centre (0.5).
    """
    if sentence_count < 1:
        raise ValueError("sentence_count must be positive")
    if not 0 <= sentence_index < sentence_count:
        raise ValueError("sentence_index is outside the report")
    if sentence_count == 1:
        return 0.5
    return sentence_index / (sentence_count - 1)


def central_window_indices(sentence_count: int, window_fraction: float) -> list[int]:
    """Return sentences inside a normalized window centred at 0.5.

    For a 0.4 window, the inclusive bounds are [0.3, 0.7], regardless of report
    length. Very short reports can have no discrete sentence inside those bounds
    (for example, a two-sentence report). In that case all sentences tied for the
    closest normalized position to 0.5 are eligible. This deterministic fallback
    avoids silently dropping short reports or biasing a tie toward the first one.
    """
    if sentence_count < 1:
        raise ValueError("sentence_count must be positive")
    if not 0 < window_fraction <= 1:
        raise ValueError("window_fraction must be in (0, 1]")
    lower = 0.5 - window_fraction / 2
    upper = 0.5 + window_fraction / 2
    positions = [
        normalized_sentence_position(index, sentence_count)
        for index in range(sentence_count)
    ]
    eligible = [
        index for index, position in enumerate(positions) if lower <= position <= upper
    ]
    if eligible:
        return eligible
    minimum_distance = min(abs(position - 0.5) for position in positions)
    return [
        index
        for index, position in enumerate(positions)
        if abs(abs(position - 0.5) - minimum_distance) < 1e-12
    ]


def central_window_indices_excluding(
    sentence_count: int,
    window_fraction: float,
    excluded_indices: set[int],
) -> list[int]:
    """Return central eligible indices while respecting sentence de-duplication.

    A hard window can exhaust its eligible strings after previously selected
    sentences are excluded. Unlike a distribution with non-zero support throughout
    the report, it then needs an explicit fallback. The window expands only as far
    as necessary, admitting every unused position tied for minimum distance to the
    centre. The function raises rather than loops if no unused position remains.
    """
    invalid = [index for index in excluded_indices if not 0 <= index < sentence_count]
    if invalid:
        raise ValueError("excluded index is outside the report")
    central = [
        index
        for index in central_window_indices(sentence_count, window_fraction)
        if index not in excluded_indices
    ]
    if central:
        return central
    available = [index for index in range(sentence_count) if index not in excluded_indices]
    if not available:
        raise ValueError("No unused sentence remains for fixed-window sampling")
    distances = {
        index: abs(normalized_sentence_position(index, sentence_count) - 0.5)
        for index in available
    }
    minimum_distance = min(distances.values())
    return [
        index
        for index in available
        if abs(distances[index] - minimum_distance) < 1e-12
    ]


def within_report_probabilities(
    candidate_indices: Sequence[int],
    sentence_count: int,
    strategy: str,
    *,
    sigma_factor: float = 0.2,
    window_fraction: float = 0.4,
) -> list[float]:
    """Return normalized position probabilities for eligible report sentences.

    ``candidate_indices`` refer to positions in the original sentence sequence,
    before annotated findings are removed.  Gaussian probabilities are discrete
    weights evaluated at sentence indices.  Fixed-window probabilities are uniform
    over eligible candidates inside the normalized window; if annotation removal
    leaves none there, the closest eligible position(s) to the centre are used.
    """
    if sentence_count < 1:
        raise ValueError("sentence_count must be positive")
    if not candidate_indices:
        raise ValueError("candidate_indices cannot be empty")
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("candidate_indices must be unique")
    if any(not 0 <= index < sentence_count for index in candidate_indices):
        raise ValueError("candidate index is outside the report")
    if strategy == "uniform":
        return [1.0 / len(candidate_indices)] * len(candidate_indices)
    if strategy == "gaussian":
        if sigma_factor <= 0:
            raise ValueError("sigma_factor must be positive")
        mean = (sentence_count - 1) / 2
        sigma = sentence_count * sigma_factor
        scores = [
            math.exp(-0.5 * ((index - mean) / sigma) ** 2)
            for index in candidate_indices
        ]
    elif strategy == "central_window":
        if not 0 < window_fraction <= 1:
            raise ValueError("window_fraction must be in (0, 1]")
        central = set(central_window_indices(sentence_count, window_fraction))
        selected = [index for index in candidate_indices if index in central]
        if not selected:
            distances = {
                index: abs(normalized_sentence_position(index, sentence_count) - 0.5)
                for index in candidate_indices
            }
            minimum = min(distances.values())
            selected = [
                index
                for index in candidate_indices
                if abs(distances[index] - minimum) < 1e-12
            ]
        selected_set = set(selected)
        scores = [1.0 if index in selected_set else 0.0 for index in candidate_indices]
    else:
        raise ValueError(f"Unknown negative sampling strategy: {strategy}")
    total = sum(scores)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Position probabilities have no positive finite mass")
    return [score / total for score in scores]


def global_report_sentence_weights(
    candidate_indices_by_report: Sequence[Sequence[int]],
    sentence_counts: Sequence[int],
    strategy: str,
    *,
    sigma_factor: float = 0.2,
    window_fraction: float = 0.4,
    report_prior_power: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Assign one global weight to every eligible report--sentence pair.

    A report with ``m`` eligible candidates receives prior mass proportional to
    ``m ** report_prior_power``.  The default power 0 gives every non-empty report
    equal total mass; power 1 recovers a length-proportional report prior.  Within
    each report that mass is divided according to the requested position strategy.
    """
    if len(candidate_indices_by_report) != len(sentence_counts):
        raise ValueError("candidate and sentence-count lists must align")
    if not math.isfinite(report_prior_power):
        raise ValueError("report_prior_power must be finite")
    nonempty = [
        index
        for index, candidates in enumerate(candidate_indices_by_report)
        if candidates
    ]
    if not nonempty:
        raise ValueError("At least one report must contain an eligible negative")
    report_scores = {
        index: len(candidate_indices_by_report[index]) ** report_prior_power
        for index in nonempty
    }
    report_total = sum(report_scores.values())
    if not math.isfinite(report_total) or report_total <= 0:
        raise ValueError("Report priors have no positive finite mass")

    weighted = []
    for report_index in nonempty:
        candidates = candidate_indices_by_report[report_index]
        conditional = within_report_probabilities(
            candidates,
            sentence_counts[report_index],
            strategy,
            sigma_factor=sigma_factor,
            window_fraction=window_fraction,
        )
        report_probability = report_scores[report_index] / report_total
        weighted.extend(
            (report_index, sentence_index, report_probability * probability)
            for sentence_index, probability in zip(candidates, conditional)
            if probability > 0
        )
    if abs(sum(item[2] for item in weighted) - 1.0) > 1e-10:
        raise AssertionError("Global report--sentence weights do not sum to one")
    return weighted


def probability_proportional_sample_without_replacement(
    weights: Sequence[float], sample_size: int, *, seed: int
) -> list[int]:
    """Select indices by a seeded exponential race without replacement.

    The first draw is exactly proportional to ``weights``. Subsequent draws are
    made from the remaining probability mass, so no report quota is imposed and no
    report--sentence pair can be repeated.
    """
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")
    positive = [
        index
        for index, weight in enumerate(weights)
        if isinstance(weight, (int, float))
        and not isinstance(weight, bool)
        and math.isfinite(float(weight))
        and weight > 0
    ]
    if len(positive) != len(weights):
        raise ValueError("Every sampling weight must be positive and finite")
    if sample_size > len(weights):
        raise ValueError(
            f"Cannot sample {sample_size} unique items from {len(weights)} "
            "positive-weight candidates"
        )
    rng = random.Random(seed)
    keys = []
    for index, weight in enumerate(weights):
        # random() is in [0, 1); 1-random() is therefore in (0, 1].
        key = -math.log(1.0 - rng.random()) / float(weight)
        keys.append((key, index))
    keys.sort()
    return [index for _, index in keys[:sample_size]]


def probability_proportional_unique_group_sample(
    weights: Sequence[float],
    group_ids: Sequence[str],
    sample_size: int,
    *,
    seed: int,
) -> list[int]:
    """Sample globally unique groups while retaining a weighted context choice.

    Each occurrence receives an exponential-race key. The first occurrence of a
    group determines both that group's arrival time and which report context is
    retained; later occurrences of the same normalized sentence are skipped while
    the sampler continues. Equivalently, a group's rate is the sum of its
    occurrence weights and its retained context is chosen proportional to the
    occurrence weights. No report quota is imposed.
    """
    if len(weights) != len(group_ids):
        raise ValueError("weights and group_ids must align")
    if sample_size == 0:
        return []
    if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
        raise ValueError("Every group_id must be a non-empty string")
    unique_groups = len(set(group_ids))
    if sample_size > unique_groups:
        raise ValueError(
            f"Cannot sample {sample_size} globally unique sentences from only "
            f"{unique_groups} positive-weight sentence groups"
        )
    # Request the full weighted ordering, then retain the first occurrence of each
    # normalized sentence until the target number of unique groups is reached.
    ordering = probability_proportional_sample_without_replacement(
        weights, len(weights), seed=seed
    )
    selected = []
    seen = set()
    for index in ordering:
        group_id = group_ids[index]
        if group_id in seen:
            continue
        seen.add(group_id)
        selected.append(index)
        if len(selected) == sample_size:
            return selected
    raise AssertionError("Unique-group sampling ended before reaching its target")
