import pytest

from finding_level_itv.sampling import (
    central_window_indices,
    global_report_sentence_weights,
    normalized_sentence_position,
    probability_proportional_unique_group_sample,
    within_report_probabilities,
)


def test_normalized_positions_handle_even_and_odd_report_lengths():
    assert normalized_sentence_position(5, 11) == 0.5
    assert normalized_sentence_position(2, 6) == 0.4
    assert normalized_sentence_position(3, 6) == 0.6


def test_fixed_window_scales_with_report_length_and_is_symmetric():
    assert central_window_indices(11, 0.4) == [3, 4, 5, 6, 7]
    assert central_window_indices(6, 0.4) == [2, 3]
    assert central_window_indices(2, 0.4) == [0, 1]


def test_gaussian_probabilities_prefer_the_report_centre():
    weights = within_report_probabilities(range(5), 5, "gaussian", sigma_factor=0.2)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[2] > weights[0]
    assert weights[0] == weights[-1]


def test_equal_report_prior_does_not_impose_equal_sample_counts():
    weighted = global_report_sentence_weights(
        [[0, 1], [0, 1, 2, 3]],
        [2, 4],
        "uniform",
        report_prior_power=0,
    )
    mass = {
        report: sum(weight for item_report, _, weight in weighted if item_report == report)
        for report in (0, 1)
    }
    assert mass[0] == mass[1] == 0.5


def test_unique_group_sampling_is_seeded_and_global():
    groups = ["shared", "shared", "unique-a", "unique-b"]
    first = probability_proportional_unique_group_sample(
        [0.4, 0.1, 0.2, 0.3], groups, 3, seed=42
    )
    second = probability_proportional_unique_group_sample(
        [0.4, 0.1, 0.2, 0.3], groups, 3, seed=42
    )
    assert first == second
    assert len({groups[index] for index in first}) == 3

