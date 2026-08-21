"""Input contracts for the robust-evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from hf_finetuning_lab.evaluation.robust import (
    STABLE_SUBGROUP_SUPPORT,
    bootstrap_metric,
    expected_calibration_error,
    find_best_threshold,
    reliability_curve,
    subgroup_metrics,
)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def test_bootstrap_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        bootstrap_metric(np.array([]), np.array([]), _accuracy, n_iter=10)


def test_bootstrap_reports_which_sample_broke_the_metric() -> None:
    def explodes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise RuntimeError("undefined for this resample")

    with pytest.raises(ValueError, match="metric_fn failed on a sample of 4 rows"):
        bootstrap_metric(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), explodes, n_iter=5)


def test_stratified_bootstrap_keeps_every_class_in_each_replicate() -> None:
    """An ordinary replicate can drop the minority class; a stratified one cannot."""
    y_true = np.array([0] * 20 + [1] * 3)
    y_pred = np.array([0] * 20 + [1] * 3)
    seen: list[int] = []

    def counting_metric(yt: np.ndarray, yp: np.ndarray) -> float:
        seen.append(len(np.unique(yt)))
        return _accuracy(yt, yp)

    bootstrap_metric(y_true, y_pred, counting_metric, n_iter=50, seed=1, stratify=True)

    assert set(seen) == {2}


def test_bootstrap_returns_an_interval_around_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = y_true.copy()
    y_pred[:20] = 1 - y_pred[:20]

    result = bootstrap_metric(y_true, y_pred, _accuracy, n_iter=200, seed=7)

    assert result["ci_low"] <= result["value"] <= result["ci_high"]


def test_reliability_curve_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        reliability_curve(np.array([]), np.empty((0, 2)))


@pytest.mark.parametrize("bad", [np.array([[0.5, 1.7]]), np.array([[0.5, np.nan]])])
def test_calibration_rejects_impossible_probabilities(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1]), bad)


def test_find_best_threshold_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        find_best_threshold(np.array([0, 1, 1]), np.array([0.2, 0.8]), _accuracy)


def test_find_best_threshold_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="thresholds must not be empty"):
        find_best_threshold(np.array([0, 1]), np.array([0.2, 0.8]), _accuracy, thresholds=[])


def test_find_best_threshold_rejects_out_of_range_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        find_best_threshold(np.array([0, 1]), np.array([0.2, 1.4]), _accuracy)


def test_find_best_threshold_picks_a_separating_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    result = find_best_threshold(y_true, y_prob, lambda t, p: float(f1_score(t, p)))

    assert 0.2 < result["threshold"] <= 0.8
    assert result["metric"] == 1.0


def test_subgroup_metrics_drops_groups_below_minimum_support() -> None:
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1])
    groups = ["big", "big", "big", "big", "tiny"]

    frame = subgroup_metrics(y_true, y_pred, groups, {"accuracy": _accuracy}, min_support=2)

    assert list(frame.index) == ["big"]


def test_subgroup_metrics_flags_groups_too_small_to_trust() -> None:
    n = STABLE_SUBGROUP_SUPPORT
    y_true = np.array([0, 1] * n)
    y_pred = y_true.copy()
    groups = ["small"] * 2 + ["large"] * (2 * n - 2)

    frame = subgroup_metrics(y_true, y_pred, groups, {"accuracy": _accuracy})

    assert bool(frame.loc["small", "low_support"]) is True
    assert bool(frame.loc["large", "low_support"]) is False


def test_subgroup_metrics_rejects_a_nonsense_support_floor() -> None:
    with pytest.raises(ValueError, match="min_support"):
        subgroup_metrics(np.array([0]), np.array([0]), ["a"], {"accuracy": _accuracy}, min_support=0)
