from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from hf_finetuning_lab.evaluation.robust import (
    bootstrap_metric,
    expected_calibration_error,
    find_best_threshold,
    prediction_share_drift,
    reliability_curve,
    subgroup_metrics,
)


def _binary_probs(rng: np.random.Generator, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    y_true = rng.integers(0, 2, size=n)
    # Confidence correlates with truth, with some noise.
    base = np.where(y_true == 1, 0.75, 0.25)
    y_prob = np.clip(base + rng.normal(scale=0.1, size=n), 0.0, 1.0)
    return y_true, y_prob


def test_reliability_curve_returns_per_bin_rows() -> None:
    rng = np.random.default_rng(0)
    y_true, y_prob = _binary_probs(rng)
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert set(["bin_lower", "bin_upper", "mean_confidence", "accuracy", "count"]).issubset(
        curve.columns
    )
    assert (curve["count"] > 0).all()
    assert curve["count"].sum() == len(y_true)


def test_reliability_curve_rejects_zero_bins() -> None:
    with pytest.raises(ValueError):
        reliability_curve(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=0)


def test_expected_calibration_error_is_zero_for_perfect_calibration() -> None:
    # A two-bin example where confidence equals accuracy in each bin.
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # For label-1 samples confidence=0.9, accuracy=1.0; for label-0 samples
    # the top-class confidence is 0.8 (1 - 0.2), and they are all correct.
    # We expect a small but bounded ECE.
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 0.25


def test_expected_calibration_error_grows_with_miscalibration() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    confident_wrong = np.array([0.95] * 50 + [0.05] * 50)  # confident but wrong
    well_calibrated = np.array([0.05] * 50 + [0.95] * 50)
    assert expected_calibration_error(y_true, confident_wrong) > expected_calibration_error(
        y_true, well_calibrated
    )


def test_bootstrap_metric_returns_interval_containing_point() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=300)
    y_pred = y_true.copy()
    # Flip 10% of predictions to add variance.
    flip = rng.choice(len(y_true), size=30, replace=False)
    y_pred[flip] = 1 - y_pred[flip]

    result = bootstrap_metric(
        y_true,
        y_pred,
        metric_fn=lambda yt, yp: float(f1_score(yt, yp, average="binary", zero_division=0)),
        n_iter=200,
        alpha=0.05,
        seed=0,
    )
    assert result["ci_low"] <= result["value"] <= result["ci_high"]
    assert 0.0 <= result["ci_low"] <= 1.0
    assert 0.0 <= result["ci_high"] <= 1.0


def test_bootstrap_metric_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        bootstrap_metric(np.array([0, 1]), np.array([0, 1]), metric_fn=lambda *_: 0.0, n_iter=0)
    with pytest.raises(ValueError):
        bootstrap_metric(np.array([0, 1]), np.array([0, 1]), metric_fn=lambda *_: 0.0, alpha=1.5)


def test_find_best_threshold_recovers_obvious_optimum() -> None:
    # Two cleanly separated clusters.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    result = find_best_threshold(
        y_true,
        y_prob,
        metric_fn=lambda yt, yp: float(f1_score(yt, yp, average="binary", zero_division=0)),
    )
    assert 0.15 < result["threshold"] < 0.85
    assert result["metric"] == pytest.approx(1.0)


def test_subgroup_metrics_handles_multiple_groups() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    frame = subgroup_metrics(
        y_true,
        y_pred,
        groups,
        metric_fns={
            "f1": lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0)),
        },
    )
    assert set(frame.index) == {"A", "B"}
    assert frame.loc["A", "count"] == 3
    assert 0.0 <= frame.loc["A", "f1"] <= 1.0


def test_prediction_share_drift_zero_when_identical() -> None:
    preds = np.array(["x", "x", "y", "y", "z"])
    frame = prediction_share_drift(preds, preds)
    assert frame.attrs["psi_total"] == pytest.approx(0.0, abs=1e-9)
    assert (frame["delta"].abs() < 1e-9).all()


def test_prediction_share_drift_positive_when_distributions_differ() -> None:
    a = np.array(["x"] * 90 + ["y"] * 10)
    b = np.array(["x"] * 30 + ["y"] * 70)
    frame = prediction_share_drift(a, b)
    assert frame.attrs["psi_total"] > 0.1
