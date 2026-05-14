"""Robust evaluation utilities: calibration, thresholds, bootstrap CIs, subgroups, drift."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _top_confidence_and_correct(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample top-class confidence and whether the top class was correct."""
    if y_prob.ndim == 1:
        # Treat as positive-class probability for a binary task with labels {0, 1}.
        confidence = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
        prediction = (y_prob >= 0.5).astype(int)
    elif y_prob.ndim == 2:
        confidence = y_prob.max(axis=1)
        prediction = y_prob.argmax(axis=1)
    else:
        raise ValueError("y_prob must be 1D (binary) or 2D (multiclass).")
    correct = (prediction == np.asarray(y_true)).astype(int)
    return confidence, correct


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return a per-bin reliability table for the top-class confidence.

    Each row contains the bin range, mean confidence, observed accuracy, and
    sample count. Empty bins are dropped.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    confidence, correct = _top_confidence_and_correct(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Use right-inclusive bins so that confidence == 1.0 falls in the last bucket.
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE) over equal-width bins."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    if curve.empty:
        return 0.0
    total = float(curve["count"].sum())
    weighted_gap = (
        (curve["mean_confidence"] - curve["accuracy"]).abs() * curve["count"]
    ).sum()
    return float(weighted_gap / total)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate a metric with a bootstrap confidence interval.

    Returns a dictionary with ``value`` (point estimate on the full sample),
    ``ci_low``, ``ci_high``, and ``n_iter``.
    """
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(metric_fn(y_true[idx], y_pred[idx]))
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return {"value": point, "ci_low": low, "ci_high": high, "n_iter": float(n_iter)}


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    """Sweep thresholds over a binary positive-class probability and return the best.

    ``y_prob`` is a 1D array of probabilities for the positive class (1).
    The returned dictionary contains ``threshold`` and ``metric``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 1:
        raise ValueError("find_best_threshold expects a 1D positive-class probability array.")
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_metric = -np.inf
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = float(metric_fn(y_true, pred))
        if score > best_metric:
            best_metric = score
            best_threshold = float(t)
    return {"threshold": float(best_threshold), "metric": float(best_metric)}


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence[Any],
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> pd.DataFrame:
    """Compute metrics per subgroup defined by ``groups``.

    The returned DataFrame is indexed by group value and has one column per
    metric in ``metric_fns`` plus a ``count`` column. Subgroups with fewer than
    one sample are omitted.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_arr = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(group_arr)):
        raise ValueError("y_true, y_pred, and groups must have the same length.")

    rows: dict[Any, dict[str, float]] = {}
    for value in pd.unique(group_arr):
        mask = group_arr == value
        if not mask.any():
            continue
        entry: dict[str, float] = {"count": int(mask.sum())}
        for name, fn in metric_fns.items():
            entry[name] = float(fn(y_true[mask], y_pred[mask]))
        rows[value] = entry
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "group"
    if "count" in frame.columns:
        frame["count"] = frame["count"].astype(int)
    return frame


def prediction_share_drift(
    predictions_a: Sequence[Any],
    predictions_b: Sequence[Any],
    labels: Sequence[Any] | None = None,
    smoothing: float = 1e-6,
) -> pd.DataFrame:
    """Compare label-share distributions between two prediction sets.

    Returns one row per label with ``share_a``, ``share_b``, ``delta`` and a
    Population Stability Index (PSI) contribution. The PSI total is recorded
    in the ``psi_total`` attribute of the DataFrame (``frame.attrs``) and
    written to a single ``__total__`` row labelled ``total`` for visibility.
    """
    a = np.asarray(predictions_a)
    b = np.asarray(predictions_b)
    if labels is None:
        labels = sorted({*a.tolist(), *b.tolist()}, key=str)

    rows: list[dict[str, Any]] = []
    psi_total = 0.0
    n_a = max(len(a), 1)
    n_b = max(len(b), 1)
    for label in labels:
        share_a = float((a == label).sum()) / n_a
        share_b = float((b == label).sum()) / n_b
        safe_a = share_a + smoothing
        safe_b = share_b + smoothing
        psi = float((safe_b - safe_a) * np.log(safe_b / safe_a))
        psi_total += psi
        rows.append(
            {
                "label": label,
                "share_a": share_a,
                "share_b": share_b,
                "delta": share_b - share_a,
                "psi": psi,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["psi_total"] = psi_total
    return frame
