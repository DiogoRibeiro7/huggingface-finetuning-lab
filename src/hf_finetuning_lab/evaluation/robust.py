"""Robust evaluation utilities: calibration, thresholds, bootstrap CIs, subgroups, drift."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: Below this many rows a subgroup metric is too noisy to compare directly.
STABLE_SUBGROUP_SUPPORT = 30


def _validate_probabilities(y_prob: np.ndarray, name: str = "y_prob") -> None:
    """Reject probability arrays that cannot describe a distribution."""
    if not np.isfinite(y_prob).all():
        raise ValueError(f"{name} must be finite; got NaN or infinity.")
    if y_prob.size and (y_prob.min() < 0.0 or y_prob.max() > 1.0):
        raise ValueError(f"{name} must lie in [0, 1].")


def _apply_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Call a user metric, naming the sample when it fails."""
    try:
        return float(metric_fn(y_true, y_pred))
    except Exception as exc:
        labels = sorted(set(np.asarray(y_true).tolist()))
        raise ValueError(
            f"metric_fn failed on a sample of {len(y_true)} rows with labels {labels}: {exc}"
        ) from exc


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
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty.")
    _validate_probabilities(y_prob)

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
    """Compute the Expected Calibration Error over equal-width bins.

    This is *top-confidence* ECE: rows are binned by the confidence of the
    predicted class and compared with the accuracy in that bin. It is not
    positive-class calibration, which bins by the probability assigned to one
    specific class.
    """
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
    stratify: bool = False,
) -> dict[str, float]:
    """Estimate a metric with a percentile bootstrap confidence interval.

    This is an ordinary non-parametric bootstrap: every replicate draws ``n``
    rows with replacement, independently and identically. That assumes rows are
    independent, so it understates uncertainty on grouped data (repeated
    customers, threads, documents), which needs a grouped bootstrap instead.

    With ``stratify=True`` each replicate resamples within each class of
    ``y_true``, holding the class balance fixed. Prefer it for classification
    metrics on small or imbalanced samples, where an ordinary replicate can
    omit a class entirely and leave the metric undefined.

    Returns ``value`` (point estimate on the full sample), ``ci_low``,
    ``ci_high``, and ``n_iter``.
    """
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ValueError("y_true must not be empty; a bootstrap needs a sample.")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = _apply_metric(metric_fn, y_true, y_pred)
    class_indices = (
        [np.flatnonzero(y_true == label) for label in np.unique(y_true)] if stratify else []
    )
    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        if stratify:
            idx = np.concatenate(
                [rng.choice(members, size=len(members), replace=True) for members in class_indices]
            )
        else:
            idx = rng.integers(0, n, size=n)
        samples[i] = _apply_metric(metric_fn, y_true[idx], y_pred[idx])
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

    Choosing a threshold is *fitting*. Run this on validation data and score
    the chosen threshold once on the held-out test split; selecting and
    evaluating on the same rows reports an optimistic number.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 1:
        raise ValueError("find_best_threshold expects a 1D positive-class probability array.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty.")
    _validate_probabilities(y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    grid = sorted({float(t) for t in thresholds})
    if not grid:
        raise ValueError("thresholds must not be empty.")
    best_threshold = 0.5
    best_metric = -np.inf
    for t in grid:
        pred = (y_prob >= t).astype(int)
        score = _apply_metric(metric_fn, y_true, pred)
        if score > best_metric:
            best_metric = score
            best_threshold = float(t)
    return {"threshold": float(best_threshold), "metric": float(best_metric)}


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence[Any],
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    min_support: int = 1,
) -> pd.DataFrame:
    """Compute metrics per subgroup defined by ``groups``.

    The returned DataFrame is indexed by group value and has one column per
    metric in ``metric_fns``, plus ``count`` and ``low_support``. Groups
    smaller than ``min_support`` are dropped.

    A metric over a handful of rows is not comparable with one over thousands,
    so ``low_support`` flags groups below
    :data:`STABLE_SUBGROUP_SUPPORT` rows: read those as indicative rather than
    as evidence of a disparity.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_arr = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(group_arr)):
        raise ValueError("y_true, y_pred, and groups must have the same length.")

    if min_support < 1:
        raise ValueError("min_support must be at least 1.")

    rows: dict[Any, dict[str, float]] = {}
    for value in pd.unique(group_arr):
        mask = group_arr == value
        count = int(mask.sum())
        if count < min_support:
            continue
        entry: dict[str, float] = {
            "count": count,
            "low_support": count < STABLE_SUBGROUP_SUPPORT,
        }
        for name, fn in metric_fns.items():
            entry[name] = _apply_metric(fn, y_true[mask], y_pred[mask])
        rows[value] = entry
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "group"
    if "count" in frame.columns:
        frame["count"] = frame["count"].astype(int)
        frame["low_support"] = frame["low_support"].astype(bool)
    return frame


def prediction_share_drift(
    predictions_a: Sequence[Any],
    predictions_b: Sequence[Any],
    labels: Sequence[Any] | None = None,
    smoothing: float = 1e-6,
) -> pd.DataFrame:
    """Compare label-share distributions between two prediction sets.

    Returns one row per label with ``share_a``, ``share_b``, ``delta`` and a
    Population Stability Index (PSI) contribution. The PSI total (the sum of the
    per-label contributions) is recorded in ``frame.attrs["psi_total"]``. Note
    that ``attrs`` does not survive a CSV round-trip, so persist the total
    separately if you need it downstream.
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
