from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax probabilities."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return np.asarray(exp_logits / exp_logits.sum(axis=1, keepdims=True))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "weighted",
) -> dict[str, float]:
    """Compute classification metrics for model evaluation."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # ROC-AUC is only well-defined here for a genuinely binary classifier
    # (a 2-column probability matrix) evaluated on data containing both
    # classes. Requiring shape[1] == 2 prevents a multiclass model from
    # reporting a meaningless AUC when a batch happens to contain two labels.
    if y_prob is not None:
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim == 2 and prob_arr.shape[1] == 2 and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_arr[:, 1]))

    return metrics


def trainer_compute_metrics(eval_prediction) -> dict[str, float]:
    """Metric callback compatible with Hugging Face Trainer."""
    logits, labels = eval_prediction
    probs = softmax(np.asarray(logits))
    preds = probs.argmax(axis=1)
    return compute_classification_metrics(labels, preds, probs)


def confusion_matrix_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: dict[int, str],
) -> pd.DataFrame:
    """Return a labelled confusion matrix as a DataFrame."""
    labels = sorted(id2label)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = [id2label[idx] for idx in labels]
    return pd.DataFrame(matrix, index=names, columns=names)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return a per-class precision/recall/F1/support DataFrame.

    Rows are per-class entries plus ``accuracy``, ``macro avg``, and
    ``weighted avg`` (as produced by :func:`sklearn.metrics.classification_report`).
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).T
    if "support" in frame.columns:
        frame["support"] = frame["support"].astype(int)
    return frame
