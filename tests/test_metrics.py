import numpy as np

from hf_finetuning_lab.evaluation.metrics import compute_classification_metrics, softmax


def test_softmax_rows_sum_to_one() -> None:
    probs = softmax(np.array([[1.0, 2.0], [0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_compute_classification_metrics() -> None:
    metrics = compute_classification_metrics(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
    )
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
