from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hf_finetuning_lab.evaluation.evaluator import evaluate_model


class _FakePredictor:
    """Predictor stub used to test evaluation behavior without transformers."""

    def __init__(self, predictions: list[dict[str, object]], id2label: dict[int, str]) -> None:
        self._predictions = predictions
        self.id2label = id2label

    def predict(self, texts: list[str]) -> list[dict[str, object]]:
        return self._predictions


def _write_eval_input(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "text": ["alpha account", "beta billing"],
            "label": ["account", "billing"],
        }
    )
    frame.to_csv(path, index=False)


def test_evaluate_model_normalizes_label_n_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "eval.csv"
    output_path = tmp_path / "evaluation.json"
    _write_eval_input(input_path)

    fake_predictor = _FakePredictor(
        predictions=[
            {"predicted_label": "LABEL_0", "confidence": 0.9},
            {"predicted_label": "LABEL_1", "confidence": 0.8},
        ],
        id2label={0: "account", 1: "billing"},
    )
    monkeypatch.setattr(
        "hf_finetuning_lab.evaluation.evaluator.TextClassificationPredictor",
        lambda model_dir: fake_predictor,
    )

    written = evaluate_model(tmp_path, input_path, output_path)

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["metrics"]["accuracy"] == pytest.approx(1.0)
    assert payload["confusion_matrix"]["account"]["account"] == 1
    assert payload["confusion_matrix"]["billing"]["billing"] == 1


def test_evaluate_model_rejects_unknown_predicted_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "eval.csv"
    output_path = tmp_path / "evaluation.json"
    _write_eval_input(input_path)

    fake_predictor = _FakePredictor(
        predictions=[
            {"predicted_label": "account", "confidence": 0.9},
            {"predicted_label": "escalation", "confidence": 0.6},
        ],
        id2label={0: "account", 1: "billing"},
    )
    monkeypatch.setattr(
        "hf_finetuning_lab.evaluation.evaluator.TextClassificationPredictor",
        lambda model_dir: fake_predictor,
    )

    with pytest.raises(ValueError, match="outside the trained label space"):
        evaluate_model(tmp_path, input_path, output_path)
