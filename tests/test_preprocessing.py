from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hf_finetuning_lab.inference.predictor import TextClassificationPredictor
from hf_finetuning_lab.tokenization.preprocessing import (
    PreprocessingConfig,
    load_preprocessing_config,
    write_preprocessing_config,
)


def test_round_trips_through_the_model_directory(tmp_path: Path) -> None:
    write_preprocessing_config(tmp_path, PreprocessingConfig(max_length=160))

    assert load_preprocessing_config(tmp_path) == PreprocessingConfig(
        max_length=160, truncation=True
    )


def test_falls_back_to_the_training_config_for_older_artifacts(tmp_path: Path) -> None:
    (tmp_path / "training_config.json").write_text(
        json.dumps({"max_length": 96, "epochs": 2}), encoding="utf-8"
    )

    assert load_preprocessing_config(tmp_path).max_length == 96


def test_defaults_when_the_artifact_records_nothing(tmp_path: Path) -> None:
    config = load_preprocessing_config(tmp_path)

    assert config.max_length is None
    assert config.truncation is True
    # Without a recorded length the pipeline still truncates at the model limit.
    assert config.tokenizer_kwargs() == {"truncation": True}


def test_tokenizer_kwargs_carry_the_recorded_length() -> None:
    assert PreprocessingConfig(max_length=32).tokenizer_kwargs() == {
        "truncation": True,
        "max_length": 32,
    }


class _RecordingPipeline:
    """Stands in for the transformers pipeline, capturing its call kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return [[{"label": "account", "score": 1.0}] for _ in texts]


def test_predictor_applies_the_training_max_length(tmp_path: Path, monkeypatch: Any) -> None:
    """Training and inference must clip at the same boundary."""
    write_preprocessing_config(tmp_path, PreprocessingConfig(max_length=48))
    recorder = _RecordingPipeline()
    monkeypatch.setattr(
        TextClassificationPredictor, "_load_pipeline", lambda self: recorder
    )

    predictor = TextClassificationPredictor(model_dir=tmp_path)
    predictor.predict(["some ticket text"])

    assert recorder.calls == [{"truncation": True, "max_length": 48}]
