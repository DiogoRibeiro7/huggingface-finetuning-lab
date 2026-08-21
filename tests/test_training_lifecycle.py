"""End-to-end contract: train, save, verify, reload, predict.

The rest of the suite tests helpers in isolation. This exercises the real
lifecycle against a genuine (tiny) transformer, which is what catches
contract drift between training, the artifact layout, and inference.

The model is built locally rather than downloaded, so the test needs no
network and stays fast enough for every CI run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hf_finetuning_lab.artifacts import verify_artifact
from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.inference.predictor import TextClassificationPredictor
from hf_finetuning_lab.tokenization.preprocessing import load_preprocessing_config
from hf_finetuning_lab.training.trainer import train_text_classifier

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

LABELS = ("account", "billing", "delivery")

VOCAB_WORDS = (
    "my card was charged twice again cannot log into account password reset "
    "package never arrived tracking delivery late refund invoice billing login"
).split()


def _build_tiny_model(destination: Path) -> Path:
    """Create a small local sequence-classification checkpoint."""
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    destination.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *VOCAB_WORDS]
    vocab_path = destination / "vocab.txt"
    vocab_path.write_text("\n".join(dict.fromkeys(vocab)) + "\n", encoding="utf-8")

    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab_path))
    config = DistilBertConfig(
        vocab_size=max(len(tokenizer.get_vocab()), 32),
        dim=32,
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_position_embeddings=64,
    )
    model = DistilBertForSequenceClassification(config)

    checkpoint = destination / "base"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint


def _training_frame() -> pd.DataFrame:
    rows = []
    for _ in range(6):
        rows += [
            {"text": "cannot log into account password reset", "label": "account"},
            {"text": "my card was charged twice refund invoice", "label": "billing"},
            {"text": "package never arrived tracking delivery late", "label": "delivery"},
        ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_tiny_model(tmp_path_factory.mktemp("tiny-model"))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "tickets.csv"
    _training_frame().to_csv(path, index=False)
    return path


def _config(checkpoint: Path, **overrides: object) -> TrainingConfig:
    base = {
        "model_name": str(checkpoint),
        "epochs": 1,
        "batch_size": 4,
        "max_length": 32,
        "test_size": 0.2,
        "validation_size": 0.2,
    }
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def trained_model(tiny_checkpoint: Path, dataset: Path, tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("run") / "model"
    return train_text_classifier(dataset, output, _config(tiny_checkpoint))


def test_training_produces_a_valid_artifact(trained_model: Path) -> None:
    report = verify_artifact(trained_model)

    assert report.ok, [c.name for c in report.checks if c.status == "missing"]


def test_artifact_records_the_label_space(trained_model: Path) -> None:
    mapping = json.loads((trained_model / "label_mapping.json").read_text(encoding="utf-8"))

    assert sorted(mapping["label2id"]) == sorted(LABELS)


def test_artifact_records_the_preprocessing_contract(trained_model: Path) -> None:
    assert load_preprocessing_config(trained_model).max_length == 32


def test_reloaded_model_predicts_within_the_trained_label_space(trained_model: Path) -> None:
    predictor = TextClassificationPredictor(model_dir=trained_model)

    predictions = predictor.predict(
        ["my card was charged twice", "package never arrived"]
    )

    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction["predicted_label"] in LABELS
        assert 0.0 <= prediction["confidence"] <= 1.0


def test_reloaded_model_reports_a_probability_per_label(trained_model: Path) -> None:
    predictor = TextClassificationPredictor(model_dir=trained_model)

    prediction = predictor.predict(["cannot log into account"])[0]

    probabilities = {k: v for k, v in prediction.items() if k.startswith("prob_")}
    assert len(probabilities) == len(LABELS)
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_lora_run_saves_a_standalone_model(tiny_checkpoint: Path, dataset: Path, tmp_path: Path) -> None:
    """The adapter must be merged, so inference does not need peft or the base model."""
    pytest.importorskip("peft")
    output = tmp_path / "lora-model"

    train_text_classifier(
        dataset,
        output,
        _config(tiny_checkpoint, use_lora=True, lora_target_modules=["q_lin", "v_lin"]),
    )

    assert verify_artifact(output).ok
    assert not (output / "adapter_config.json").exists()
    prediction = TextClassificationPredictor(model_dir=output).predict(["refund invoice"])[0]
    assert prediction["predicted_label"] in LABELS
