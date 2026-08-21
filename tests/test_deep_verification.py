"""Deep verification distinguishes a real checkpoint from placeholder files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_finetuning_lab.artifacts import (
    ALTERNATIVE_REQUIRED_FILES,
    RECOMMENDED_FILES,
    REQUIRED_FILES,
    verify_artifact,
)


def _placeholder_artifact(path: Path) -> Path:
    """The layout-valid but meaningless artifact the shallow check accepts."""
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    for group in ALTERNATIVE_REQUIRED_FILES:
        (path / group[0]).write_text("placeholder", encoding="utf-8")
    for name in RECOMMENDED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    return path


def _status(report, name: str) -> str:
    return next(check.status for check in report.checks if check.name == name)


def test_shallow_check_accepts_placeholders(tmp_path: Path) -> None:
    """Documents the limit of the fast path, which deep mode exists to close."""
    assert verify_artifact(_placeholder_artifact(tmp_path)).ok


def test_deep_check_rejects_placeholders(tmp_path: Path) -> None:
    report = verify_artifact(_placeholder_artifact(tmp_path), deep=True)

    assert not report.ok


def test_deep_check_flags_unparseable_json(tmp_path: Path) -> None:
    _placeholder_artifact(tmp_path)
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")

    report = verify_artifact(tmp_path, deep=True)

    assert _status(report, "deep:config.json") == "missing"


def test_deep_check_flags_an_empty_weight_file(tmp_path: Path) -> None:
    _placeholder_artifact(tmp_path)
    (tmp_path / "model.safetensors").write_text("", encoding="utf-8")

    report = verify_artifact(tmp_path, deep=True)

    assert _status(report, "deep:weights") == "missing"


def test_deep_check_flags_a_label_space_mismatch(tmp_path: Path) -> None:
    """A head sized for two classes cannot serve a three-label mapping."""
    _placeholder_artifact(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "a", "1": "b"}}), encoding="utf-8"
    )
    (tmp_path / "label_mapping.json").write_text(
        json.dumps({"label2id": {"a": 0, "b": 1, "c": 2}}), encoding="utf-8"
    )

    report = verify_artifact(tmp_path, deep=True)

    assert _status(report, "deep:label_space") == "missing"


def test_deep_check_accepts_a_matching_label_space(tmp_path: Path) -> None:
    _placeholder_artifact(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "a", "1": "b"}}), encoding="utf-8"
    )
    (tmp_path / "label_mapping.json").write_text(
        json.dumps({"label2id": {"a": 0, "b": 1}}), encoding="utf-8"
    )

    report = verify_artifact(tmp_path, deep=True)

    assert _status(report, "deep:label_space") == "ok"


def test_deep_check_flags_non_finite_metrics(tmp_path: Path) -> None:
    _placeholder_artifact(tmp_path)
    (tmp_path / "test_metrics.json").write_text('{"eval_f1": NaN}', encoding="utf-8")

    report = verify_artifact(tmp_path, deep=True)

    assert _status(report, "deep:test_metrics.json") == "missing"


@pytest.mark.parametrize(
    "weight_file",
    ["model.safetensors.index.json", "pytorch_model.bin.index.json"],
)
def test_sharded_checkpoints_satisfy_the_layout(tmp_path: Path, weight_file: str) -> None:
    """A sharded checkpoint ships an index manifest instead of one weight file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / weight_file).write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert verify_artifact(tmp_path).ok


@pytest.mark.parametrize("tokenizer_file", ["spiece.model", "tokenizer.model", "sentencepiece.bpe.model"])
def test_sentencepiece_tokenizers_satisfy_the_layout(tmp_path: Path, tokenizer_file: str) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_text("weights", encoding="utf-8")
    (tmp_path / tokenizer_file).write_text("sp", encoding="utf-8")

    assert verify_artifact(tmp_path).ok
