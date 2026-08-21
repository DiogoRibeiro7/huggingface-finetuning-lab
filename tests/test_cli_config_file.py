"""The --config-file guard on `hf-lab train`.

This path had no coverage, which is how a dependency bump that broke it
reached a pull request with green checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from hf_finetuning_lab.cli import app

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_name": "distilbert-base-uncased",
                "text_col": "text",
                "label_col": "label",
                "epochs": 3,
                "batch_size": 8,
                "max_length": 64,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def captured_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the config the CLI builds, without running any training."""
    seen: dict[str, Any] = {}

    def fake_train(input_path: Any, output_dir: Any, config: Any) -> Path:
        seen["config"] = config
        return Path(output_dir)

    monkeypatch.setattr("hf_finetuning_lab.cli.train_text_classifier", fake_train)
    return seen


def test_config_file_alone_is_accepted(
    config_file: Path, captured_config: dict[str, Any], tmp_path: Path
) -> None:
    """The documented config-driven path: no flags, everything from YAML."""
    result = runner.invoke(
        app,
        ["train", "--input", "x.csv", "--output-dir", str(tmp_path / "out"),
         "--config-file", str(config_file)],
    )

    assert result.exit_code == 0, result.output
    assert captured_config["config"].epochs == 3
    assert captured_config["config"].batch_size == 8


def test_config_file_with_an_explicit_flag_is_rejected(
    config_file: Path, tmp_path: Path
) -> None:
    """A flag that would be silently ignored must fail loudly instead."""
    result = runner.invoke(
        app,
        ["train", "--input", "x.csv", "--output-dir", str(tmp_path / "out"),
         "--config-file", str(config_file), "--epochs", "7"],
    )

    assert result.exit_code != 0
    assert "cannot be combined" in result.output
    assert "epochs" in result.output


def test_rejection_names_every_conflicting_flag(config_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["train", "--input", "x.csv", "--output-dir", str(tmp_path / "out"),
         "--config-file", str(config_file), "--epochs", "7", "--batch-size", "32"],
    )

    assert result.exit_code != 0
    assert "epochs" in result.output
    assert "batch_size" in result.output


def test_flags_without_a_config_file_build_the_config(
    captured_config: dict[str, Any], tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["train", "--input", "x.csv", "--output-dir", str(tmp_path / "out"),
         "--epochs", "5", "--batch-size", "4"],
    )

    assert result.exit_code == 0, result.output
    assert captured_config["config"].epochs == 5
    assert captured_config["config"].batch_size == 4
