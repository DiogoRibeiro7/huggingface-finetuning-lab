from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hf_finetuning_lab import __version__
from hf_finetuning_lab.artifacts import (
    ALTERNATIVE_REQUIRED_FILES,
    REQUIRED_FILES,
    ArtifactCheck,
    ArtifactReport,
    verify_artifact,
)
from hf_finetuning_lab.cli import app

runner = CliRunner()


def _write_minimal_artifact(path: Path) -> None:
    """Drop the files that satisfy the required + alternative-required contract."""
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    # First option from each alternative group.
    for group in ALTERNATIVE_REQUIRED_FILES:
        (path / group[0]).write_text("placeholder", encoding="utf-8")


def test_verify_artifact_reports_ok_for_minimal_artifact(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    report = verify_artifact(tmp_path)
    assert isinstance(report, ArtifactReport)
    assert report.ok
    # Required files are present; recommended ones are warnings.
    assert all(check.status in {"ok", "warning"} for check in report.checks)
    assert report.warnings  # recommended files are not present


def test_verify_artifact_marks_missing_required_files(tmp_path: Path) -> None:
    # Only drop weights, no config or tokenizer.
    (tmp_path / "model.safetensors").write_text("weights", encoding="utf-8")
    report = verify_artifact(tmp_path)
    assert not report.ok
    assert "config.json" in report.missing


def test_verify_artifact_accepts_alternative_files(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    # Replace safetensors with pytorch_model.bin.
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "pytorch_model.bin").write_text("weights", encoding="utf-8")
    report = verify_artifact(tmp_path)
    assert report.ok


def test_verify_artifact_promotes_recommended_files_when_present(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    (tmp_path / "model_card.md").write_text("# card", encoding="utf-8")
    report = verify_artifact(tmp_path)
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["model_card.md"] == "ok"


def test_verify_artifact_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_artifact(tmp_path / "does-not-exist")


def test_verify_artifact_rejects_non_directory(tmp_path: Path) -> None:
    target = tmp_path / "a-file"
    target.write_text("not a dir", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        verify_artifact(target)


def test_artifact_check_dataclass_field_defaults() -> None:
    check = ArtifactCheck(name="x", status="ok")
    assert check.detail == ""


def test_cli_version_matches_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


def test_cli_list_commands_lists_known_subcommands() -> None:
    result = runner.invoke(app, ["list-commands"])
    assert result.exit_code == 0, result.output
    names = {line.strip() for line in result.output.splitlines() if line.strip()}
    for expected in [
        "sample-data",
        "train",
        "evaluate",
        "predict",
        "serve",
        "list-hub-datasets",
        "fetch-hub-dataset",
        "verify-artifact",
        "version",
        "list-commands",
    ]:
        assert expected in names, f"missing CLI command in listing: {expected}"


def test_cli_verify_artifact_exits_zero_for_valid_artifact(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    result = runner.invoke(app, ["verify-artifact", "--model-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "[OK ]" in result.output


def test_cli_verify_artifact_exits_one_when_required_missing(tmp_path: Path) -> None:
    # Empty directory — required files are absent.
    result = runner.invoke(app, ["verify-artifact", "--model-dir", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "[MISS]" in result.output


def test_cli_verify_artifact_strict_flag_treats_warnings_as_failure(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    result = runner.invoke(
        app, ["verify-artifact", "--model-dir", str(tmp_path), "--strict"]
    )
    assert result.exit_code == 1, result.output
    assert "[WARN]" in result.output


def test_cli_verify_artifact_strict_passes_when_recommended_present(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    for name in ["tokenizer_config.json", "special_tokens_map.json", "model_card.md", "metrics.json"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["verify-artifact", "--model-dir", str(tmp_path), "--strict"]
    )
    assert result.exit_code == 0, result.output
