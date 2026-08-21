"""The Hub publication commands, with the Hub client mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from hf_finetuning_lab.artifacts import (
    ALTERNATIVE_REQUIRED_FILES,
    RECOMMENDED_FILES,
    REQUIRED_FILES,
)
from hf_finetuning_lab.cli import app

runner = CliRunner()


def _artifact(path: Path, *, with_heldout_rows: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    for group in ALTERNATIVE_REQUIRED_FILES:
        (path / group[0]).write_text("weights", encoding="utf-8")
    for name in RECOMMENDED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    (path / "training_config.json").write_text(
        '{"model_name": "distilbert-base-uncased", "max_length": 160}', encoding="utf-8"
    )
    (path / "test_metrics.json").write_text(
        '{"eval_f1": 0.88, "eval_runtime": 1.5}', encoding="utf-8"
    )
    if with_heldout_rows:
        (path / "heldout_test.csv").write_text("text,label\na,b\n", encoding="utf-8")
    return path


@pytest.fixture
def recorded_api(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Capture Hub calls made through the publishing module."""
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Api:
        def create_repo(self, **kw: Any) -> str:
            calls.append(("create_repo", kw))
            return kw.get("repo_id", "")

        def upload_folder(self, **kw: Any) -> str:
            calls.append(("upload_folder", kw))
            return "sha"

        def create_branch(self, **kw: Any) -> None:
            calls.append(("create_branch", kw))

        def snapshot_download(self, **kw: Any) -> str:
            calls.append(("snapshot_download", kw))
            return kw.get("local_dir", "")

    monkeypatch.setattr("hf_finetuning_lab.publishing._default_api", lambda: _Api())
    return calls


def test_dry_run_reports_the_plan_without_uploading(
    tmp_path: Path, recorded_api: list[tuple[str, dict[str, Any]]]
) -> None:
    result = runner.invoke(
        app,
        ["push-to-hub", "--model-dir", str(_artifact(tmp_path)), "--repo-id", "me/support",
         "--license", "mit", "--dataset", "fancyzhx/ag_news", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Would publish" in result.output
    # The card it would publish is shown, and nothing is sent.
    assert "pipeline_tag: text-classification" in result.output
    assert recorded_api == []


def test_push_uploads_to_staging_by_default(
    tmp_path: Path, recorded_api: list[tuple[str, dict[str, Any]]]
) -> None:
    result = runner.invoke(
        app, ["push-to-hub", "--model-dir", str(_artifact(tmp_path)), "--repo-id", "me/support"]
    )

    assert result.exit_code == 0, result.output
    uploads = [kw for name, kw in recorded_api if name == "upload_folder"]
    assert uploads[0]["revision"] == "staging"


def test_push_refuses_public_publication_of_heldout_rows(
    tmp_path: Path, recorded_api: list[tuple[str, dict[str, Any]]]
) -> None:
    result = runner.invoke(
        app,
        ["push-to-hub", "--model-dir", str(_artifact(tmp_path, with_heldout_rows=True)),
         "--repo-id", "me/support"],
    )

    assert result.exit_code == 1
    assert "held-out rows" in result.output
    assert recorded_api == []


def test_push_rejects_a_bare_repo_id(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["push-to-hub", "--model-dir", str(_artifact(tmp_path)), "--repo-id", "support"]
    )

    assert result.exit_code != 0


def test_card_metrics_come_from_the_artifact(
    tmp_path: Path, recorded_api: list[tuple[str, dict[str, Any]]]
) -> None:
    """Derived from what the run recorded, and runtime entries are excluded."""
    result = runner.invoke(
        app,
        ["push-to-hub", "--model-dir", str(_artifact(tmp_path)), "--repo-id", "me/support",
         "--dry-run"],
    )

    assert "f1" in result.output
    assert "eval_runtime" not in result.output
    assert "base_model: distilbert-base-uncased" in result.output


def test_promote_points_release_at_the_reviewed_commit(
    recorded_api: list[tuple[str, dict[str, Any]]],
) -> None:
    result = runner.invoke(
        app, ["promote-to-hub", "--repo-id", "me/support", "--from", "staging", "--to", "v1"]
    )

    assert result.exit_code == 0, result.output
    branch = [kw for name, kw in recorded_api if name == "create_branch"][0]
    assert branch["branch"] == "v1"
    assert branch["revision"] == "staging"
    assert not [name for name, _ in recorded_api if name == "upload_folder"]


def test_promote_rejects_a_no_op(recorded_api: list[tuple[str, dict[str, Any]]]) -> None:
    result = runner.invoke(
        app, ["promote-to-hub", "--repo-id", "me/support", "--from", "main", "--to", "main"]
    )

    assert result.exit_code != 0


def test_pull_pins_the_revision(
    tmp_path: Path, recorded_api: list[tuple[str, dict[str, Any]]]
) -> None:
    result = runner.invoke(
        app,
        ["pull-model", "--repo-id", "me/support", "--output-dir", str(tmp_path / "dl"),
         "--revision", "v1"],
    )

    assert result.exit_code == 0, result.output
    assert [kw for name, kw in recorded_api if name == "snapshot_download"][0]["revision"] == "v1"
