"""Resolving mutable names into the commits behind them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hf_finetuning_lab.provenance import (
    PROVENANCE_FILENAME,
    ArtifactProvenance,
    capture_provenance,
    load_provenance,
    resolve_repo_revision,
    write_provenance,
)

COMMIT = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def test_round_trips_through_the_artifact(tmp_path: Path) -> None:
    written = write_provenance(
        tmp_path, ArtifactProvenance(base_model="distilbert-base-uncased", base_model_revision=COMMIT)
    )

    assert written.name == PROVENANCE_FILENAME
    assert load_provenance(tmp_path).base_model_revision == COMMIT


def test_unset_fields_are_not_written(tmp_path: Path) -> None:
    """An empty key reads as 'unknown', which is not the same as 'absent'."""
    write_provenance(tmp_path, ArtifactProvenance(base_model="x"))

    payload = json.loads((tmp_path / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert payload == {"base_model": "x"}


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert load_provenance(tmp_path) == ArtifactProvenance()


def test_unparseable_file_reads_as_empty(tmp_path: Path) -> None:
    (tmp_path / PROVENANCE_FILENAME).write_text("{not json", encoding="utf-8")

    assert load_provenance(tmp_path) == ArtifactProvenance()


def test_a_local_checkpoint_has_no_hub_revision(tmp_path: Path) -> None:
    assert resolve_repo_revision(str(tmp_path)) is None


def test_lookup_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline, private or rate-limited: a run must not fail over metadata."""

    class _Api:
        def model_info(self, repo_id: str) -> Any:
            raise RuntimeError("no network")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    assert resolve_repo_revision("owner/name") is None


def test_resolution_records_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Info:
        sha = COMMIT

    class _Api:
        def model_info(self, repo_id: str) -> Any:
            return _Info()

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    assert resolve_repo_revision("owner/name") == COMMIT


def test_capture_skips_resolution_when_asked() -> None:
    captured = capture_provenance(base_model="owner/name", resolve=False)

    assert captured.base_model == "owner/name"
    assert captured.base_model_revision is None


def test_capture_records_the_source_commit() -> None:
    captured = capture_provenance(base_model=None, resolve=False)

    # None only when the tree is not a git checkout.
    assert captured.source_commit is None or len(captured.source_commit) == 40


def test_card_metadata_prefers_recorded_provenance(tmp_path: Path) -> None:
    """The card should show the commit, not just the mutable name."""
    from hf_finetuning_lab.publishing import metadata_from_artifact

    (tmp_path / "training_config.json").write_text(
        '{"model_name": "distilbert-base-uncased"}', encoding="utf-8"
    )
    write_provenance(
        tmp_path,
        ArtifactProvenance(
            base_model="distilbert-base-uncased",
            base_model_revision=COMMIT,
            dataset_id="fancyzhx/ag_news",
            dataset_revision="b" * 40,
        ),
    )

    metadata = metadata_from_artifact(tmp_path, model_name="support")

    assert metadata.base_model_revision == COMMIT
    assert metadata.datasets == ["fancyzhx/ag_news"]
    assert metadata.dataset_revision == "b" * 40
