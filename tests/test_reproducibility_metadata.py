"""What the record must carry to actually reconstruct a run."""

from __future__ import annotations

import json

from hf_finetuning_lab.governance.reproducibility import (
    TRACKED_PACKAGES,
    ReproducibilityRecord,
    capture_environment,
)


def test_environment_pins_the_resolved_package_versions() -> None:
    """A dependency range is not a record of what ran."""
    environment = capture_environment()

    assert set(environment["packages"]) == set(TRACKED_PACKAGES)
    assert environment["packages"]["transformers"] is not None


def test_environment_records_the_full_git_commit() -> None:
    commit = capture_environment()["git_commit"]

    # Short hashes are not stable as a repository grows.
    assert commit is None or len(commit) == 40


def test_environment_describes_the_accelerator() -> None:
    accelerator = capture_environment()["accelerator"]

    assert "torch_available" in accelerator
    if accelerator["torch_available"]:
        assert "cuda_available" in accelerator


def test_environment_is_json_serialisable() -> None:
    json.dumps(capture_environment())


def test_record_carries_model_and_dataset_revisions() -> None:
    record = ReproducibilityRecord(
        run_id="r1",
        task="text-classification",
        seed=42,
        dataset_hash="abc123",
        model_name="distilbert-base-uncased",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        dataset_id="fancyzhx/ag_news",
        dataset_config="default",
        dataset_revision="c" * 40,
    )

    payload = record.to_dict()

    assert payload["model_revision"] == "a" * 40
    assert payload["dataset_id"] == "fancyzhx/ag_news"
    assert payload["dataset_revision"] == "c" * 40


def test_revisions_default_to_none_for_local_runs() -> None:
    record = ReproducibilityRecord(
        run_id="r1",
        task="text-classification",
        seed=1,
        dataset_hash="abc",
        model_name="./local-checkpoint",
    )

    assert record.model_revision is None
    assert record.dataset_id is None
