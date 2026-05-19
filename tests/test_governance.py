from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_finetuning_lab.governance import (
    SUPPORTED_TASKS,
    DatasetCard,
    DatasetColumn,
    DatasetSplit,
    ReproducibilityRecord,
    capture_environment,
    task_limitations,
    write_dataset_card,
    write_reproducibility_checklist,
    write_task_model_card,
)


def test_supported_tasks_cover_repo_workflows() -> None:
    assert {"text-classification", "token-classification", "retrieval"}.issubset(SUPPORTED_TASKS)


def test_task_limitations_returned_for_known_tasks() -> None:
    for task in SUPPORTED_TASKS:
        bullets = task_limitations(task)
        assert bullets
        assert all(isinstance(item, str) and item for item in bullets)


def test_task_limitations_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        task_limitations("translation")


def test_write_task_model_card_appends_extra_limitations(tmp_path: Path) -> None:
    path = write_task_model_card(
        output_path=tmp_path / "card.md",
        model_name="tfidf-logreg",
        task="text-classification",
        label_names=["a", "b"],
        metrics={"macro_f1": 0.5},
        extra_limitations=["Synthetic labels only."],
    )
    text = path.read_text(encoding="utf-8")
    assert "Synthetic labels only." in text
    assert "robust-evaluation" in text  # the curated v0.4 limitation


def test_dataset_card_round_trip(tmp_path: Path) -> None:
    card = DatasetCard(
        name="synthetic-support-triage",
        description="Synthetic support tickets for triage demonstrations.",
        task="text-classification",
        columns=[
            DatasetColumn(name="text", dtype="str", role="feature", description="ticket body"),
            DatasetColumn(name="label", dtype="str", role="target", description="triage category"),
        ],
        splits=[
            DatasetSplit(
                name="train",
                n_rows=640,
                label_distribution={"account": 100, "billing": 540},
            ),
            DatasetSplit(name="test", n_rows=160),
        ],
        privacy_notes=["No real customer data; all rows are generated."],
        intended_use=["Workflow demonstrations and CI smoke."],
        not_intended_use=["Production triage decisions."],
        limitations=["Labels are synthetic and do not reflect any real taxonomy."],
        sources=["src/hf_finetuning_lab/sample_data.py"],
    )
    path = write_dataset_card(card, tmp_path / "dataset_card.md")
    text = path.read_text(encoding="utf-8")
    for needle in [
        "synthetic-support-triage",
        "text-classification",
        "ticket body",
        "Rows: **640**",
        "`account`: 100",
        "Production triage decisions",
        "src/hf_finetuning_lab/sample_data.py",
    ]:
        assert needle in text, f"missing fragment: {needle!r}"


def test_dataset_card_empty_sections_have_fallbacks(tmp_path: Path) -> None:
    card = DatasetCard(
        name="bare",
        description="bare-bones card",
        task="text-classification",
    )
    text = write_dataset_card(card, tmp_path / "card.md").read_text(encoding="utf-8")
    assert "No columns documented." in text
    assert "No splits documented." in text
    assert "No privacy notes" in text


def test_capture_environment_returns_python_version() -> None:
    env = capture_environment()
    assert "python_version" in env and env["python_version"]
    assert "platform" in env and env["platform"]
    # git_commit may be None when running outside a checkout; just check key is present.
    assert "git_commit" in env
    assert "git_dirty" in env


def test_write_reproducibility_checklist_writes_md_and_json(tmp_path: Path) -> None:
    record = ReproducibilityRecord(
        run_id="run-test-abc",
        task="text-classification",
        seed=42,
        dataset_hash="deadbeefcafebabe",
        model_name="tfidf-logreg",
        config={"epochs": 2, "lr": 0.01},
        metrics={"macro_f1": 0.42},
        environment={
            "python_version": "3.12.0",
            "platform": "Test-OS",
            "captured_at_utc": "2026-05-19T00:00:00+00:00",
            "git_commit": "abcdef1",
            "git_dirty": False,
        },
        notes="Test record.",
    )
    md_path = write_reproducibility_checklist(record, tmp_path / "repro.md")
    json_path = md_path.with_suffix(".json")
    text = md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "run-test-abc" in text
    assert "`deadbeefcafebabe`" in text
    assert "[x] Git working tree clean" in text  # git_dirty is False -> checkbox set
    assert "[x] Run ID recorded" in text
    assert payload["run_id"] == "run-test-abc"
    assert payload["metrics"]["macro_f1"] == pytest.approx(0.42)


def test_reproducibility_checklist_marks_dirty_tree_unchecked(tmp_path: Path) -> None:
    record = ReproducibilityRecord(
        run_id="run-dirty",
        task="text-classification",
        seed=0,
        dataset_hash="h",
        model_name="m",
        environment={"git_dirty": True, "git_commit": "abc", "python_version": "3.12.0"},
    )
    text = write_reproducibility_checklist(record, tmp_path / "repro.md").read_text(encoding="utf-8")
    assert "[ ] Git working tree clean" in text
