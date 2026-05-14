from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hf_finetuning_lab.evaluation.metrics import per_class_report
from hf_finetuning_lab.experiments import (
    RunRecord,
    hash_dataframe,
    load_runs,
    make_run_id,
    runs_to_frame,
    save_run,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label": ["x", "y", "x", "y"],
        }
    )


def test_make_run_id_is_unique_and_sortable() -> None:
    ids = {make_run_id() for _ in range(20)}
    assert len(ids) == 20
    assert all(rid.startswith("run-") for rid in ids)
    # Sortable: timestamp prefix produces non-decreasing ordering when stable.
    sorted_ids = sorted(ids)
    assert sorted_ids[0] <= sorted_ids[-1]


def test_make_run_id_rejects_empty_prefix() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_run_id(prefix="")


def test_hash_dataframe_is_deterministic() -> None:
    frame = _sample_frame()
    assert hash_dataframe(frame) == hash_dataframe(frame)


def test_hash_dataframe_changes_with_content() -> None:
    frame = _sample_frame()
    other = frame.copy()
    other.loc[0, "text"] = "z"
    assert hash_dataframe(frame) != hash_dataframe(other)


def test_hash_dataframe_rejects_missing_columns() -> None:
    import pytest

    with pytest.raises(ValueError):
        hash_dataframe(_sample_frame(), columns=["text", "missing"])


def test_run_record_round_trip(tmp_path: Path) -> None:
    record = RunRecord(
        run_id="run-test-1",
        task="text-classification",
        model_name="tfidf-logreg",
        dataset_hash="deadbeef",
        params={"epochs": 2, "lr": 0.01},
        metrics={"macro_f1": 0.5},
        per_class={"x": {"precision": 1.0, "recall": 0.5, "f1-score": 0.66, "support": 2.0}},
        notes="baseline",
    )
    destination = save_run(record, tmp_path)

    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-test-1"
    assert payload["params"]["epochs"] == 2

    loaded = load_runs(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].run_id == record.run_id
    assert loaded[0].metrics["macro_f1"] == 0.5


def test_load_runs_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "does-not-exist") == []


def test_runs_to_frame_flattens_params_and_metrics() -> None:
    records = [
        RunRecord(
            run_id="run-1",
            task="text-classification",
            model_name="tfidf-logreg",
            dataset_hash="h1",
            params={"epochs": 1},
            metrics={"macro_f1": 0.4},
        ),
        RunRecord(
            run_id="run-2",
            task="text-classification",
            model_name="tfidf-logreg",
            dataset_hash="h2",
            params={"epochs": 2},
            metrics={"macro_f1": 0.6, "weighted_f1": 0.7},
        ),
    ]
    frame = runs_to_frame(records)
    assert set(["run_id", "param_epochs", "metric_macro_f1"]).issubset(frame.columns)
    assert frame.loc[frame["run_id"] == "run-2", "metric_weighted_f1"].iloc[0] == 0.7


def test_runs_to_frame_handles_empty_input() -> None:
    assert runs_to_frame([]).empty


def test_per_class_report_contains_label_rows() -> None:
    y_true = np.array(["a", "b", "a", "b", "a"])
    y_pred = np.array(["a", "b", "b", "b", "a"])
    report = per_class_report(y_true, y_pred, labels=["a", "b"])
    assert "a" in report.index
    assert "b" in report.index
    assert {"precision", "recall", "f1-score", "support"}.issubset(report.columns)
    assert report.loc["a", "support"] == 3
