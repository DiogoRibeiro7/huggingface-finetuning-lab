"""The held-out split is described, not copied, unless asked."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hf_finetuning_lab.data.heldout import (
    HELDOUT_MANIFEST_FILENAME,
    HELDOUT_ROWS_FILENAME,
    build_heldout_manifest,
    row_id,
    write_heldout_manifest,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["my card was charged twice", "cannot log in", "package never arrived"],
            "label": ["billing", "account", "delivery"],
            "label_id": [1, 0, 2],
        }
    )


def test_manifest_describes_the_split() -> None:
    manifest = build_heldout_manifest(_frame(), text_col="text", label_col="label", seed=42)

    assert manifest["n_rows"] == 3
    assert manifest["seed"] == 42
    assert manifest["label_distribution"] == {"account": 1, "billing": 1, "delivery": 1}
    assert len(manifest["row_ids"]) == 3


def test_manifest_does_not_carry_the_text(tmp_path: Path) -> None:
    """The whole point: the artifact identifies rows without redistributing them."""
    frame = _frame()
    write_heldout_manifest(frame, tmp_path, text_col="text", label_col="label", seed=42)

    written = (tmp_path / HELDOUT_MANIFEST_FILENAME).read_text(encoding="utf-8")

    for text in frame["text"]:
        assert text not in written
    assert not (tmp_path / HELDOUT_ROWS_FILENAME).exists()


def test_row_ids_identify_rows_without_revealing_them() -> None:
    manifest = build_heldout_manifest(_frame(), text_col="text", label_col="label", seed=1)

    assert manifest["row_ids"][0] == row_id("my card was charged twice")
    assert row_id("my card was charged twice") != "my card was charged twice"


def test_raw_rows_are_written_only_when_opted_into(tmp_path: Path) -> None:
    write_heldout_manifest(
        _frame(), tmp_path, text_col="text", label_col="label", seed=42, persist_rows=True
    )

    assert (tmp_path / HELDOUT_ROWS_FILENAME).exists()
    payload = json.loads((tmp_path / HELDOUT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["rows_persisted"] is True


def test_fingerprint_changes_with_the_content() -> None:
    original = build_heldout_manifest(_frame(), text_col="text", label_col="label", seed=1)
    altered_frame = _frame()
    altered_frame.loc[0, "text"] = "something else entirely"
    altered = build_heldout_manifest(altered_frame, text_col="text", label_col="label", seed=1)

    assert original["fingerprint"] != altered["fingerprint"]


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing column 'body'"):
        build_heldout_manifest(_frame(), text_col="body", label_col="label", seed=1)
