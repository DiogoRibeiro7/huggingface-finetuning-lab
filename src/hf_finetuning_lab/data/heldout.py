"""Privacy-safe description of the held-out test split.

A model directory travels: it gets copied between machines, attached to a
release, or pushed to the Hub. Writing the raw held-out rows into it means the
underlying text travels too, which for a real dataset can mean customer
messages, personal data, licensed benchmark content, or private annotations.

The manifest records everything needed to *identify and reproduce* the split —
its size, label balance, seed, and a fingerprint plus per-row hashes — without
carrying the text itself. Hashes are one-way, so a holder of the manifest can
confirm which rows were used but cannot recover them.

Persisting the rows themselves stays available through
``TrainingConfig.persist_heldout_rows`` for synthetic or already-public data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from hf_finetuning_lab.experiments.runs import hash_dataframe

#: Artifact filename holding the manifest.
HELDOUT_MANIFEST_FILENAME = "heldout_manifest.json"

#: Filename used when raw rows are explicitly opted into.
HELDOUT_ROWS_FILENAME = "heldout_test.csv"


def row_id(text: str) -> str:
    """Return a stable, non-reversible identifier for one row of text."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def build_heldout_manifest(
    frame: pd.DataFrame,
    *,
    text_col: str,
    label_col: str,
    seed: int,
    split: str = "test",
) -> dict[str, Any]:
    """Describe a held-out split without copying its contents."""
    for column in (text_col, label_col):
        if column not in frame.columns:
            raise ValueError(f"Held-out frame is missing column '{column}'.")

    counts = frame[label_col].value_counts().sort_index()
    return {
        "split": split,
        "seed": seed,
        "n_rows": int(len(frame)),
        "columns": {"text": text_col, "label": label_col},
        "label_distribution": {str(label): int(count) for label, count in counts.items()},
        "fingerprint": hash_dataframe(frame[[text_col, label_col]]),
        "row_ids": [row_id(value) for value in frame[text_col]],
        "rows_persisted": False,
    }


def write_heldout_manifest(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    text_col: str,
    label_col: str,
    seed: int,
    persist_rows: bool = False,
) -> Path:
    """Write the manifest, and the raw rows only when explicitly requested."""
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = build_heldout_manifest(
        frame, text_col=text_col, label_col=label_col, seed=seed
    )
    manifest["rows_persisted"] = bool(persist_rows)

    if persist_rows:
        frame.to_csv(directory / HELDOUT_ROWS_FILENAME, index=False)

    destination = directory / HELDOUT_MANIFEST_FILENAME
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination
