"""Lightweight experiment tracking: run IDs, dataset hashes, run records."""

from __future__ import annotations

import hashlib
import json
import secrets
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def make_run_id(prefix: str = "run") -> str:
    """Return a sortable, unique run identifier.

    The format is ``<prefix>-YYYYMMDDTHHMMSSZ-<6 hex>``. The timestamp prefix
    makes runs sort chronologically; the random suffix avoids collisions when
    multiple runs are launched in the same second.
    """
    if not prefix:
        raise ValueError("prefix must not be empty.")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"{prefix}-{timestamp}-{suffix}"


def hash_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Return a deterministic sha256 hex digest for the rows of ``df``.

    Only the listed columns are considered (defaults to all columns, sorted).
    Row order is preserved so two datasets with the same content but different
    orderings produce different digests; callers that want order-insensitive
    hashes should sort before calling this.
    """
    if columns is None:
        columns = sorted(df.columns.astype(str).tolist())
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not in dataframe: {missing}")

    payload = df[columns].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class RunRecord:
    """JSON-serialisable record describing one experiment run."""

    run_id: str
    task: str
    model_name: str
    dataset_hash: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] | None = None
    notes: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRecord:
        """Rebuild a :class:`RunRecord` from a dictionary.

        Unknown keys are ignored so a record written by a newer version (with
        extra fields) still loads instead of raising ``TypeError``.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


def save_run(record: RunRecord, runs_dir: str | Path) -> Path:
    """Persist a run record to ``runs_dir/<run_id>.json`` and return the path."""
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{record.run_id}.json"
    destination.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_runs(runs_dir: str | Path) -> list[RunRecord]:
    """Load every ``*.json`` run record under ``runs_dir`` sorted by creation time."""
    directory = Path(runs_dir)
    if not directory.exists():
        return []

    records: list[RunRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(RunRecord.from_dict(payload))
        except (json.JSONDecodeError, TypeError, ValueError):
            # Skip unparseable/foreign JSON rather than failing the whole load.
            warnings.warn(f"Skipping unreadable run record: {path}", stacklevel=2)
            continue
    records.sort(key=lambda r: r.created_at_utc)
    return records


def runs_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    """Flatten run records into a comparison DataFrame.

    ``params`` keys become ``param_<key>`` columns and ``metrics`` keys become
    ``metric_<key>`` columns, so the resulting frame is easy to filter, sort,
    and join with other experiment outputs.
    """
    if not records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "run_id": record.run_id,
            "created_at_utc": record.created_at_utc,
            "task": record.task,
            "model_name": record.model_name,
            "dataset_hash": record.dataset_hash,
            "notes": record.notes,
        }
        for key, value in record.params.items():
            row[f"param_{key}"] = value
        for key, value in record.metrics.items():
            row[f"metric_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
