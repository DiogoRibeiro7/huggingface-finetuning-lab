"""Reproducibility checklist + environment capture for one model run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_head_commit(cwd: Path | None = None) -> str | None:
    """Return the current git HEAD short hash, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _git_is_dirty(cwd: Path | None = None) -> bool | None:
    """Return ``True`` if the working tree has uncommitted changes, ``None`` on error."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return bool(result.stdout.strip())


def capture_environment(cwd: Path | None = None) -> dict[str, Any]:
    """Snapshot a small, JSON-serialisable picture of the runtime environment."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "captured_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_head_commit(cwd),
        "git_dirty": _git_is_dirty(cwd),
    }


@dataclass(slots=True)
class ReproducibilityRecord:
    """All the metadata required to rerun an experiment."""

    run_id: str
    task: str
    seed: int
    dataset_hash: str
    model_name: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bool(name: str, ok: bool) -> str:
    return f"- [{'x' if ok else ' '}] {name}"


def write_reproducibility_checklist(
    record: ReproducibilityRecord,
    output_path: str | Path,
) -> Path:
    """Write a Markdown reproducibility checklist alongside the raw JSON record."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_lines = (
        "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(record.config.items()))
        if record.config
        else "_No training config recorded._"
    )
    metric_lines = (
        "\n".join(f"- `{key}`: {value:.4f}" for key, value in sorted(record.metrics.items()))
        if record.metrics
        else "_No metrics recorded._"
    )
    env = record.environment or {}
    checklist = "\n".join(
        [
            _bool("Run ID recorded", bool(record.run_id)),
            _bool("Seed recorded", record.seed is not None),
            _bool("Dataset hash recorded", bool(record.dataset_hash)),
            _bool("Model name / version recorded", bool(record.model_name)),
            _bool("Training config recorded", bool(record.config)),
            _bool("Evaluation metrics recorded", bool(record.metrics)),
            _bool("Python version captured", bool(env.get("python_version"))),
            _bool("Platform captured", bool(env.get("platform"))),
            _bool("Git commit captured", env.get("git_commit") is not None),
            _bool(
                "Git working tree clean",
                env.get("git_dirty") is False,
            ),
        ]
    )

    body = f"""# Reproducibility Checklist: {record.run_id}

## Overview

- **Run ID:** `{record.run_id}`
- **Task:** {record.task}
- **Model:** `{record.model_name}`
- **Seed:** `{record.seed}`
- **Dataset hash:** `{record.dataset_hash}`

## Environment

- **Python:** `{env.get("python_version", "unknown")}`
- **Platform:** `{env.get("platform", "unknown")}`
- **Captured at (UTC):** `{env.get("captured_at_utc", "unknown")}`
- **Git commit:** `{env.get("git_commit", "unknown")}`
- **Git dirty:** `{env.get("git_dirty", "unknown")}`

## Configuration

{config_lines}

## Metrics

{metric_lines}

## Checklist

{checklist}

## Notes

{record.notes or "_No notes._"}
"""
    destination.write_text(body.strip() + "\n", encoding="utf-8")
    json_path = destination.with_suffix(".json")
    json_path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return destination
