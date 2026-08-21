"""Capture what a training run was actually built from.

A model name is not provenance. ``distilbert-base-uncased`` resolves to
whatever that repository points at today, so a run recorded by name alone
cannot be reconstructed later — the same command can produce different weights
next month, and nothing in the artifact would show it.

This module resolves the mutable names used by a run into the commits behind
them, and writes them next to the weights. Resolution is best effort: training
from a local checkpoint, or without network, records what is known and leaves
the rest unset rather than failing a run over metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Artifact filename holding the resolved provenance.
PROVENANCE_FILENAME = "provenance.json"


@dataclass(slots=True)
class ArtifactProvenance:
    """The immutable identifiers behind a run's mutable inputs."""

    base_model: str | None = None
    base_model_revision: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None
    dataset_fingerprint: str | None = None
    source_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactProvenance:
        fields = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in payload.items() if k in fields})


def resolve_repo_revision(repo_id: str, repo_type: str = "model") -> str | None:
    """Return the commit a Hub repository id currently points at.

    Returns ``None`` for a local path, or when the Hub cannot be reached — a
    run should not fail because provenance could not be looked up.
    """
    if not repo_id or "/" not in repo_id and Path(repo_id).exists():
        return None
    if Path(repo_id).exists():
        return None
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None

    api = HfApi()
    try:
        if repo_type == "dataset":
            return str(api.dataset_info(repo_id).sha)
        return str(api.model_info(repo_id).sha)
    except Exception:
        # Offline, private, renamed, rate-limited: all recoverable here.
        return None


def capture_provenance(
    *,
    base_model: str | None = None,
    dataset_id: str | None = None,
    dataset_fingerprint: str | None = None,
    source_commit: str | None = None,
    resolve: bool = True,
) -> ArtifactProvenance:
    """Resolve the run's inputs into commits where possible."""
    if source_commit is None:
        from hf_finetuning_lab.governance.reproducibility import _git_head_commit

        source_commit = _git_head_commit()

    return ArtifactProvenance(
        base_model=base_model,
        base_model_revision=(
            resolve_repo_revision(base_model) if resolve and base_model else None
        ),
        dataset_id=dataset_id,
        dataset_revision=(
            resolve_repo_revision(dataset_id, "dataset") if resolve and dataset_id else None
        ),
        dataset_fingerprint=dataset_fingerprint,
        source_commit=source_commit,
    )


def write_provenance(model_dir: str | Path, provenance: ArtifactProvenance) -> Path:
    """Persist provenance into a model directory."""
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / PROVENANCE_FILENAME
    destination.write_text(json.dumps(provenance.to_dict(), indent=2), encoding="utf-8")
    return destination


def load_provenance(model_dir: str | Path) -> ArtifactProvenance:
    """Read provenance from a model directory, empty when absent."""
    path = Path(model_dir) / PROVENANCE_FILENAME
    if not path.is_file():
        return ArtifactProvenance()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ArtifactProvenance()
    return ArtifactProvenance.from_dict(payload if isinstance(payload, dict) else {})
