"""Typed serving configuration resolved from CLI arguments and environment.

The container image and Compose file configure the server through
``HF_LAB_*`` environment variables. Resolving them in one place keeps the CLI,
the app factory, and the deployment manifests describing the same thing —
previously the variables were declared in ``docker-compose.yml`` but never
read, so two of the three had no effect at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "HF_LAB_"

#: Values accepted as true for boolean environment variables.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _env_bool(raw: str, name: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}."
    )


def _looks_like_commit(revision: str) -> bool:
    """True for a full or abbreviated commit sha."""
    return len(revision) >= 7 and all(c in "0123456789abcdef" for c in revision.lower())


def _env_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


@dataclass(slots=True, frozen=True)
class ServingConfig:
    """Everything the serving process needs to start."""

    #: Local artifact directory. Mutually exclusive with ``model_repo_id``.
    model_dir: Path | None = None
    #: Hub repository to serve from, as ``owner/name``.
    model_repo_id: str | None = None
    #: Revision to pin. Serving a branch means the weights can change under a
    #: running deployment, so a commit sha or tag is what makes a deployment
    #: reproducible.
    model_revision: str = "main"
    model_version: str | None = None
    enable_metrics: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    #: Requests carrying more texts than this are rejected.
    max_texts_per_request: int = 256
    #: Requests carrying a longer single text than this are rejected. Bounds
    #: the tokenization work one caller can queue onto the model.
    max_chars_per_text: int = 20_000

    def __post_init__(self) -> None:
        if self.model_dir is None and self.model_repo_id is None:
            raise ValueError("Give either a model directory or a Hub repository id.")
        if self.model_dir is not None and self.model_repo_id is not None:
            raise ValueError(
                "model_dir and model_repo_id are mutually exclusive; pick one source."
            )
        if self.model_repo_id is not None and "/" not in self.model_repo_id:
            raise ValueError(
                f"model_repo_id must be 'namespace/name', got {self.model_repo_id!r}."
            )
        if not self.model_revision:
            raise ValueError("model_revision must not be empty.")
        if self.port <= 0:
            raise ValueError("port must be positive.")
        if self.max_texts_per_request <= 0:
            raise ValueError("max_texts_per_request must be positive.")
        if self.max_chars_per_text <= 0:
            raise ValueError("max_chars_per_text must be positive.")

    @property
    def serves_a_moving_target(self) -> bool:
        """True when the configured source can change under a running server.

        A branch name resolves to whatever it points at today. A commit sha, or
        a local directory, does not.
        """
        if self.model_repo_id is None:
            return False
        return self.model_revision in {"main", "master"} or not _looks_like_commit(
            self.model_revision
        )

    @classmethod
    def from_env(
        cls,
        *,
        model_dir: str | Path | None = None,
        model_repo_id: str | None = None,
        model_revision: str | None = None,
        host: str | None = None,
        port: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ServingConfig:
        """Build a config from the environment, with explicit arguments winning.

        Passing ``model_dir``, ``host`` or ``port`` overrides the corresponding
        variable, so a CLI flag always beats the environment.
        """
        source = os.environ if env is None else env

        resolved_dir = model_dir if model_dir is not None else source.get(f"{ENV_PREFIX}MODEL_DIR")
        repo_id = model_repo_id if model_repo_id is not None else source.get(f"{ENV_PREFIX}MODEL_REPO")
        revision = (
            model_revision
            if model_revision is not None
            else source.get(f"{ENV_PREFIX}MODEL_REVISION") or "main"
        )
        if resolved_dir is None and repo_id is None:
            raise ValueError(
                f"No model source given. Pass --model-dir or --model-repo, or set "
                f"{ENV_PREFIX}MODEL_DIR or {ENV_PREFIX}MODEL_REPO."
            )
        if resolved_dir is not None and repo_id is not None:
            raise ValueError(
                f"{ENV_PREFIX}MODEL_DIR and {ENV_PREFIX}MODEL_REPO are mutually exclusive; "
                "pick one source."
            )

        version = source.get(f"{ENV_PREFIX}MODEL_VERSION") or None
        metrics_raw = source.get(f"{ENV_PREFIX}ENABLE_METRICS")
        texts_raw = source.get(f"{ENV_PREFIX}MAX_TEXTS_PER_REQUEST")
        chars_raw = source.get(f"{ENV_PREFIX}MAX_CHARS_PER_TEXT")
        env_port = source.get(f"{ENV_PREFIX}PORT")
        env_host = source.get(f"{ENV_PREFIX}HOST")

        defaults = cls(model_dir=Path("."))
        return cls(
            model_dir=Path(resolved_dir) if resolved_dir is not None else None,
            model_repo_id=repo_id,
            model_revision=revision,
            model_version=version,
            enable_metrics=(
                _env_bool(metrics_raw, f"{ENV_PREFIX}ENABLE_METRICS")
                if metrics_raw is not None
                else defaults.enable_metrics
            ),
            host=host if host is not None else (env_host or defaults.host),
            port=(
                port
                if port is not None
                else (_env_int(env_port, f"{ENV_PREFIX}PORT") if env_port else defaults.port)
            ),
            max_texts_per_request=(
                _env_int(texts_raw, f"{ENV_PREFIX}MAX_TEXTS_PER_REQUEST")
                if texts_raw
                else defaults.max_texts_per_request
            ),
            max_chars_per_text=(
                _env_int(chars_raw, f"{ENV_PREFIX}MAX_CHARS_PER_TEXT")
                if chars_raw
                else defaults.max_chars_per_text
            ),
        )
