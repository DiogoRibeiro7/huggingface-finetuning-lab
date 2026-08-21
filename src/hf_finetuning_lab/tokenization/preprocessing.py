"""Preprocessing contract shared by training and inference.

Training and inference must agree on how text becomes model input. When they
disagree — most commonly on ``max_length`` — the model sees inputs at
inference that it never saw during training, and the resulting metrics no
longer describe the deployed behaviour.

Training writes this contract next to the weights; the predictor reads it back
and applies it. Keeping it in its own file (rather than inferring it from the
training config) leaves room for future settings such as truncation side,
padding side, or prompt templates without changing the artifact layout again.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Artifact filename holding the persisted contract.
PREPROCESSING_FILENAME = "preprocessing.json"


@dataclass(slots=True, frozen=True)
class PreprocessingConfig:
    """How raw text is turned into tokenizer input."""

    max_length: int | None = None
    truncation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessingConfig:
        max_length = payload.get("max_length")
        return cls(
            max_length=int(max_length) if max_length is not None else None,
            truncation=bool(payload.get("truncation", True)),
        )

    def tokenizer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to apply when encoding text for this model."""
        kwargs: dict[str, Any] = {"truncation": self.truncation}
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        return kwargs


def write_preprocessing_config(model_dir: str | Path, config: PreprocessingConfig) -> Path:
    """Persist ``config`` into the model directory."""
    destination = Path(model_dir) / PREPROCESSING_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return destination


def load_preprocessing_config(model_dir: str | Path) -> PreprocessingConfig:
    """Read the contract from a model directory.

    Falls back to ``training_config.json`` for artifacts produced before the
    contract was written separately, and to permissive defaults when neither
    file is present.
    """
    directory = Path(model_dir)
    explicit = directory / PREPROCESSING_FILENAME
    if explicit.exists():
        return PreprocessingConfig.from_dict(json.loads(explicit.read_text(encoding="utf-8")))

    legacy = directory / "training_config.json"
    if legacy.exists():
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        if payload.get("max_length") is not None:
            return PreprocessingConfig(max_length=int(payload["max_length"]))

    return PreprocessingConfig()
