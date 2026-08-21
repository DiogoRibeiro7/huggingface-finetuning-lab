"""High-level entry point for a full training run."""

from __future__ import annotations

from pathlib import Path

from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.training.trainer import train_text_classifier


def run_training_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
) -> Path:
    """Run the high-level training pipeline.

    Takes a :class:`TrainingConfig` rather than re-declaring a subset of its
    fields, so there is a single configuration surface to validate and extend.
    Defaults to ``TrainingConfig()`` when none is given.
    """
    return train_text_classifier(
        input_path=input_path,
        output_dir=output_dir,
        config=config or TrainingConfig(),
    )
