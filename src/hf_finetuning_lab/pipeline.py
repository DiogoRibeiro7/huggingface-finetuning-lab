from __future__ import annotations

from pathlib import Path

from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.training.trainer import train_text_classifier


def run_training_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    text_col: str = "text",
    label_col: str = "label",
    epochs: int = 2,
    batch_size: int = 16,
    use_lora: bool = False,
) -> Path:
    """Run the high-level training pipeline."""
    config = TrainingConfig(
        model_name=model_name,
        text_col=text_col,
        label_col=label_col,
        epochs=epochs,
        batch_size=batch_size,
        use_lora=use_lora,
    )
    return train_text_classifier(input_path=input_path, output_dir=output_dir, config=config)
