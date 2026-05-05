from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for a text-classification fine-tuning run."""

    model_name: str = "distilbert-base-uncased"
    text_col: str = "text"
    label_col: str = "label"
    max_length: int = 160
    test_size: float = 0.2
    validation_size: float = 0.1
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    metric_for_best_model: str = "f1"
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate user-provided configuration values."""
        if not self.model_name:
            raise ValueError("model_name must not be empty.")
        if not self.text_col:
            raise ValueError("text_col must not be empty.")
        if not self.label_col:
            raise ValueError("label_col must not be empty.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 <= self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = cls(**payload)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable configuration dictionary."""
        return {
            "model_name": self.model_name,
            "text_col": self.text_col,
            "label_col": self.label_col,
            "max_length": self.max_length,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "metric_for_best_model": self.metric_for_best_model,
            "seed": self.seed,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
        }
