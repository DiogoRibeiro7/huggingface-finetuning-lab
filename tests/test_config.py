import pytest

from hf_finetuning_lab.config import TrainingConfig


def test_training_config_validates_positive_epochs() -> None:
    config = TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        config.validate()


def test_training_config_to_dict_contains_model_name() -> None:
    config = TrainingConfig(model_name="distilbert-base-uncased")
    assert config.to_dict()["model_name"] == "distilbert-base-uncased"


def test_validate_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        TrainingConfig(weight_decay=-0.1).validate()


def test_validate_rejects_an_empty_best_model_metric() -> None:
    with pytest.raises(ValueError, match="metric_for_best_model"):
        TrainingConfig(metric_for_best_model="").validate()


def test_validate_rejects_a_non_integer_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(seed="42").validate()  # type: ignore[arg-type]


def test_validate_rejects_a_zero_lora_rank() -> None:
    with pytest.raises(ValueError, match="lora_r"):
        TrainingConfig(use_lora=True, lora_r=0).validate()


def test_validate_rejects_a_zero_lora_alpha() -> None:
    with pytest.raises(ValueError, match="lora_alpha"):
        TrainingConfig(use_lora=True, lora_alpha=0).validate()


def test_validate_rejects_an_out_of_range_lora_dropout() -> None:
    with pytest.raises(ValueError, match="lora_dropout"):
        TrainingConfig(use_lora=True, lora_dropout=1.0).validate()


def test_validate_rejects_blank_lora_target_modules() -> None:
    with pytest.raises(ValueError, match="lora_target_modules"):
        TrainingConfig(use_lora=True, lora_target_modules=["q_lin", "  "]).validate()


def test_lora_settings_are_ignored_when_lora_is_off() -> None:
    """A stale rank in a shared config should not block a non-LoRA run."""
    TrainingConfig(use_lora=False, lora_r=0).validate()


def test_valid_lora_config_passes() -> None:
    TrainingConfig(use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05).validate()
