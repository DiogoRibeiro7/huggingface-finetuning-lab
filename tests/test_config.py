import pytest

from hf_finetuning_lab.config import TrainingConfig


def test_training_config_validates_positive_epochs() -> None:
    config = TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        config.validate()


def test_training_config_to_dict_contains_model_name() -> None:
    config = TrainingConfig(model_name="distilbert-base-uncased")
    assert config.to_dict()["model_name"] == "distilbert-base-uncased"
