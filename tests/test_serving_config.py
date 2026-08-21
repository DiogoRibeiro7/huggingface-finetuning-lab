from __future__ import annotations

from pathlib import Path

import pytest

from hf_finetuning_lab.serving.config import ServingConfig


def test_reads_the_compose_environment_variables() -> None:
    config = ServingConfig.from_env(
        env={
            "HF_LAB_MODEL_DIR": "/app/artifacts/models/support-triage",
            "HF_LAB_MODEL_VERSION": "1.0.0",
            "HF_LAB_ENABLE_METRICS": "true",
        }
    )

    assert config.model_dir == Path("/app/artifacts/models/support-triage")
    assert config.model_version == "1.0.0"
    assert config.enable_metrics is True


def test_explicit_arguments_beat_the_environment() -> None:
    config = ServingConfig.from_env(
        model_dir="/flag/path",
        port=9001,
        env={"HF_LAB_MODEL_DIR": "/env/path", "HF_LAB_PORT": "8000"},
    )

    assert config.model_dir == Path("/flag/path")
    assert config.port == 9001


def test_requires_a_model_directory_from_somewhere() -> None:
    with pytest.raises(ValueError, match="HF_LAB_MODEL_DIR"):
        ServingConfig.from_env(env={})


@pytest.mark.parametrize("raw,expected", [("1", True), ("no", False), ("On", True)])
def test_boolean_variables_accept_common_spellings(raw: str, expected: bool) -> None:
    config = ServingConfig.from_env(env={"HF_LAB_MODEL_DIR": "/m", "HF_LAB_ENABLE_METRICS": raw})

    assert config.enable_metrics is expected


def test_rejects_an_unparseable_boolean() -> None:
    with pytest.raises(ValueError, match="HF_LAB_ENABLE_METRICS"):
        ServingConfig.from_env(env={"HF_LAB_MODEL_DIR": "/m", "HF_LAB_ENABLE_METRICS": "maybe"})


def test_rejects_an_unparseable_port() -> None:
    with pytest.raises(ValueError, match="HF_LAB_PORT"):
        ServingConfig.from_env(env={"HF_LAB_MODEL_DIR": "/m", "HF_LAB_PORT": "http"})


def test_rejects_nonsense_limits() -> None:
    with pytest.raises(ValueError, match="max_chars_per_text"):
        ServingConfig(model_dir=Path("/m"), max_chars_per_text=0)


def test_defaults_are_conservative() -> None:
    config = ServingConfig.from_env(env={"HF_LAB_MODEL_DIR": "/m"})

    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.enable_metrics is False
    assert config.model_version is None
