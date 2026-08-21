"""Serving a model straight from a pinned Hub revision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hf_finetuning_lab.serving.config import ServingConfig

COMMIT = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"


def test_a_hub_repository_is_a_valid_model_source() -> None:
    config = ServingConfig.from_env(
        env={"HF_LAB_MODEL_REPO": "me/support", "HF_LAB_MODEL_REVISION": COMMIT}
    )

    assert config.model_repo_id == "me/support"
    assert config.model_revision == COMMIT
    assert config.model_dir is None


def test_a_source_is_required() -> None:
    with pytest.raises(ValueError, match="MODEL_REPO"):
        ServingConfig.from_env(env={})


def test_the_two_sources_are_mutually_exclusive() -> None:
    """Serving one directory while claiming to serve a repository is ambiguous."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        ServingConfig.from_env(
            env={"HF_LAB_MODEL_DIR": "/models", "HF_LAB_MODEL_REPO": "me/support"}
        )


def test_repo_id_must_be_namespaced() -> None:
    with pytest.raises(ValueError, match="namespace/name"):
        ServingConfig(model_repo_id="support")


def test_a_commit_sha_is_not_a_moving_target() -> None:
    config = ServingConfig(model_repo_id="me/support", model_revision=COMMIT)

    assert config.serves_a_moving_target is False


@pytest.mark.parametrize("revision", ["main", "master", "staging"])
def test_a_branch_is_a_moving_target(revision: str) -> None:
    """A branch resolves to whatever it points at today."""
    config = ServingConfig(model_repo_id="me/support", model_revision=revision)

    assert config.serves_a_moving_target is True


def test_a_local_directory_is_not_a_moving_target(tmp_path: Path) -> None:
    assert ServingConfig(model_dir=tmp_path).serves_a_moving_target is False


def test_app_downloads_the_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The download happens at startup, not on the first request."""
    calls: list[dict[str, Any]] = []
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    def fake_pull(repo_id: str, destination: Any = None, **kwargs: Any) -> Path:
        calls.append({"repo_id": repo_id, **kwargs})
        return tmp_path

    monkeypatch.setattr("hf_finetuning_lab.publishing.pull_model", fake_pull)

    from hf_finetuning_lab.serving.api import create_app_from_config

    app = create_app_from_config(
        ServingConfig(model_repo_id="me/support", model_revision=COMMIT),
        predictor_factory=lambda _path: _StubPredictor(),
    )

    assert calls == [{"repo_id": "me/support", "revision": COMMIT}]
    assert app.state.model_dir == str(tmp_path)


def test_health_reports_what_is_being_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit version, the pinned source identifies the weights."""
    monkeypatch.setattr(
        "hf_finetuning_lab.publishing.pull_model", lambda *a, **kw: tmp_path
    )

    from hf_finetuning_lab.serving.api import create_app_from_config

    app = create_app_from_config(
        ServingConfig(model_repo_id="me/support", model_revision=COMMIT),
        predictor_factory=lambda _path: _StubPredictor(),
    )

    assert app.state.model_version == f"me/support@{COMMIT}"


def test_an_explicit_version_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hf_finetuning_lab.publishing.pull_model", lambda *a, **kw: tmp_path
    )

    from hf_finetuning_lab.serving.api import create_app_from_config

    app = create_app_from_config(
        ServingConfig(
            model_repo_id="me/support", model_revision=COMMIT, model_version="release-3"
        ),
        predictor_factory=lambda _path: _StubPredictor(),
    )

    assert app.state.model_version == "release-3"


def test_a_local_directory_is_not_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("a local directory must not trigger a download")

    monkeypatch.setattr("hf_finetuning_lab.publishing.pull_model", explode)

    from hf_finetuning_lab.serving.api import create_app_from_config

    app = create_app_from_config(
        ServingConfig(model_dir=tmp_path), predictor_factory=lambda _path: _StubPredictor()
    )

    assert app.state.model_dir == str(tmp_path)


class _StubPredictor:
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        return [{"text": t, "predicted_label": "a", "confidence": 1.0} for t in texts]
