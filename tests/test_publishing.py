"""Publication to the Hub, with the Hub client mocked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hf_finetuning_lab.artifacts import (
    ALTERNATIVE_REQUIRED_FILES,
    RECOMMENDED_FILES,
    REQUIRED_FILES,
)
from hf_finetuning_lab.model_cards.hub_card import HubCardMetadata
from hf_finetuning_lab.publishing import (
    STAGING_REVISION,
    HubPublishConfig,
    plan_publication,
    promote_revision,
    publish_artifact,
    pull_model,
    write_publication_record,
)


class FakeHubApi:
    """Records calls instead of contacting the Hub."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, **kwargs: Any) -> str:
        self.calls.append(("create_repo", kwargs))
        return kwargs.get("repo_id", "")

    def upload_folder(self, **kwargs: Any) -> str:
        self.calls.append(("upload_folder", kwargs))
        return "commit-sha"

    def create_branch(self, **kwargs: Any) -> None:
        self.calls.append(("create_branch", kwargs))

    def snapshot_download(self, **kwargs: Any) -> str:
        self.calls.append(("snapshot_download", kwargs))
        return kwargs.get("local_dir", "")

    def named(self, name: str) -> list[dict[str, Any]]:
        return [payload for call, payload in self.calls if call == name]


def _artifact(path: Path, *, with_heldout_rows: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    for group in ALTERNATIVE_REQUIRED_FILES:
        (path / group[0]).write_text("weights", encoding="utf-8")
    for name in RECOMMENDED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    # Local training scratch that must never be uploaded.
    (path / "trainer").mkdir(exist_ok=True)
    (path / "trainer" / "optimizer.pt").write_text("scratch", encoding="utf-8")
    if with_heldout_rows:
        (path / "heldout_test.csv").write_text("text,label\na,b\n", encoding="utf-8")
    return path


def _metadata() -> HubCardMetadata:
    return HubCardMetadata(model_name="support-triage", metrics={"eval_f1": 0.9})


def test_repo_id_must_be_namespaced() -> None:
    with pytest.raises(ValueError, match="namespace/name"):
        HubPublishConfig(repo_id="support-triage")


def test_plan_lists_what_would_be_uploaded(tmp_path: Path) -> None:
    plan = plan_publication(_artifact(tmp_path), HubPublishConfig(repo_id="me/support"))

    assert plan.publishable
    assert "config.json" in plan.files
    assert "trainer/optimizer.pt" in plan.excluded


def test_incomplete_artifact_is_refused(tmp_path: Path) -> None:
    """An incomplete directory on the Hub is worse than no directory."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    plan = plan_publication(tmp_path, HubPublishConfig(repo_id="me/support"))

    assert not plan.publishable
    assert any("incomplete" in reason for reason in plan.blockers)


def test_public_publication_refuses_raw_heldout_rows(tmp_path: Path) -> None:
    """The privacy contract holds at the point it actually matters."""
    plan = plan_publication(
        _artifact(tmp_path, with_heldout_rows=True),
        HubPublishConfig(repo_id="me/support", private=False),
    )

    assert not plan.publishable
    assert any("held-out rows" in reason for reason in plan.blockers)


def test_private_publication_allows_heldout_rows(tmp_path: Path) -> None:
    plan = plan_publication(
        _artifact(tmp_path, with_heldout_rows=True),
        HubPublishConfig(repo_id="me/support", private=True),
    )

    assert plan.publishable


def test_publish_refuses_a_blocked_plan(tmp_path: Path) -> None:
    api = FakeHubApi()
    with pytest.raises(ValueError, match="Refusing to publish"):
        publish_artifact(
            _artifact(tmp_path, with_heldout_rows=True),
            HubPublishConfig(repo_id="me/support"),
            api=api,
        )

    assert api.calls == []


def test_dry_run_uploads_nothing(tmp_path: Path) -> None:
    api = FakeHubApi()

    plan = publish_artifact(
        _artifact(tmp_path), HubPublishConfig(repo_id="me/support"), _metadata(),
        api=api, dry_run=True,
    )

    assert plan.publishable
    assert api.calls == []
    # The dry run still renders the card it would publish.
    assert "# support-triage" in plan.card
    assert not (tmp_path / "README.md").exists()


def test_publish_creates_repo_branch_and_uploads(tmp_path: Path) -> None:
    api = FakeHubApi()

    publish_artifact(
        _artifact(tmp_path), HubPublishConfig(repo_id="me/support"), _metadata(), api=api
    )

    assert api.named("create_repo")[0]["repo_id"] == "me/support"
    upload = api.named("upload_folder")[0]
    assert upload["revision"] == STAGING_REVISION
    assert "heldout_test.csv" in upload["ignore_patterns"]
    # The card is written into the folder that gets uploaded.
    assert (tmp_path / "README.md").exists()


def test_publish_defaults_to_staging_not_main(tmp_path: Path) -> None:
    """Publication lands on staging so a release is a deliberate second step."""
    api = FakeHubApi()

    publish_artifact(_artifact(tmp_path), HubPublishConfig(repo_id="me/support"), api=api)

    assert api.named("upload_folder")[0]["revision"] == "staging"
    assert api.named("create_branch")[0]["branch"] == "staging"


def test_publishing_to_main_does_not_create_a_branch(tmp_path: Path) -> None:
    api = FakeHubApi()

    publish_artifact(
        _artifact(tmp_path), HubPublishConfig(repo_id="me/support", revision="main"), api=api
    )

    assert api.named("create_branch") == []


def test_private_flag_reaches_the_hub(tmp_path: Path) -> None:
    api = FakeHubApi()

    publish_artifact(
        _artifact(tmp_path), HubPublishConfig(repo_id="me/support", private=True), api=api
    )

    assert api.named("create_repo")[0]["private"] is True


def test_promote_points_the_target_at_the_reviewed_commit() -> None:
    api = FakeHubApi()

    promote_revision("me/support", "staging", "v1", api=api)

    branch = api.named("create_branch")[0]
    assert branch["branch"] == "v1"
    assert branch["revision"] == "staging"
    # Promotion moves a pointer; it never re-uploads.
    assert api.named("upload_folder") == []


def test_promote_rejects_a_no_op() -> None:
    with pytest.raises(ValueError, match="must differ"):
        promote_revision("me/support", "main", "main", api=FakeHubApi())


def test_pull_downloads_a_pinned_revision(tmp_path: Path) -> None:
    api = FakeHubApi()

    pull_model("me/support", tmp_path, revision="v1", api=api)

    assert api.named("snapshot_download")[0]["revision"] == "v1"


def test_publication_record_is_written(tmp_path: Path) -> None:
    plan = plan_publication(_artifact(tmp_path), HubPublishConfig(repo_id="me/support"))

    written = write_publication_record(plan, tmp_path / "publication.json")

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["repo_id"] == "me/support"
    assert payload["publishable"] is True
