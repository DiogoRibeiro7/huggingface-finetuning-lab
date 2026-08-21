"""Publish a model artifact to the Hugging Face Hub, and pull one back.

Publication is the point where a local directory becomes something other people
run, so it is also the last point at which the artifact contracts can be
enforced. Three of them are checked before anything is uploaded:

* the artifact must satisfy the layout contract — an incomplete directory on the
  Hub is worse than no directory at all;
* the raw held-out rows must not be published. Training keeps them out of the
  artifact by default for exactly this reason, and a public repository is the
  scenario that motivated it;
* a repository id is mutable, so what was published is pinned by revision and
  recorded in the card rather than left implicit.

Staging is a branch, not a separate repository: push to a staging revision,
evaluate it, then promote that exact commit to the release revision. Promotion
moves a pointer to an already-reviewed commit, so what ships is what was
approved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from hf_finetuning_lab.artifacts import verify_artifact
from hf_finetuning_lab.data.heldout import HELDOUT_ROWS_FILENAME
from hf_finetuning_lab.model_cards.hub_card import (
    HUB_CARD_FILENAME,
    HubCardMetadata,
    write_hub_card,
)

#: Files never uploaded: local training scratch, and the raw evaluation rows.
NEVER_PUBLISH: tuple[str, ...] = (
    HELDOUT_ROWS_FILENAME,
    "trainer/*",
    "checkpoint-*/*",
    "*.log",
)

#: Default staging revision. Publication lands here first.
STAGING_REVISION = "staging"


class HubApi(Protocol):
    """The subset of ``huggingface_hub.HfApi`` this module uses."""

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any: ...
    def upload_folder(self, **kwargs: Any) -> Any: ...
    def create_branch(self, repo_id: str, **kwargs: Any) -> Any: ...
    def snapshot_download(self, repo_id: str, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class HubPublishConfig:
    """Where and how an artifact is published."""

    repo_id: str
    revision: str = STAGING_REVISION
    private: bool = False
    commit_message: str | None = None
    token: str | None = None

    def __post_init__(self) -> None:
        if "/" not in self.repo_id:
            raise ValueError(
                f"repo_id must be 'namespace/name', got {self.repo_id!r}."
            )
        if not self.revision:
            raise ValueError("revision must not be empty.")

    def message(self) -> str:
        return self.commit_message or f"Publish model artifact to {self.revision}"


@dataclass(slots=True)
class PublicationPlan:
    """What a publication would do. Returned by a dry run, and before upload."""

    repo_id: str
    revision: str
    private: bool
    files: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    blockers: tuple[str, ...] = field(default_factory=tuple)
    card: str = ""

    @property
    def publishable(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "private": self.private,
            "files": list(self.files),
            "excluded": list(self.excluded),
            "blockers": list(self.blockers),
            "publishable": self.publishable,
        }


def _matches(name: str, pattern: str) -> bool:
    from fnmatch import fnmatch

    return fnmatch(name, pattern)


def plan_publication(
    model_dir: str | Path,
    config: HubPublishConfig,
    metadata: HubCardMetadata | None = None,
    **card_kwargs: Any,
) -> PublicationPlan:
    """Decide what would be uploaded, and why it might be refused.

    Runs every check without contacting the Hub, so a dry run is a real
    rehearsal rather than a different code path.
    """
    directory = Path(model_dir)
    blockers: list[str] = []

    if not directory.is_dir():
        return PublicationPlan(
            repo_id=config.repo_id,
            revision=config.revision,
            private=config.private,
            blockers=(f"{directory} is not a directory",),
        )

    report = verify_artifact(directory)
    if not report.ok:
        blockers.append(
            "artifact is incomplete, missing: " + ", ".join(sorted(report.missing))
        )

    included: list[str] = []
    excluded: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if any(_matches(relative, pattern) for pattern in NEVER_PUBLISH):
            excluded.append(relative)
        else:
            included.append(relative)

    if not config.private and HELDOUT_ROWS_FILENAME in excluded:
        # Excluding it is not enough to be comfortable: its presence means the
        # run opted into keeping raw evaluation text, and a public repository is
        # the case that motivated keeping it out.
        blockers.append(
            f"{HELDOUT_ROWS_FILENAME} is present in the artifact; a public repository "
            "must not carry raw held-out rows. Retrain without "
            "persist_heldout_rows, or publish privately."
        )

    card = ""
    if metadata is not None:
        from hf_finetuning_lab.model_cards.hub_card import render_hub_card

        card = render_hub_card(metadata, **card_kwargs)

    return PublicationPlan(
        repo_id=config.repo_id,
        revision=config.revision,
        private=config.private,
        files=tuple(included),
        excluded=tuple(excluded),
        blockers=tuple(blockers),
        card=card,
    )


def publish_artifact(
    model_dir: str | Path,
    config: HubPublishConfig,
    metadata: HubCardMetadata | None = None,
    *,
    api: HubApi | None = None,
    dry_run: bool = False,
    **card_kwargs: Any,
) -> PublicationPlan:
    """Upload an artifact to ``config.revision`` of ``config.repo_id``.

    Returns the plan either way, so a dry run and a real publication report the
    same thing. Raises ``ValueError`` when the plan is blocked.
    """
    plan = plan_publication(model_dir, config, metadata, **card_kwargs)
    if not plan.publishable:
        raise ValueError(
            "Refusing to publish:\n" + "\n".join(f"  - {reason}" for reason in plan.blockers)
        )
    if dry_run:
        return plan

    client = api if api is not None else _default_api()
    directory = Path(model_dir)

    if metadata is not None:
        write_hub_card(directory, metadata, **card_kwargs)

    client.create_repo(
        repo_id=config.repo_id,
        private=config.private,
        exist_ok=True,
        token=config.token,
        repo_type="model",
    )
    _ensure_revision(client, config)
    client.upload_folder(
        repo_id=config.repo_id,
        folder_path=str(directory),
        revision=config.revision,
        commit_message=config.message(),
        token=config.token,
        repo_type="model",
        ignore_patterns=list(NEVER_PUBLISH),
    )
    return plan


def promote_revision(
    repo_id: str,
    source_revision: str,
    target_revision: str,
    *,
    api: HubApi | None = None,
    token: str | None = None,
) -> str:
    """Point ``target_revision`` at the commit already on ``source_revision``.

    Promotion never re-uploads: it moves a pointer to a commit that has already
    been evaluated, so the released weights are the reviewed ones.
    """
    if source_revision == target_revision:
        raise ValueError("source_revision and target_revision must differ.")
    client = api if api is not None else _default_api()
    client.create_branch(
        repo_id=repo_id,
        branch=target_revision,
        revision=source_revision,
        token=token,
        exist_ok=True,
        repo_type="model",
    )
    return target_revision


def pull_model(
    repo_id: str,
    destination: str | Path | None = None,
    *,
    revision: str = "main",
    api: HubApi | None = None,
    token: str | None = None,
) -> Path:
    """Download a published model at a pinned revision.

    With no destination the Hub cache is used, which is what a server wants:
    a restart reuses the existing download instead of refetching the weights.
    """
    client = api if api is not None else _default_api()
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "token": token,
        "repo_type": "model",
    }
    if destination is not None:
        kwargs["local_dir"] = str(destination)
    path = client.snapshot_download(**kwargs)
    return Path(path)


def metadata_from_artifact(
    model_dir: str | Path,
    model_name: str,
    *,
    license: str | None = None,
    datasets: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> HubCardMetadata:
    """Build card metadata from what the artifact already records.

    The training config, metrics and preprocessing contract are already in the
    directory, so the card is derived rather than re-typed at the command line
    — which is also what keeps it honest about the run it describes.
    """
    directory = Path(model_dir)

    def _load(name: str) -> dict[str, Any]:
        path = directory / name
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    training = _load("training_config.json")
    metrics_payload = _load("test_metrics.json")
    manifest = _load("heldout_manifest.json")

    from hf_finetuning_lab.model_cards.model_card import quality_metrics

    return HubCardMetadata(
        model_name=model_name,
        license=license,
        base_model=training.get("model_name"),
        datasets=list(datasets),
        tags=list(tags),
        metrics=quality_metrics(metrics_payload),
        dataset_fingerprint=manifest.get("fingerprint"),
    )


def write_publication_record(plan: PublicationPlan, output_path: str | Path) -> Path:
    """Persist what was published, for the run record."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return destination


def _ensure_revision(client: HubApi, config: HubPublishConfig) -> None:
    """Create the target branch when it does not exist yet."""
    if config.revision in {"main", "master"}:
        return
    client.create_branch(
        repo_id=config.repo_id,
        branch=config.revision,
        token=config.token,
        exist_ok=True,
        repo_type="model",
    )


def _default_api() -> HubApi:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "Install `huggingface-hub` to publish to the Hub."
        ) from exc
    # HfApi's signatures are keyword-only and wider than the subset declared
    # above. Every call site here passes keywords, so the narrowing is safe;
    # the Protocol documents what this module actually depends on.
    return cast(HubApi, HfApi())


def hub_card_filename() -> str:
    """The filename the Hub renders as the model card."""
    return HUB_CARD_FILENAME


def default_never_publish() -> Sequence[str]:
    """Patterns excluded from every upload."""
    return NEVER_PUBLISH
