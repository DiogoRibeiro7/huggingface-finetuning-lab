"""The Hub repository card and its YAML metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hf_finetuning_lab.model_cards.hub_card import (
    HUB_CARD_FILENAME,
    HubCardMetadata,
    render_hub_card,
    write_hub_card,
)


def _metadata(**overrides: object) -> HubCardMetadata:
    base: dict[str, object] = {
        "model_name": "support-triage",
        "license": "mit",
        "base_model": "distilbert-base-uncased",
        "base_model_revision": "a" * 40,
        "datasets": ["fancyzhx/ag_news"],
        "dataset_revision": "b" * 40,
        "dataset_fingerprint": "sha256:abc123",
        "source_commit": "c" * 40,
        "metrics": {"eval_accuracy": 0.9123, "eval_f1": 0.8899},
        "tags": ["fine-tuned"],
    }
    base.update(overrides)
    return HubCardMetadata(**base)  # type: ignore[arg-type]


def _frontmatter(card: str) -> dict:
    assert card.startswith("---\n")
    block = card.split("---\n", 2)[1]
    return yaml.safe_load(block)


def test_card_starts_with_a_yaml_block() -> None:
    """The Hub reads discoverability metadata from the leading YAML block."""
    card = render_hub_card(_metadata())

    front = _frontmatter(card)
    assert front["library_name"] == "transformers"
    assert front["pipeline_tag"] == "text-classification"
    assert front["license"] == "mit"
    assert front["base_model"] == "distilbert-base-uncased"
    assert front["datasets"] == ["fancyzhx/ag_news"]
    assert front["language"] == ["en"]


def test_task_is_always_present_as_a_tag() -> None:
    front = _frontmatter(render_hub_card(_metadata()))

    assert "text-classification" in front["tags"]
    assert "fine-tuned" in front["tags"]
    # Tags are de-duplicated even when the task is passed explicitly.
    assert len(front["tags"]) == len(set(front["tags"]))


def test_metrics_render_as_a_model_index() -> None:
    """model-index is what makes metrics show on the Hub listing."""
    front = _frontmatter(render_hub_card(_metadata()))

    results = front["model-index"][0]["results"][0]
    assert results["task"]["type"] == "text-classification"
    assert results["dataset"]["type"] == "fancyzhx/ag_news"
    values = {m["type"]: m["value"] for m in results["metrics"]}
    assert values == {"accuracy": 0.9123, "f1": 0.8899}


def test_metric_keys_are_normalised() -> None:
    """`eval_` is a Trainer prefix, not part of the metric's name."""
    front = _frontmatter(render_hub_card(_metadata()))

    assert front["metrics"] == ["accuracy", "f1"]


def test_model_index_is_omitted_without_metrics() -> None:
    front = _frontmatter(render_hub_card(_metadata(metrics={})))

    assert "model-index" not in front
    assert "metrics" not in front


def test_body_records_the_revisions_behind_the_ids() -> None:
    card = render_hub_card(_metadata())

    assert "a" * 40 in card
    assert "b" * 40 in card
    assert "c" * 40 in card
    assert "sha256:abc123" in card


def test_provenance_omits_fields_that_were_not_captured() -> None:
    card = render_hub_card(
        _metadata(base_model_revision=None, dataset_revision=None, dataset_fingerprint=None)
    )

    assert "Base model revision" not in card
    assert "Base model" in card


def test_optional_sections_render_when_supplied() -> None:
    card = render_hub_card(
        _metadata(),
        summary="Routes support tickets.",
        intended_use=["Triage inbound tickets."],
        limitations=["Not validated for safety-critical use."],
        training_config={"epochs": 2, "max_length": 160},
    )

    assert "Routes support tickets." in card
    assert "Triage inbound tickets." in card
    assert "Not validated for safety-critical use." in card
    assert "max_length: 160" in card


def test_write_hub_card_writes_readme(tmp_path: Path) -> None:
    """The Hub renders README.md, not model_card.md."""
    written = write_hub_card(tmp_path, _metadata())

    assert written == tmp_path / HUB_CARD_FILENAME
    assert written.name == "README.md"
    assert _frontmatter(written.read_text(encoding="utf-8"))["license"] == "mit"


def test_card_is_valid_when_only_a_name_is_known() -> None:
    card = render_hub_card(HubCardMetadata(model_name="bare"))

    front = _frontmatter(card)
    assert front["library_name"] == "transformers"
    assert "# bare" in card


def test_card_parses_with_the_hub_client() -> None:
    """Validated by the library that consumes it, not just by our own assertions.

    ``eval_results`` being populated means the ``model-index`` block is
    structurally valid, which is what makes the Hub render the metrics table.
    """
    hub = pytest.importorskip("huggingface_hub")

    parsed = hub.ModelCard(render_hub_card(_metadata()))

    assert parsed.data.pipeline_tag == "text-classification"
    assert parsed.data.library_name == "transformers"
    assert parsed.data.eval_results is not None
    assert {result.metric_type for result in parsed.data.eval_results} == {"accuracy", "f1"}
