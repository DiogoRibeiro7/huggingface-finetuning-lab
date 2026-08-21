"""Hub-native model card: a repository ``README.md`` with YAML metadata.

A Hugging Face model repository renders its ``README.md`` as the model card, and
reads the YAML block at the top for discoverability — the task filter, the
dataset and base-model links, and the metrics table all come from there.

That is a different artifact from ``model_card.md``, which is an internal report
about a training run. Both are kept: the report describes what happened, the
repository card describes what the published model *is*.

The metadata block is also where provenance becomes machine-readable. A model
name alone is mutable, so the card records the base model and dataset revisions
that produced these weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Filename the Hub renders as the model card.
HUB_CARD_FILENAME = "README.md"

#: Metric keys the Hub understands in a ``model-index`` results block.
_KNOWN_METRIC_TYPES = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC AUC",
    "loss": "Loss",
}


def _metric_type(name: str) -> str:
    """Normalise a Trainer metric key to a Hub metric type."""
    return name.removeprefix("eval_").removeprefix("test_").removeprefix("train_")


@dataclass(slots=True)
class HubCardMetadata:
    """The YAML block at the top of a Hub model repository README."""

    model_name: str
    task: str = "text-classification"
    library_name: str = "transformers"
    license: str | None = None
    language: Sequence[str] = ("en",)
    base_model: str | None = None
    datasets: Sequence[str] = ()
    tags: Sequence[str] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    #: Revisions that pin what a mutable id points at.
    base_model_revision: str | None = None
    dataset_revision: str | None = None
    dataset_fingerprint: str | None = None
    source_commit: str | None = None

    def to_frontmatter(self) -> dict[str, Any]:
        """Build the YAML mapping the Hub reads."""
        block: dict[str, Any] = {
            "library_name": self.library_name,
            "pipeline_tag": self.task,
        }
        if self.language:
            block["language"] = list(self.language)
        if self.license:
            block["license"] = self.license
        if self.base_model:
            block["base_model"] = self.base_model
        if self.datasets:
            block["datasets"] = list(self.datasets)

        tags = list(dict.fromkeys([*self.tags, self.task]))
        block["tags"] = tags

        if self.metrics:
            block["metrics"] = sorted({_metric_type(name) for name in self.metrics})
            block["model-index"] = [
                {
                    "name": self.model_name,
                    "results": [
                        {
                            "task": {"type": self.task, "name": self.task.replace("-", " ").title()},
                            **(
                                {"dataset": {"type": self.datasets[0], "name": self.datasets[0]}}
                                if self.datasets
                                else {}
                            ),
                            "metrics": [
                                {
                                    "type": _metric_type(name),
                                    "name": _KNOWN_METRIC_TYPES.get(
                                        _metric_type(name), _metric_type(name)
                                    ),
                                    "value": float(value),
                                }
                                for name, value in sorted(self.metrics.items())
                            ],
                        }
                    ],
                }
            ]
        return block

    def provenance(self) -> dict[str, str]:
        """Revisions and fingerprints, as a flat mapping for rendering."""
        entries = {
            "Base model": self.base_model,
            "Base model revision": self.base_model_revision,
            "Dataset": self.datasets[0] if self.datasets else None,
            "Dataset revision": self.dataset_revision,
            "Dataset fingerprint": self.dataset_fingerprint,
            "Source commit": self.source_commit,
        }
        return {key: value for key, value in entries.items() if value}


def render_hub_card(
    metadata: HubCardMetadata,
    *,
    summary: str | None = None,
    intended_use: Sequence[str] = (),
    limitations: Sequence[str] = (),
    training_config: Mapping[str, Any] | None = None,
) -> str:
    """Render the repository README, YAML block first."""
    frontmatter = yaml.safe_dump(
        metadata.to_frontmatter(), sort_keys=False, default_flow_style=False
    ).strip()

    lines = [
        "---",
        frontmatter,
        "---",
        "",
        f"# {metadata.model_name}",
        "",
        summary
        or f"A {metadata.task.replace('-', ' ')} model fine-tuned with the "
        "[Hugging Face Fine-Tuning Lab]"
        "(https://github.com/DiogoRibeiro7/huggingface-finetuning-lab).",
        "",
    ]

    if metadata.metrics:
        lines += ["## Evaluation", "", "| Metric | Value |", "| --- | --- |"]
        lines += [
            f"| `{_metric_type(name)}` | {float(value):.4f} |"
            for name, value in sorted(metadata.metrics.items())
        ]
        lines += [
            "",
            "Measured on the held-out test split, which was not used to select "
            "any threshold, calibration parameter or policy.",
            "",
        ]

    provenance = metadata.provenance()
    if provenance:
        lines += ["## Provenance", "", "| Field | Value |", "| --- | --- |"]
        lines += [f"| {key} | `{value}` |" for key, value in provenance.items()]
        lines += [
            "",
            "Revisions are recorded because a repository id is mutable: the same "
            "name can resolve to different weights or rows later.",
            "",
        ]

    if intended_use:
        lines += ["## Intended use", ""]
        lines += [f"- {item}" for item in intended_use]
        lines.append("")

    if limitations:
        lines += ["## Limitations", ""]
        lines += [f"- {item}" for item in limitations]
        lines.append("")

    if training_config:
        lines += ["## Training configuration", "", "```yaml"]
        lines.append(yaml.safe_dump(dict(training_config), sort_keys=True).strip())
        lines += ["```", ""]

    return "\n".join(lines).rstrip() + "\n"


def write_hub_card(
    model_dir: str | Path,
    metadata: HubCardMetadata,
    **kwargs: Any,
) -> Path:
    """Write the repository README into a model directory."""
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / HUB_CARD_FILENAME
    destination.write_text(render_hub_card(metadata, **kwargs), encoding="utf-8")
    return destination
