from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "This artifact may be trained or evaluated on synthetic or simplified data.",
    "Performance on the target domain, subgroups, and edge cases must be validated separately.",
    "This model is not validated for clinical, legal, safety-critical, or other high-stakes decisions.",
)


def write_model_card(
    output_path: str | Path,
    model_name: str,
    task: str,
    label_names: list[str],
    metrics: dict[str, float],
    limitations: list[str] | None = None,
) -> Path:
    """Write a lightweight Hugging Face-style model card."""
    limitations = limitations or list(DEFAULT_LIMITATIONS)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    metric_lines = "\n".join(f"- **{name}**: {value:.4f}" for name, value in sorted(metrics.items()))
    label_lines = "\n".join(f"- {label}" for label in label_names)
    limitation_lines = "\n".join(f"- {item}" for item in limitations)
    created = datetime.now(UTC).strftime("%Y-%m-%d")

    intended_use = {
        "text-classification": "This model classifies text into one of the labels listed above.",
        "token-classification": (
            "This model assigns a label to each token (e.g. named-entity "
            "recognition) using the labels listed above."
        ),
        "retrieval": (
            "This model produces embeddings for semantic search and retrieval "
            "over the documented corpus."
        ),
    }.get(task, f"This model is intended for {task} experiments and workflow demonstrations.")

    content = f"""
# Model Card: {model_name}

## Overview

- **Base model:** `{model_name}`
- **Task:** {task}
- **Created:** {created}

## Labels

{label_lines}

## Evaluation Metrics

{metric_lines if metric_lines else "No metrics were provided."}

## Intended Use

{intended_use} Real deployment requires validation on the target domain.

## Limitations

{limitation_lines if limitation_lines else "No limitations were documented."}

## Ethical Considerations

Review privacy, label quality, subgroup performance, and operational failure modes before production use.
""".strip()
    destination.write_text(content + "\n", encoding="utf-8")
    return destination
