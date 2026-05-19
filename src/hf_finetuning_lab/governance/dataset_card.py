"""Structured dataset-card generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class DatasetColumn:
    """One column in the dataset schema."""

    name: str
    dtype: str
    role: str = "feature"
    description: str = ""


@dataclass(slots=True)
class DatasetSplit:
    """Row count and label distribution for one split."""

    name: str
    n_rows: int
    label_distribution: dict[str, int] | None = None


@dataclass(slots=True)
class DatasetCard:
    """All the metadata needed to render a dataset card."""

    name: str
    description: str
    task: str
    columns: list[DatasetColumn] = field(default_factory=list)
    splits: list[DatasetSplit] = field(default_factory=list)
    collection_method: str = "synthetic"
    license: str = "internal"
    privacy_notes: list[str] = field(default_factory=list)
    intended_use: list[str] = field(default_factory=list)
    not_intended_use: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _bullets(items: list[str], fallback: str) -> str:
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)


def _columns_table(columns: list[DatasetColumn]) -> str:
    if not columns:
        return "No columns documented."
    rows = ["| name | dtype | role | description |", "| --- | --- | --- | --- |"]
    for column in columns:
        rows.append(
            f"| `{column.name}` | `{column.dtype}` | {column.role} | {column.description or '—'} |"
        )
    return "\n".join(rows)


def _splits_section(splits: list[DatasetSplit]) -> str:
    if not splits:
        return "No splits documented."
    parts: list[str] = []
    for split in splits:
        header = f"### {split.name}\n\n- Rows: **{split.n_rows}**"
        if split.label_distribution:
            ordered = sorted(split.label_distribution.items())
            dist_lines = "\n".join(f"  - `{label}`: {count}" for label, count in ordered)
            header += f"\n- Label distribution:\n{dist_lines}"
        parts.append(header)
    return "\n\n".join(parts)


def write_dataset_card(card: DatasetCard, output_path: str | Path) -> Path:
    """Render ``card`` as a Markdown dataset card and write it to ``output_path``."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).strftime("%Y-%m-%d")

    body = f"""# Dataset Card: {card.name}

## Overview

- **Task:** {card.task}
- **Collection method:** {card.collection_method}
- **License:** {card.license}
- **Created:** {created}

## Description

{card.description}

## Schema

{_columns_table(card.columns)}

## Splits

{_splits_section(card.splits)}

## Intended Use

{_bullets(card.intended_use, "Intended use was not documented.")}

## Not Intended Use

{_bullets(card.not_intended_use, "No explicit out-of-scope uses were documented.")}

## Privacy Considerations

{_bullets(card.privacy_notes, "No privacy notes were documented.")}

## Limitations

{_bullets(card.limitations, "No limitations were documented.")}

## Sources

{_bullets(card.sources, "No external sources were documented.")}
"""
    destination.write_text(body.strip() + "\n", encoding="utf-8")
    return destination
