"""Synthetic NER dataset generation and validation.

The schema mirrors CoNLL-style token classification: each example has a list
of word-level tokens and a parallel list of BIO labels (``B-<TYPE>``,
``I-<TYPE>``, or ``O``). Persistence uses JSONL with one example per line so
the data stays human-inspectable and easy to diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class NERExample:
    """One NER example: parallel token + label lists."""

    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.labels):
            raise ValueError(
                f"tokens and labels must have the same length "
                f"(got {len(self.tokens)} and {len(self.labels)})."
            )


_PERSON_FIRST = ["Maria", "John", "Aisha", "Liu", "Carlos", "Priya", "Olu", "Hana"]
_PERSON_LAST = ["Silva", "Smith", "Khan", "Wei", "Garcia", "Patel", "Adeyemi", "Park"]
_ORGS = [
    "Globex Corp",
    "Initech",
    "Hooli",
    "Soylent Industries",
    "Pied Piper",
    "Stark Solutions",
]
_LOCS = ["Lisbon", "Berlin", "Tokyo", "Sao Paulo", "Nairobi", "Toronto", "Mumbai", "Madrid"]

_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("PERSON", "joined", "ORG", "in", "LOC", "last", "week", "."),
    ("ORG", "opened", "a", "new", "office", "in", "LOC", "."),
    ("PERSON", "visited", "LOC", "to", "meet", "the", "ORG", "team", "."),
    ("Yesterday", "ORG", "announced", "a", "partnership", "with", "ORG2", "."),
    ("PERSON", "and", "PERSON2", "co-founded", "ORG", "in", "LOC", "."),
    ("The", "ORG", "headquarters", "moved", "to", "LOC", "last", "year", "."),
)


def _tokenize_phrase(phrase: str) -> list[str]:
    """Split a multi-word entity into space-separated tokens."""
    return phrase.split()


def _emit_entity(tokens_out: list[str], labels_out: list[str], phrase: str, entity_type: str) -> None:
    parts = _tokenize_phrase(phrase)
    for idx, part in enumerate(parts):
        tokens_out.append(part)
        labels_out.append(f"B-{entity_type}" if idx == 0 else f"I-{entity_type}")


def generate_synthetic_ner_data(n_examples: int = 300, seed: int = 42) -> list[NERExample]:
    """Generate a small synthetic CoNLL-style NER dataset.

    Entity types are ``PER`` (people), ``ORG`` (organisations), and ``LOC``
    (locations). The data is intentionally short and noisy so the baseline
    workflow exercises every code path quickly.
    """
    if n_examples <= 0:
        raise ValueError("n_examples must be positive.")

    rng = np.random.default_rng(seed)
    examples: list[NERExample] = []

    for _ in range(n_examples):
        template = _TEMPLATES[int(rng.integers(0, len(_TEMPLATES)))]
        tokens: list[str] = []
        labels: list[str] = []
        person_pool = rng.permutation(len(_PERSON_FIRST))
        person_idx = 0
        org_pool = rng.permutation(len(_ORGS))
        org_idx = 0
        for slot in template:
            if slot == "PERSON":
                first = _PERSON_FIRST[int(person_pool[person_idx % len(person_pool)])]
                last = _PERSON_LAST[int(person_pool[person_idx % len(_PERSON_LAST)])]
                _emit_entity(tokens, labels, f"{first} {last}", "PER")
                person_idx += 1
            elif slot == "PERSON2":
                first = _PERSON_FIRST[int(person_pool[(person_idx + 3) % len(person_pool)])]
                last = _PERSON_LAST[int(person_pool[(person_idx + 3) % len(_PERSON_LAST)])]
                _emit_entity(tokens, labels, f"{first} {last}", "PER")
                person_idx += 1
            elif slot == "ORG":
                _emit_entity(tokens, labels, _ORGS[int(org_pool[org_idx % len(org_pool)])], "ORG")
                org_idx += 1
            elif slot == "ORG2":
                _emit_entity(
                    tokens,
                    labels,
                    _ORGS[int(org_pool[(org_idx + 2) % len(org_pool)])],
                    "ORG",
                )
                org_idx += 1
            elif slot == "LOC":
                _emit_entity(tokens, labels, _LOCS[int(rng.integers(0, len(_LOCS)))], "LOC")
            else:
                tokens.append(slot)
                labels.append("O")
        examples.append(NERExample(tokens=tokens, labels=labels))
    return examples


def validate_ner_dataset(examples: list[NERExample]) -> None:
    """Check that every example has aligned token/label lists and at least one token."""
    if not examples:
        raise ValueError("Empty NER dataset.")
    for i, example in enumerate(examples):
        if len(example.tokens) == 0:
            raise ValueError(f"Example {i}: empty tokens list.")
        if len(example.tokens) != len(example.labels):
            raise ValueError(
                f"Example {i}: tokens/labels length mismatch "
                f"({len(example.tokens)} vs {len(example.labels)})."
            )
        for label in example.labels:
            # Valid labels are "O" or "B-/I-" with a non-empty entity type;
            # a bare "B-"/"I-" (empty type) must not pass validation.
            if label == "O":
                continue
            if label[:2] in ("B-", "I-") and len(label) > 2:
                continue
            raise ValueError(
                f"Example {i}: invalid BIO label '{label}'. Expected 'O', 'B-<TYPE>', or 'I-<TYPE>'."
            )


def write_synthetic_ner_jsonl(
    output: str | Path,
    n_examples: int = 300,
    seed: int = 42,
) -> Path:
    """Generate synthetic NER data and write it to a JSONL file."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = generate_synthetic_ner_data(n_examples=n_examples, seed=seed)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"tokens": example.tokens, "labels": example.labels}) + "\n")
    return path
