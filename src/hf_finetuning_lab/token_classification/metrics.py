"""Entity-level (BIO span) precision/recall/F1 for token classification."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def extract_entities(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """Walk a BIO label sequence and return ``(entity_type, start, end_exclusive)`` spans.

    Malformed sequences (``I-<TYPE>`` without a preceding ``B-<TYPE>`` or
    ``I-<TYPE>`` of the same type) start a new entity, matching the common
    relaxed/strict-mode convention used by seqeval.
    """
    spans: list[tuple[str, int, int]] = []
    current_type: str | None = None
    current_start: int | None = None

    def _flush(end: int) -> None:
        nonlocal current_type, current_start
        if current_type is not None and current_start is not None:
            spans.append((current_type, current_start, end))
        current_type = None
        current_start = None

    for idx, label in enumerate(labels):
        if label == "O":
            _flush(idx)
            continue
        if label.startswith("B-"):
            _flush(idx)
            current_type = label[2:]
            current_start = idx
        elif label.startswith("I-"):
            tag_type = label[2:]
            if current_type == tag_type:
                continue
            # Relaxed: a stray I- starts a new entity.
            _flush(idx)
            current_type = tag_type
            current_start = idx
        else:
            # Unknown prefix; treat as O.
            _flush(idx)
    _flush(len(labels))
    return spans


def sequence_tagging_report(
    true_sequences: Sequence[Sequence[str]],
    pred_sequences: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Return per-entity-type and micro/macro precision/recall/F1.

    The report counts entity spans, not individual tokens, so a multi-token
    entity is one hit regardless of length.
    """
    if len(true_sequences) != len(pred_sequences):
        raise ValueError(
            f"true_sequences and pred_sequences must have matching lengths "
            f"(got {len(true_sequences)} and {len(pred_sequences)})."
        )

    type_counts: dict[str, dict[str, int]] = {}

    def _bucket(entity_type: str) -> dict[str, int]:
        return type_counts.setdefault(entity_type, {"tp": 0, "fp": 0, "fn": 0})

    for true_seq, pred_seq in zip(true_sequences, pred_sequences, strict=True):
        if len(true_seq) != len(pred_seq):
            raise ValueError(
                f"Sequence length mismatch in one example "
                f"({len(true_seq)} vs {len(pred_seq)})."
            )
        true_spans = set(extract_entities(true_seq))
        pred_spans = set(extract_entities(pred_seq))
        for entity_type, _, _ in true_spans & pred_spans:
            _bucket(entity_type)["tp"] += 1
        for entity_type, _, _ in pred_spans - true_spans:
            _bucket(entity_type)["fp"] += 1
        for entity_type, _, _ in true_spans - pred_spans:
            _bucket(entity_type)["fn"] += 1

    per_type: list[tuple[str, float, float, float, int]] = []
    for entity_type in sorted(type_counts):
        counts = type_counts[entity_type]
        per_type.append((entity_type, *_prf_support(counts)))

    micro_counts = {
        "tp": sum(c["tp"] for c in type_counts.values()),
        "fp": sum(c["fp"] for c in type_counts.values()),
        "fn": sum(c["fn"] for c in type_counts.values()),
    }
    micro_row = ("micro avg", *_prf_support(micro_counts))

    rows = [
        {
            "entity_type": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        for name, precision, recall, f1, support in per_type
    ]
    rows.append(
        {
            "entity_type": micro_row[0],
            "precision": micro_row[1],
            "recall": micro_row[2],
            "f1": micro_row[3],
            "support": micro_row[4],
        }
    )
    if per_type:
        n = len(per_type)
        macro_p = sum(row[1] for row in per_type) / n
        macro_r = sum(row[2] for row in per_type) / n
        macro_f = sum(row[3] for row in per_type) / n
        macro_support = sum(row[4] for row in per_type)
        rows.append(
            {
                "entity_type": "macro avg",
                "precision": macro_p,
                "recall": macro_r,
                "f1": macro_f,
                "support": macro_support,
            }
        )

    frame = pd.DataFrame(rows).set_index("entity_type")
    if "support" in frame.columns:
        frame["support"] = frame["support"].astype(int)
    return frame


def _prf_support(counts: dict[str, int]) -> tuple[float, float, float, int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1), int(tp + fn)
