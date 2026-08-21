"""Model cards report quality, and document ids identify one document."""

from __future__ import annotations

import numpy as np
import pytest

from hf_finetuning_lab.model_cards.model_card import is_quality_metric, quality_metrics
from hf_finetuning_lab.retrieval.index import EmbeddingIndex, IndexEntry

# A representative Trainer.evaluate() payload.
TRAINER_OUTPUT = {
    "eval_loss": 0.42,
    "eval_accuracy": 0.91,
    "eval_f1": 0.89,
    "eval_runtime": 1.2345,
    "eval_samples_per_second": 435.6,
    "eval_steps_per_second": 72.6,
    "epoch": 2.0,
    "total_flos": 1.2e15,
}


def test_runtime_entries_are_not_presented_as_quality_metrics() -> None:
    selected = quality_metrics(TRAINER_OUTPUT)

    assert set(selected) == {"eval_loss", "eval_accuracy", "eval_f1"}


def test_throughput_keys_are_recognised() -> None:
    assert is_quality_metric("eval_f1")
    assert not is_quality_metric("eval_samples_per_second")
    assert not is_quality_metric("eval_runtime")
    assert not is_quality_metric("epoch")


def test_non_numeric_and_boolean_entries_are_dropped() -> None:
    selected = quality_metrics({"eval_f1": 0.5, "label": "billing", "converged": True})

    assert selected == {"eval_f1": 0.5}


def test_index_rejects_duplicate_document_ids() -> None:
    """nDCG@k would otherwise count one relevant document twice."""
    embeddings = np.eye(3, dtype=np.float32)
    entries = [IndexEntry("a", "x"), IndexEntry("b", "y"), IndexEntry("a", "z")]

    with pytest.raises(ValueError, match="doc_id values must be unique"):
        EmbeddingIndex(embeddings, entries)


def test_index_accepts_unique_document_ids() -> None:
    embeddings = np.eye(2, dtype=np.float32)

    index = EmbeddingIndex(embeddings, [IndexEntry("a", "x"), IndexEntry("b", "y")])

    assert index.size == 2
