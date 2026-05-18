"""Retrieval utilities: an embedding index, cosine search, and retrieval metrics."""

from hf_finetuning_lab.retrieval.index import EmbeddingIndex, IndexEntry, l2_normalize
from hf_finetuning_lab.retrieval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    retrieval_report,
)

__all__ = [
    "EmbeddingIndex",
    "IndexEntry",
    "l2_normalize",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "retrieval_report",
]
