"""Retrieval metrics: Recall@k, MRR, and nDCG@k over ranked doc-id lists."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import pandas as pd


def _validate_pair(
    rankings: Sequence[Sequence[str]],
    relevance: Sequence[Iterable[str]],
) -> None:
    if len(rankings) != len(relevance):
        raise ValueError(
            f"rankings and relevance must have the same length "
            f"(got {len(rankings)} and {len(relevance)})."
        )


def recall_at_k(
    rankings: Sequence[Sequence[str]],
    relevance: Sequence[Iterable[str]],
    k: int,
) -> float:
    """Mean fraction of relevant docs found in the first ``k`` ranked positions.

    Queries with an empty relevance set are skipped (they contribute neither
    to the numerator nor the denominator).
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    _validate_pair(rankings, relevance)
    scored: list[float] = []
    for ranking, gold in zip(rankings, relevance, strict=True):
        gold_set = set(gold)
        if not gold_set:
            continue
        top_k = set(ranking[:k])
        scored.append(len(top_k & gold_set) / len(gold_set))
    return float(sum(scored) / len(scored)) if scored else 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevance: Sequence[Iterable[str]],
) -> float:
    """Mean of ``1 / rank`` of the first relevant document per query (or 0 if none).

    Queries with an empty relevance set are skipped.
    """
    _validate_pair(rankings, relevance)
    scored: list[float] = []
    for ranking, gold in zip(rankings, relevance, strict=True):
        gold_set = set(gold)
        if not gold_set:
            continue
        rr = 0.0
        for position, doc_id in enumerate(ranking, start=1):
            if doc_id in gold_set:
                rr = 1.0 / position
                break
        scored.append(rr)
    return float(sum(scored) / len(scored)) if scored else 0.0


def ndcg_at_k(
    rankings: Sequence[Sequence[str]],
    relevance: Sequence[Iterable[str]],
    k: int,
) -> float:
    """Mean Normalised Discounted Cumulative Gain at depth ``k`` (binary relevance)."""
    if k <= 0:
        raise ValueError("k must be positive.")
    _validate_pair(rankings, relevance)
    scored: list[float] = []
    for ranking, gold in zip(rankings, relevance, strict=True):
        gold_set = set(gold)
        if not gold_set:
            continue
        dcg = 0.0
        for position, doc_id in enumerate(ranking[:k], start=1):
            if doc_id in gold_set:
                dcg += 1.0 / math.log2(position + 1)
        ideal_hits = min(len(gold_set), k)
        idcg = sum(1.0 / math.log2(p + 1) for p in range(1, ideal_hits + 1))
        scored.append(dcg / idcg if idcg > 0 else 0.0)
    return float(sum(scored) / len(scored)) if scored else 0.0


def retrieval_report(
    rankings: Sequence[Sequence[str]],
    relevance: Sequence[Iterable[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Return a one-row-per-k frame with Recall@k and nDCG@k, plus MRR (constant)."""
    if not ks:
        raise ValueError("ks must not be empty.")
    mrr = mean_reciprocal_rank(rankings, relevance)
    rows = [
        {
            "k": int(k),
            "recall_at_k": recall_at_k(rankings, relevance, k),
            "ndcg_at_k": ndcg_at_k(rankings, relevance, k),
            "mrr": mrr,
        }
        for k in ks
    ]
    return pd.DataFrame(rows).set_index("k")
