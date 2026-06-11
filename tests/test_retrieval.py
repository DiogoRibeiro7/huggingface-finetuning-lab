from __future__ import annotations

import numpy as np
import pytest

from hf_finetuning_lab.retrieval import (
    EmbeddingIndex,
    IndexEntry,
    l2_normalize,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    retrieval_report,
)


def _toy_entries() -> list[IndexEntry]:
    return [
        IndexEntry(doc_id="d1", text="alpha"),
        IndexEntry(doc_id="d2", text="bravo"),
        IndexEntry(doc_id="d3", text="charlie"),
    ]


def _toy_embeddings() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_l2_normalize_rows_have_unit_norm() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    normalized = l2_normalize(matrix)
    norms = np.linalg.norm(normalized, axis=1)
    assert pytest.approx(norms[0], rel=1e-6) == 1.0
    # Zero row stays zero (no NaN).
    assert np.allclose(normalized[1], 0.0)
    assert pytest.approx(norms[2], rel=1e-6) == 1.0


def test_l2_normalize_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        l2_normalize(np.array([1.0, 2.0]))


def test_embedding_index_search_returns_top_matches_in_order() -> None:
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    assert index.size == 3
    assert index.dim == 3
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    results = index.search(query, k=2)
    assert [entry.doc_id for entry, _ in results] == ["d1", "d2"]
    assert results[0][1] > results[1][1]


def test_embedding_index_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError):
        EmbeddingIndex(_toy_embeddings(), _toy_entries()[:2])


def test_embedding_index_rejects_dim_mismatch_on_query() -> None:
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    with pytest.raises(ValueError):
        index.search(np.array([1.0, 0.0], dtype=np.float32), k=1)


def test_embedding_index_rejects_non_positive_k() -> None:
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    with pytest.raises(ValueError):
        index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=0)


def test_embedding_index_search_rejects_multi_row_query() -> None:
    # A 2D query with more than one row would otherwise ravel into m*N scores
    # and silently return wrong documents; search() must reject it.
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    batched = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        index.search(batched, k=1)


def test_embedding_index_search_clamps_k_above_size() -> None:
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=10)
    assert len(results) == 3  # clamped to index size, no error


def test_embedding_index_search_tie_break_is_deterministic() -> None:
    # All documents are equidistant from this query, so every score ties.
    # Ordering must be stable (ascending index) and reproducible across calls.
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    query = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    first = [entry.doc_id for entry, _ in index.search(query, k=3)]
    second = [entry.doc_id for entry, _ in index.search(query, k=3)]
    assert first == second == ["d1", "d2", "d3"]


def test_embedding_index_search_batch_shapes() -> None:
    index = EmbeddingIndex(_toy_embeddings(), _toy_entries())
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    batched = index.search_batch(queries, k=1)
    assert len(batched) == 2
    assert batched[0][0][0].doc_id == "d1"
    assert batched[1][0][0].doc_id == "d3"


def test_recall_at_k_perfect_ranking_is_one() -> None:
    rankings = [["d1", "d2", "d3"]]
    relevance = [["d1"]]
    assert recall_at_k(rankings, relevance, k=1) == 1.0


def test_recall_at_k_missed_relevant_is_zero() -> None:
    rankings = [["d2", "d3", "d1"]]
    relevance = [["d1"]]
    assert recall_at_k(rankings, relevance, k=2) == 0.0
    assert recall_at_k(rankings, relevance, k=3) == 1.0


def test_recall_at_k_fractional_with_multiple_relevant() -> None:
    # Two relevant docs, only one inside the top-2 window -> recall 0.5.
    rankings = [["d1", "d4", "d2", "d3"]]
    relevance = [["d1", "d2"]]
    assert recall_at_k(rankings, relevance, k=2) == pytest.approx(0.5)
    assert recall_at_k(rankings, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_multiple_relevant_uses_ideal_dcg() -> None:
    # Both relevant docs ranked first is the ideal ordering -> nDCG == 1.0;
    # pushing one relevant doc later must lower it below 1.
    relevance = [["d1", "d2"]]
    ideal = ndcg_at_k([["d1", "d2", "d3"]], relevance, k=3)
    worse = ndcg_at_k([["d1", "d3", "d2"]], relevance, k=3)
    assert ideal == pytest.approx(1.0)
    assert worse < ideal


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k([["a"]], [["a"]], k=0)


def test_recall_at_k_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k([["a"]], [["a"], ["b"]], k=1)


def test_mean_reciprocal_rank_known_value() -> None:
    rankings = [["d2", "d1", "d3"], ["d3", "d2", "d1"]]
    relevance = [["d1"], ["d1"]]
    # First query: rank 2 -> 1/2. Second query: rank 3 -> 1/3. Mean = 5/12.
    assert mean_reciprocal_rank(rankings, relevance) == pytest.approx(5 / 12)


def test_mean_reciprocal_rank_zero_when_never_retrieved() -> None:
    assert mean_reciprocal_rank([["d2"]], [["d1"]]) == 0.0


def test_ndcg_at_k_perfect_ranking() -> None:
    rankings = [["d1", "d2", "d3"]]
    relevance = [["d1"]]
    assert ndcg_at_k(rankings, relevance, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_lower_for_later_match() -> None:
    perfect = ndcg_at_k([["d1", "d2"]], [["d1"]], k=2)
    later = ndcg_at_k([["d2", "d1"]], [["d1"]], k=2)
    assert later < perfect
    assert later > 0


def test_retrieval_report_columns_and_k_index() -> None:
    rankings = [["d1", "d2", "d3"], ["d2", "d3", "d1"]]
    relevance = [["d1"], ["d3"]]
    report = retrieval_report(rankings, relevance, ks=(1, 3))
    assert list(report.index) == [1, 3]
    assert {"recall_at_k", "ndcg_at_k", "mrr"}.issubset(report.columns)
    # MRR is constant across rows.
    assert report["mrr"].nunique() == 1


def test_retrieval_report_rejects_empty_ks() -> None:
    with pytest.raises(ValueError):
        retrieval_report([["a"]], [["a"]], ks=())
