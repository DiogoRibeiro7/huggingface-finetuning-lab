"""A small, dependency-free cosine-similarity index for retrieval demos.

The index stores L2-normalized row vectors so a single matrix multiplication
between a query vector and the index yields cosine similarity scores. It is
deliberately small in scope: build the embedding outside the index (e.g. with
``TfidfVectorizer`` or ``sentence-transformers``) and hand the resulting
matrix in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class IndexEntry:
    """One document in the index."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def l2_normalize(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm.

    Zero-norm rows are left at zero rather than producing NaNs.
    """
    if matrix.ndim != 2:
        raise ValueError(f"l2_normalize expects a 2D matrix; got shape {matrix.shape}.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms < epsilon, 1.0, norms)
    return np.asarray(matrix / safe)


class EmbeddingIndex:
    """Cosine-similarity index over a fixed corpus of L2-normalized embeddings."""

    def __init__(self, embeddings: np.ndarray, entries: Sequence[IndexEntry]) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be a 2D matrix; got shape {embeddings.shape}."
            )
        if len(entries) != embeddings.shape[0]:
            raise ValueError(
                f"entries length ({len(entries)}) does not match embeddings rows "
                f"({embeddings.shape[0]})."
            )
        self._embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self._entries: list[IndexEntry] = list(entries)

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._entries)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self._embeddings.shape[1])

    @property
    def entries(self) -> list[IndexEntry]:
        """A shallow copy of the index entries in insertion order."""
        return list(self._entries)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[IndexEntry, float]]:
        """Return the top-``k`` entries ranked by cosine similarity, highest first."""
        if k <= 0:
            raise ValueError("k must be positive.")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                f"query dim {query.shape[1]} does not match index dim "
                f"{self._embeddings.shape[1]}."
            )
        query = l2_normalize(query)
        scores = (query @ self._embeddings.T).ravel()
        top = min(k, len(scores))
        if top == 0:
            return []
        # argpartition then sort the partitioned slice for stable descending order.
        partition = np.argpartition(-scores, top - 1)[:top]
        ordered = partition[np.argsort(-scores[partition])]
        return [(self._entries[idx], float(scores[idx])) for idx in ordered]

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> list[list[tuple[IndexEntry, float]]]:
        """Search multiple queries at once. Each row of ``query_embeddings`` is one query."""
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 2D; got shape {query_embeddings.shape}."
            )
        return [self.search(row, k=k) for row in query_embeddings]
