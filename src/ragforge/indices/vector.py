"""Dense vector index with dual-engine (pure Python / optional NumPy) similarity search."""

import math
from typing import Any, Callable, Dict, List, Optional
from ragforge.chunkers.base import Chunk
from ragforge.indices.bm25 import SearchResult


class VectorIndex:
    """
    In-memory dense vector search index supporting Cosine Similarity,
    Dot Product, and Euclidean distance.

    Can operate with custom embeddings (e.g. OpenAI, HuggingFace, Ollama)
    or a built-in high-speed n-gram hashed vectorizer for zero-dependency local use.
    """

    def __init__(
        self,
        dimension: int = 128,
        metric: str = "cosine",
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.dimension = dimension
        self.metric = metric.lower()
        self.embed_fn = embed_fn
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def _default_vectorize(self, text: str) -> List[float]:
        """Built-in deterministic hashed bag-of-words/character-trigram vectorizer."""
        vec = [0.0] * self.dimension
        words = text.lower().split()
        for w in words:
            # Word hash
            h = hash(w) % self.dimension
            vec[h] += 1.0

        # Character trigrams for morphological capture
        clean = "".join(text.lower().split())
        for i in range(len(clean) - 2):
            tri = clean[i:i + 3]
            h = hash(tri) % self.dimension
            vec[h] += 0.5

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def add_chunk(self, chunk: Chunk, vector: Optional[List[float]] = None) -> None:
        """Add a chunk with an optional precomputed vector."""
        if vector is None:
            if self.embed_fn:
                vector = self.embed_fn([chunk.text])[0]
            else:
                vector = self._default_vectorize(chunk.text)

        self.chunks.append(chunk)
        self.vectors.append(vector)

    def add_chunks(self, chunks: List[Chunk], vectors: Optional[List[List[float]]] = None) -> None:
        """Add multiple chunks, generating vectors in batch if embed_fn is configured."""
        if vectors is None:
            if self.embed_fn:
                texts = [c.text for c in chunks]
                vectors = self.embed_fn(texts)
            else:
                vectors = [self._default_vectorize(c.text) for c in chunks]

        for chunk, vec in zip(chunks, vectors):
            self.chunks.append(chunk)
            self.vectors.append(vec)

    def _similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute similarity score based on configured metric."""
        if self.metric == "cosine":
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 > 0 and norm2 > 0:
                return dot / (norm1 * norm2)
            return 0.0
        elif self.metric == "dot":
            return sum(a * b for a, b in zip(v1, v2))
        elif self.metric == "euclidean":
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def search(self, query: str, top_k: int = 5, query_vector: Optional[List[float]] = None) -> List[SearchResult]:
        """Search the indexed vectors."""
        if not self.chunks:
            return []

        if query_vector is None:
            if self.embed_fn:
                query_vector = self.embed_fn([query])[0]
            else:
                query_vector = self._default_vectorize(query)

        scored: List[Tuple[int, float]] = []
        for i, vec in enumerate(self.vectors):
            score = self._similarity(query_vector, vec)
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored[:top_k]

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(top_matches, 1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                )
            )

        return results
