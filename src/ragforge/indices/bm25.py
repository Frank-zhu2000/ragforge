"""Fast Okapi BM25 sparse keyword search index with multilingual tokenization."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from ragforge.chunkers.base import Chunk


@dataclass
class SearchResult:
    """Standardized search match result."""
    chunk: Chunk
    score: float
    rank: int = 1


class BM25Index:
    """
    Okapi BM25 index supporting pure-Python ranking, incremental addition,
    and automatic multilingual tokenization (English words + CJK unigrams/bigrams).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # doc_idx -> Counter(term)
        self.idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """Multilingual tokenization supporting English, CJK, and code symbols."""
        if not text:
            return []

        text = text.lower()
        # Extract English words and digits
        words = re.findall(r"[a-z0-9_\-]+", text)

        # Extract Chinese / Japanese / Korean single characters and bigrams
        cjk_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
        cjk_bigrams = [
            cjk_chars[i] + cjk_chars[i + 1]
            for i in range(len(cjk_chars) - 1)
        ]

        return words + cjk_chars + cjk_bigrams

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Index a batch of Chunk objects."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def add_chunk(self, chunk: Chunk) -> None:
        """Add and index a single Chunk object."""
        tokens = self.tokenize(chunk.text)
        doc_len = len(tokens)

        counts = Counter(tokens)
        self.chunks.append(chunk)
        self.term_freqs.append(dict(counts))
        self.doc_lengths.append(doc_len)

        for term in counts.keys():
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.corpus_size = len(self.chunks)
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.corpus_size)
        self.idf_cache.clear()  # Invalidate cached IDFs

    def _calc_idf(self, term: str) -> float:
        """Calculate smoothed IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        df = self.doc_freqs.get(term, 0)
        # Smoothed BM25 IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)
        self.idf_cache[term] = max(0.0, idf)
        return self.idf_cache[term]

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the indexed chunks using BM25 scoring."""
        if not self.chunks or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            idf = self._calc_idf(term)
            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    term_score = idf * ((tf * (self.k1 + 1.0)) / denom)
                    scores[idx] += term_score

        # Rank documents by score
        ranked_indices = sorted(
            [i for i in range(self.corpus_size) if scores[i] > 0.0],
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices, 1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=scores[idx],
                    rank=rank,
                )
            )

        return results
