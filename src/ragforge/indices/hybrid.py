"""Hybrid search index combining BM25 keyword and dense vector retrieval."""

from typing import Any, Callable, Dict, List, Optional
from ragforge.chunkers.base import Chunk
from ragforge.indices.bm25 import BM25Index, SearchResult
from ragforge.indices.vector import VectorIndex


class HybridIndex:
    """
    Hybrid search combining BM25 sparse keyword retrieval and dense vector retrieval.
    Merges candidate lists using Reciprocal Rank Fusion (RRF) or Alpha Score Fusion.
    """

    def __init__(
        self,
        vector_dim: int = 128,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        rrf_k: int = 60,
    ):
        self.bm25 = BM25Index()
        self.vector = VectorIndex(dimension=vector_dim, embed_fn=embed_fn)
        self.rrf_k = rrf_k
        self.chunks_map: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to both BM25 and vector indices."""
        for c in chunks:
            self.chunks_map[c.chunk_id] = c
        self.bm25.add_chunks(chunks)
        self.vector.add_chunks(chunks)

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a single chunk to both indices."""
        self.chunks_map[chunk.chunk_id] = chunk
        self.bm25.add_chunk(chunk)
        self.vector.add_chunk(chunk)

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "rrf",
        alpha: float = 0.5,
        candidate_multiplier: int = 3,
    ) -> List[SearchResult]:
        """
        Execute hybrid search.

        :param query: Query string.
        :param top_k: Final number of ranked results.
        :param method: 'rrf' (Reciprocal Rank Fusion) or 'alpha' (Weighted score blend).
        :param alpha: Weight for vector score when method='alpha' (1.0 = pure vector, 0.0 = pure BM25).
        :param candidate_multiplier: Top-N candidates to fetch from each index before fusion.
        """
        fetch_k = top_k * candidate_multiplier
        bm25_res = self.bm25.search(query, top_k=fetch_k)
        vector_res = self.vector.search(query, top_k=fetch_k)

        if method == "rrf":
            return self._fuse_rrf(bm25_res, vector_res, top_k)
        elif method == "alpha":
            return self._fuse_alpha(bm25_res, vector_res, top_k, alpha)
        else:
            raise ValueError(f"Unknown fusion method '{method}'. Choose 'rrf' or 'alpha'.")

    def _fuse_rrf(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion: RRF_score = sum(1 / (k + rank))."""
        rrf_scores: Dict[str, float] = {}

        for rank, res in enumerate(bm25_results, 1):
            cid = res.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, res in enumerate(vector_results, 1):
            cid = res.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        final_results: List[SearchResult] = []
        for rank, cid in enumerate(sorted_cids, 1):
            final_results.append(
                SearchResult(
                    chunk=self.chunks_map[cid],
                    score=rrf_scores[cid],
                    rank=rank,
                )
            )

        return final_results

    def _fuse_alpha(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        top_k: int,
        alpha: float,
    ) -> List[SearchResult]:
        """Min-Max normalized score fusion: alpha * vector + (1 - alpha) * bm25."""
        def normalize(res_list: List[SearchResult]) -> Dict[str, float]:
            if not res_list:
                return {}
            scores = [r.score for r in res_list]
            min_s, max_s = min(scores), max(scores)
            norm_map = {}
            for r in res_list:
                if max_s > min_s:
                    norm_map[r.chunk.chunk_id] = (r.score - min_s) / (max_s - min_s)
                else:
                    norm_map[r.chunk.chunk_id] = 1.0
            return norm_map

        bm25_norm = normalize(bm25_results)
        vec_norm = normalize(vector_results)

        all_cids = set(bm25_norm.keys()) | set(vec_norm.keys())
        blended: Dict[str, float] = {}

        for cid in all_cids:
            b_s = bm25_norm.get(cid, 0.0)
            v_s = vec_norm.get(cid, 0.0)
            blended[cid] = alpha * v_s + (1.0 - alpha) * b_s

        sorted_cids = sorted(blended.keys(), key=lambda x: blended[x], reverse=True)[:top_k]

        final_results: List[SearchResult] = []
        for rank, cid in enumerate(sorted_cids, 1):
            final_results.append(
                SearchResult(
                    chunk=self.chunks_map[cid],
                    score=blended[cid],
                    rank=rank,
                )
            )

        return final_results
