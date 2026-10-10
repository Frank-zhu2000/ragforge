"""Unit tests for indexing and search retrieval algorithms."""

import unittest
from ragforge.chunkers.base import Chunk
from ragforge.indices.bm25 import BM25Index
from ragforge.indices.vector import VectorIndex
from ragforge.indices.hybrid import HybridIndex


class TestIndices(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            Chunk(chunk_id="c1", text="Python is a dynamic programming language widely used in machine learning."),
            Chunk(chunk_id="c2", text="PostgreSQL and SQLite are powerful relational database management systems."),
            Chunk(chunk_id="c3", text="Docker and Kubernetes simplify container deployment and orchestration."),
            Chunk(chunk_id="c4", text="BM25 and vector embeddings form the foundation of hybrid search in RAG."),
        ]

    def test_bm25_index(self):
        index = BM25Index()
        index.add_chunks(self.chunks)

        results = index.search("machine learning Python", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertTrue(results[0].score > 0)

        # Non-matching query
        no_res = index.search("quantum teleportation astrophysics", top_k=2)
        self.assertEqual(no_res, [])

    def test_vector_index(self):
        index = VectorIndex(dimension=64, metric="cosine")
        index.add_chunks(self.chunks)

        results = index.search("container Docker deployment", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].chunk.chunk_id, "c3")
        self.assertTrue(results[0].score > 0.5)

    def test_vector_metrics(self):
        # Test Euclidean and Dot product metrics
        idx_euc = VectorIndex(dimension=32, metric="euclidean")
        idx_euc.add_chunks(self.chunks)
        res_euc = idx_euc.search("relational database", top_k=1)
        self.assertEqual(res_euc[0].chunk.chunk_id, "c2")

        idx_dot = VectorIndex(dimension=32, metric="dot")
        idx_dot.add_chunks(self.chunks)
        res_dot = idx_dot.search("relational database", top_k=1)
        self.assertEqual(res_dot[0].chunk.chunk_id, "c2")

    def test_hybrid_index_rrf(self):
        hybrid = HybridIndex(vector_dim=64, rrf_k=60)
        hybrid.add_chunks(self.chunks)

        # Hybrid search query
        results = hybrid.search("hybrid search RAG embeddings", top_k=2, method="rrf")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].chunk.chunk_id, "c4")
        self.assertEqual(results[0].rank, 1)

    def test_hybrid_index_alpha(self):
        hybrid = HybridIndex(vector_dim=64)
        hybrid.add_chunks(self.chunks)

        results = hybrid.search("PostgreSQL SQLite database", top_k=2, method="alpha", alpha=0.6)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].chunk.chunk_id, "c2")


if __name__ == "__main__":
    unittest.main()
