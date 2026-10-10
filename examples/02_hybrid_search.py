"""02_hybrid_search.py: Reciprocal Rank Fusion (RRF) and hybrid search example."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragforge import Chunk, HybridIndex


def main():
    # Sample corpus of documents
    knowledge_base = [
        "PostgreSQL supports ACID transactions, foreign keys, and JSONB document columns.",
        "Redis is an in-memory key-value data store used primarily for high-speed caching and pub/sub queues.",
        "Vector databases like Pinecone, Milvus, and Qdrant store high-dimensional embeddings for semantic search.",
        "BM25 is a sparse lexical probabilistic ranking algorithm that calculates term frequency and inverse document frequency.",
        "Hybrid search combines dense vector retrieval and BM25 sparse keyword search via Reciprocal Rank Fusion (RRF).",
        "Docker packages applications and dependencies into standardized containers for reproducible deployments.",
    ]

    chunks = [Chunk(chunk_id=f"doc_{i}", text=text) for i, text in enumerate(knowledge_base, 1)]

    print("=== Step 1: Initialize and Populate Hybrid Index ===")
    index = HybridIndex(vector_dim=64, rrf_k=60)
    index.add_chunks(chunks)
    print(f"Indexed {len(chunks)} documents into BM25 and Vector indices.")

    print("\n=== Step 2: Query using Reciprocal Rank Fusion (RRF) ===")
    query = "How does hybrid search combine BM25 and vector embeddings?"
    print(f"Query: \"{query}\"\n")

    results = index.search(query, top_k=3, method="rrf")

    for r in results:
        print(f"Rank #{r.rank} | Combined RRF Score: {r.score:.6f} | Chunk ID: {r.chunk.chunk_id}")
        print(f"Text: {r.chunk.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
