"""03_pipeline_eval.py: RAG evaluation metrics and report generation example."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragforge.eval import EvaluationSample, RAGEvaluator


def main():
    print("=== Step 1: Create Evaluation Test Cases ===")
    samples = [
        EvaluationSample(
            query="What is the time complexity of binary search?",
            retrieved_chunk_ids=["chunk_bs_1", "chunk_sort_2", "chunk_tree_1"],
            ground_truth_chunk_ids={"chunk_bs_1"},
            context_text="Binary search runs in O(log n) time by halving search intervals.",
            generated_answer="Binary search has a logarithmic time complexity of O(log n).",
        ),
        EvaluationSample(
            query="Explain CAP theorem tradeoffs in distributed systems.",
            retrieved_chunk_ids=["chunk_raft_1", "chunk_cap_1", "chunk_paxos_2"],
            ground_truth_chunk_ids={"chunk_cap_1"},
            context_text="CAP theorem states a distributed store can only guarantee 2 of Consistency, Availability, Partition tolerance.",
            generated_answer="Under network partitions, a system must choose between consistency and availability according to CAP.",
        ),
        EvaluationSample(
            query="How does Redis persist data to disk?",
            retrieved_chunk_ids=["chunk_redis_rdb", "chunk_redis_aof", "chunk_memcached"],
            ground_truth_chunk_ids={"chunk_redis_rdb", "chunk_redis_aof"},
            context_text="Redis offers point-in-time snapshotting (RDB) and append-only logging (AOF).",
            generated_answer="Redis uses RDB snapshots and AOF logs for persistence.",
        ),
    ]

    print(f"Loaded {len(samples)} benchmark evaluation cases.")

    print("\n=== Step 2: Run RAGEvaluator (@k=3) ===")
    evaluator = RAGEvaluator(k=3)
    report = evaluator.evaluate(samples)

    print("\nBenchmark Results:")
    print(report.to_markdown_table())


if __name__ == "__main__":
    main()
