"""Unit tests for RAG evaluation metrics and reporting."""

import unittest
from ragforge.eval.evaluator import EvaluationReport, EvaluationSample, RAGEvaluator
from ragforge.eval.metrics import (
    context_relevance,
    faithfulness,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestEvaluation(unittest.TestCase):

    def test_basic_metrics(self):
        retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
        gt = {"doc_B", "doc_F"}

        # hit_rate
        self.assertEqual(hit_rate(retrieved, gt, k=1), 0.0)
        self.assertEqual(hit_rate(retrieved, gt, k=3), 1.0)

        # precision@k
        self.assertEqual(precision_at_k(retrieved, gt, k=5), 1 / 5)

        # recall@k
        self.assertEqual(recall_at_k(retrieved, gt, k=5), 1 / 2)

        # MRR: doc_B is at index 1 -> rank 2 -> MRR = 0.5
        self.assertEqual(mrr(retrieved, gt), 0.5)

        # NDCG
        ndcg = ndcg_at_k(retrieved, gt, k=5)
        self.assertTrue(0.0 < ndcg <= 1.0)

    def test_context_relevance(self):
        query = "What is semantic chunking in RAG?"
        relevant_ctx = "Semantic chunking breaks text into meaningful units to preserve context for RAG."
        irrelevant_ctx = "Baking chocolate chip cookies requires flour, sugar, butter and eggs."

        score_rel = context_relevance(query, relevant_ctx)
        score_irrel = context_relevance(query, irrelevant_ctx)
        self.assertTrue(score_rel > score_irrel)

    def test_faithfulness(self):
        ctx = "RAGForge is an open-source Python library developed by Frank Zhu for optimizing RAG pipelines."
        faithful_ans = "RAGForge is a Python library by Frank Zhu."
        hallucinated_ans = "RAGForge is a proprietary enterprise software released in 1995 by Microsoft."

        score_f = faithfulness(faithful_ans, ctx)
        score_h = faithfulness(hallucinated_ans, ctx)
        self.assertTrue(score_f > score_h)

    def test_evaluator_runner(self):
        evaluator = RAGEvaluator(k=3)
        samples = [
            EvaluationSample(
                query="How does BM25 work?",
                retrieved_chunk_ids=["c1", "c2", "c3"],
                ground_truth_chunk_ids={"c1"},
                context_text="BM25 is a term matching ranking function.",
                generated_answer="BM25 matches terms.",
            ),
            EvaluationSample(
                query="What is vector search?",
                retrieved_chunk_ids=["c4", "c5", "c6"],
                ground_truth_chunk_ids={"c6"},
                context_text="Vector search uses embeddings and cosine similarity.",
                generated_answer="Vector search calculates cosine similarity.",
            ),
        ]

        report = evaluator.evaluate(samples)
        self.assertEqual(report.total_samples, 2)
        self.assertEqual(report.hit_rate_at_k, 1.0)
        self.assertTrue(report.mean_mrr > 0.0)

        # Check markdown table generation
        md_table = report.to_markdown_table()
        self.assertIn("Hit Rate", md_table)
        self.assertIn("MRR", md_table)
        self.assertIn("Precision@k", md_table)


if __name__ == "__main__":
    unittest.main()
