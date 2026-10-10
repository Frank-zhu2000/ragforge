"""RAG pipeline evaluation runner and report generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ragforge.eval.metrics import (
    context_relevance,
    faithfulness,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class EvaluationSample:
    """A single evaluation query sample."""
    query: str
    retrieved_chunk_ids: List[str]
    ground_truth_chunk_ids: Set[str]
    context_text: Optional[str] = None
    generated_answer: Optional[str] = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation benchmark report."""
    total_samples: int
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_context_relevance: float
    mean_faithfulness: float
    k: int
    per_sample_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "k": self.k,
            "hit_rate_at_k": round(self.hit_rate_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg_at_k": round(self.mean_ndcg_at_k, 4),
            "mean_context_relevance": round(self.mean_context_relevance, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
        }

    def to_markdown_table(self) -> str:
        """Render report as a GitHub-flavored markdown table."""
        return f"""| Metric | Value (@k={self.k}) | Description |
| :--- | :---: | :--- |
| **Hit Rate** | `{self.hit_rate_at_k:.2%}` | Proportion of queries where relevant chunk was found |
| **MRR** | `{self.mean_mrr:.4f}` | Mean Reciprocal Rank of first relevant document |
| **Precision@k** | `{self.precision_at_k:.2%}` | Ratio of retrieved chunks that are relevant |
| **Recall@k** | `{self.recall_at_k:.2%}` | Ratio of ground truth chunks successfully retrieved |
| **NDCG@k** | `{self.mean_ndcg_at_k:.4f}` | Normalized Discounted Cumulative Gain ranking quality |
| **Context Relevance** | `{self.mean_context_relevance:.2%}` | Question-to-context semantic overlap score |
| **Faithfulness** | `{self.mean_faithfulness:.2%}` | Hallucination avoidance / context grounding score |"""


class RAGEvaluator:
    """Evaluates a batch of RAG retrieval and generation results."""

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(self, samples: List[EvaluationSample]) -> EvaluationReport:
        if not samples:
            return EvaluationReport(
                total_samples=0,
                hit_rate_at_k=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                mean_mrr=0.0,
                mean_ndcg_at_k=0.0,
                mean_context_relevance=0.0,
                mean_faithfulness=0.0,
                k=self.k,
            )

        hr_scores = []
        p_scores = []
        r_scores = []
        mrr_scores = []
        ndcg_scores = []
        rel_scores = []
        faith_scores = []
        details = []

        for s in samples:
            gt = set(s.ground_truth_chunk_ids)
            hr = hit_rate(s.retrieved_chunk_ids, gt, k=self.k)
            p = precision_at_k(s.retrieved_chunk_ids, gt, k=self.k)
            r = recall_at_k(s.retrieved_chunk_ids, gt, k=self.k)
            m = mrr(s.retrieved_chunk_ids, gt)
            ndcg = ndcg_at_k(s.retrieved_chunk_ids, gt, k=self.k)

            hr_scores.append(hr)
            p_scores.append(p)
            r_scores.append(r)
            mrr_scores.append(m)
            ndcg_scores.append(ndcg)

            # Optional generation metrics
            if s.context_text:
                rel = context_relevance(s.query, s.context_text)
                rel_scores.append(rel)
                if s.generated_answer:
                    f = faithfulness(s.generated_answer, s.context_text)
                    faith_scores.append(f)

            details.append({
                "query": s.query,
                "hit_rate": hr,
                "precision": p,
                "recall": r,
                "mrr": m,
                "ndcg": ndcg,
            })

        n = len(samples)
        return EvaluationReport(
            total_samples=n,
            hit_rate_at_k=sum(hr_scores) / n,
            precision_at_k=sum(p_scores) / n,
            recall_at_k=sum(r_scores) / n,
            mean_mrr=sum(mrr_scores) / n,
            mean_ndcg_at_k=sum(ndcg_scores) / n,
            mean_context_relevance=(sum(rel_scores) / len(rel_scores)) if rel_scores else 1.0,
            mean_faithfulness=(sum(faith_scores) / len(faith_scores)) if faith_scores else 1.0,
            k=self.k,
            per_sample_details=details,
        )
