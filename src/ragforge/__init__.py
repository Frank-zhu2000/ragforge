"""
RAGForge: A lightweight, fast, and modular Python toolkit for RAG data pipelines.

Core modules:
- chunkers: Smart text, code, and hierarchical markdown chunking.
- extractors: Clean and structure text, JSON, and markdown data.
- indices: BM25 sparse keyword search, dense vector search, and hybrid reciprocal rank fusion (RRF).
- eval: Comprehensive evaluation metrics for RAG retrieval and generation.
"""

from ragforge.chunkers.base import Chunk, Document
from ragforge.chunkers.token import FixedSizeChunker, TokenChunker
from ragforge.chunkers.semantic import MarkdownChunker, SemanticChunker
from ragforge.chunkers.code import CodeChunker

from ragforge.extractors.base import BaseExtractor
from ragforge.extractors.text import TextExtractor
from ragforge.extractors.markdown import MarkdownExtractor
from ragforge.extractors.structured import JSONExtractor

from ragforge.indices.bm25 import BM25Index
from ragforge.indices.vector import VectorIndex
from ragforge.indices.hybrid import HybridIndex

from ragforge.eval.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
)
from ragforge.eval.evaluator import RAGEvaluator, EvaluationReport, EvaluationSample

__version__ = "0.1.0"
__author__ = "Frank Zhu (Frank-zhu2000)"

__all__ = [
    # Data structures
    "Chunk",
    "Document",
    # Chunkers
    "FixedSizeChunker",
    "TokenChunker",
    "SemanticChunker",
    "MarkdownChunker",
    "CodeChunker",
    # Extractors
    "BaseExtractor",
    "TextExtractor",
    "MarkdownExtractor",
    "JSONExtractor",
    # Indices
    "BM25Index",
    "VectorIndex",
    "HybridIndex",
    # Evaluation
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
    "RAGEvaluator",
    "EvaluationReport",
    "EvaluationSample",
]
