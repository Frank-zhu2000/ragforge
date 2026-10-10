# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-22

### Added
- **Core Chunkers**:
  - `FixedSizeChunker`: Character window chunker with configurable sliding overlap.
  - `TokenChunker`: Natural boundary chunker with CJK and English token estimation.
  - `MarkdownChunker`: Hierarchical markdown heading breadcrumbs preserving context.
  - `CodeChunker`: AST-like statement and function boundary chunking for Python, Go, JS/TS, Java.
- **Data Extractors**:
  - `TextExtractor`: Cleans zero-width chars, control characters, and normalizes whitespaces.
  - `MarkdownExtractor`: Frontmatter YAML parsing, H1 title, and link extraction.
  - `JSONExtractor`: Formatted document extraction for JSON records and JSONL datasets.
- **Indices & Search**:
  - `BM25Index`: Okapi BM25 implementation with English & CJK n-gram tokenization.
  - `VectorIndex`: Dense vector cosine similarity, dot product, and Euclidean search with built-in zero-dependency hashing vectorizer.
  - `HybridIndex`: Reciprocal Rank Fusion (RRF) and Alpha-Score blend search.
- **Evaluation & Benchmarking**:
  - Metric functions: `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`, `hit_rate`, `context_relevance`, `faithfulness`.
  - `RAGEvaluator`: Batch evaluation runner and GitHub Markdown comparison table generator.
- **Developer Experience**:
  - Interactive CLI with ANSI color support (`ragforge chunk`, `ragforge search`).
  - GitHub Actions CI/CD workflows for multi-version Python testing and CodeQL security analysis.
  - 100% unit test coverage across 21 test scenarios.
