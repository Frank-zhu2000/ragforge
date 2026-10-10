"""01_quickstart.py: Basic semantic chunking and text extraction example."""

import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ragforge import MarkdownChunker, MarkdownExtractor, TextExtractor


def main():
    sample_markdown = """---
title: Understanding Modern RAG Systems
author: Frank Zhu
tags: [ai, rag, llm]
---

# Introduction to RAG

Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM responses by grounding the model on external knowledge bases.

## The Ingestion Pipeline

During ingestion, documents are cleaned, divided into discrete semantic chunks, and embedded into high-dimensional vectors.

### Why Chunking Matters
Naive chunking often cuts sentences in half, causing context fragmentation. Semantic chunking preserves contextual boundaries.

## Retrieval & Generation

When a user submits a query, the retrieval engine finds the most relevant passages using vector similarity or hybrid search before handing them to the generator LLM.
"""

    print("=== Step 1: Extract Document and Metadata ===")
    extractor = MarkdownExtractor()
    doc = extractor.extract_from_string(sample_markdown)
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Author: {doc.metadata.get('author')}")
    print(f"Total Characters: {doc.metadata.get('char_count')}")

    print("\n=== Step 2: Hierarchical Markdown Chunking ===")
    chunker = MarkdownChunker(max_chunk_size=300)
    chunks = chunker.chunk_document(doc)
    print(f"Generated {len(chunks)} chunks:\n")

    for i, c in enumerate(chunks, 1):
        breadcrumbs = " > ".join(c.metadata.get("breadcrumbs", []))
        print(f"[Chunk #{i}] Path: {breadcrumbs}")
        print(f"Content:\n{c.text.strip()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
