"""Unit tests for chunking algorithms."""

import unittest
from ragforge.chunkers.base import Chunk, Document
from ragforge.chunkers.code import CodeChunker
from ragforge.chunkers.semantic import MarkdownChunker, SemanticChunker
from ragforge.chunkers.token import FixedSizeChunker, TokenChunker


class TestChunkers(unittest.TestCase):

    def test_chunk_dataclass(self):
        c = Chunk(text="Hello world")
        self.assertEqual(c.text, "Hello world")
        self.assertEqual(c.start_char, 0)
        self.assertEqual(c.end_char, 11)
        self.assertEqual(c.token_count, 2)
        d = c.to_dict()
        self.assertEqual(d["text"], "Hello world")
        c2 = Chunk.from_dict(d)
        self.assertEqual(c2.text, c.text)

    def test_fixed_size_chunker(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 120
        chunks = chunker.chunk(text)
        self.assertTrue(len(chunks) >= 3)
        self.assertEqual(chunks[0].text, "A" * 50)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[0].end_char, 50)

        # Edge cases
        self.assertEqual(chunker.chunk(""), [])
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=15)

    def test_token_chunker(self):
        chunker = TokenChunker(target_tokens=6, overlap_tokens=2)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk(text)
        self.assertTrue(len(chunks) >= 2)
        for c in chunks:
            self.assertTrue(c.token_count > 0)
            self.assertTrue(len(c.text) > 0)

        # Chinese token chunker test
        zh_text = "自然语言处理是人工智能领域的重要分支。语义检索和大语言模型结合带来了RAG范式。"
        zh_chunks = chunker.chunk(zh_text)
        self.assertTrue(len(zh_chunks) >= 1)

    def test_markdown_chunker(self):
        md_text = """# Machine Learning
Introduction to ML.

## Supervised Learning
Supervised learning requires labeled data.

### Classification
Classification predicts discrete classes.

## Unsupervised Learning
Clustering finds patterns.
"""
        chunker = MarkdownChunker(max_chunk_size=500)
        chunks = chunker.chunk(md_text)
        self.assertTrue(len(chunks) >= 3)

        # Check breadcrumbs
        classification_chunk = next(c for c in chunks if "Classification" in c.text)
        breadcrumbs = classification_chunk.metadata.get("breadcrumbs", [])
        self.assertIn("Machine Learning", breadcrumbs)
        self.assertIn("Supervised Learning", breadcrumbs)
        self.assertIn("Classification", breadcrumbs)

    def test_code_chunker(self):
        code_text = """def function_one():
    x = 1
    y = 2
    return x + y

class MyService:
    def __init__(self):
        self.active = True

    def process(self):
        return self.active
"""
        chunker = CodeChunker(max_lines=15, min_lines=2)
        chunks = chunker.chunk(code_text)
        self.assertTrue(len(chunks) >= 1)
        for c in chunks:
            self.assertEqual(c.metadata["chunk_strategy"], "code_boundary")
            self.assertTrue("start_line" in c.metadata)

    def test_chunk_document_flow(self):
        doc = Document(content="Sample document text.", doc_id="doc_101", metadata={"source": "manual"})
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "doc_101")
        self.assertEqual(chunks[0].metadata["source"], "manual")


if __name__ == "__main__":
    unittest.main()
