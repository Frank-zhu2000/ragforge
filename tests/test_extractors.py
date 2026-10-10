"""Unit tests for extractors."""

import unittest
from ragforge.extractors.text import TextExtractor
from ragforge.extractors.markdown import MarkdownExtractor
from ragforge.extractors.structured import JSONExtractor


class TestExtractors(unittest.TestCase):

    def test_text_extractor(self):
        extractor = TextExtractor()
        raw = "Hello   \r\n\r\n\r\n\r\nWorld!\u200b\ufeff\tTesting."
        doc = extractor.extract_from_string(raw, metadata={"tenant": "unit"})
        self.assertEqual(doc.metadata["tenant"], "unit")
        self.assertNotIn("\u200b", doc.content)
        self.assertNotIn("\ufeff", doc.content)
        self.assertNotIn("\n\n\n", doc.content)
        self.assertEqual(doc.metadata["format"], "text")
        self.assertTrue(doc.metadata["word_count"] >= 3)

    def test_markdown_extractor(self):
        extractor = MarkdownExtractor()
        md = """---
title: System Architecture
author: Frank Zhu
version: 1.0
---

# Main Title

This is a [documentation link](https://github.com/Frank-zhu2000/ragforge).
<b>Clean HTML tag test</b>.
"""
        doc = extractor.extract_from_string(md)
        self.assertEqual(doc.metadata.get("title"), "Main Title")
        self.assertEqual(doc.metadata.get("author"), "Frank Zhu")
        self.assertEqual(doc.metadata.get("version"), "1.0")
        self.assertEqual(len(doc.metadata["links"]), 1)
        self.assertEqual(doc.metadata["links"][0]["url"], "https://github.com/Frank-zhu2000/ragforge")
        self.assertNotIn("<b>", doc.content)
        self.assertNotIn("</b>", doc.content)

    def test_json_extractor(self):
        extractor = JSONExtractor(content_key="body", metadata_keys=["category", "id"])
        json_str = '{"id": 42, "category": "AI", "body": "Retrieval Augmented Generation with Hybrid Search."}'
        doc = extractor.extract_from_string(json_str)
        self.assertEqual(doc.content, "Retrieval Augmented Generation with Hybrid Search.")
        self.assertEqual(doc.metadata["id"], 42)
        self.assertEqual(doc.metadata["category"], "AI")

    def test_jsonl_extractor(self):
        extractor = JSONExtractor(content_key="text")
        jsonl_str = '{"text": "Line 1"}\n{"text": "Line 2"}\n{"text": "Line 3"}'
        docs = extractor.extract_jsonl(jsonl_str)
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].content, "Line 1")
        self.assertEqual(docs[2].content, "Line 3")


if __name__ == "__main__":
    unittest.main()
