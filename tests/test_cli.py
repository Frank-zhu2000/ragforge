"""Unit tests for CLI parser and subcommands."""

import os
import tempfile
import unittest
from ragforge.cli.main import build_parser, cmd_chunk, cmd_search


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8")
        self.temp_file.write("# Heading 1\nSection 1 text.\n\n# Heading 2\nSection 2 text.")
        self.temp_file.flush()
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_parser_construction(self):
        parser = build_parser()
        args = parser.parse_args(["chunk", self.temp_file.name, "--strategy", "markdown", "--output", "json"])
        self.assertEqual(args.command, "chunk")
        self.assertEqual(args.strategy, "markdown")
        self.assertEqual(args.output, "json")

    def test_search_parser(self):
        parser = build_parser()
        args = parser.parse_args(["search", "Section 1", "--file", self.temp_file.name, "--mode", "bm25", "--top-k", "1"])
        self.assertEqual(args.command, "search")
        self.assertEqual(args.query, "Section 1")
        self.assertEqual(args.mode, "bm25")
        self.assertEqual(args.top_k, 1)


if __name__ == "__main__":
    unittest.main()
