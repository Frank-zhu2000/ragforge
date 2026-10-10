"""Command-line interface entrypoint for ragforge."""

import argparse
import json
import os
import sys
from typing import List

from ragforge import __version__
from ragforge.chunkers.code import CodeChunker
from ragforge.chunkers.semantic import MarkdownChunker, SemanticChunker
from ragforge.chunkers.token import FixedSizeChunker, TokenChunker
from ragforge.indices.bm25 import BM25Index
from ragforge.indices.hybrid import HybridIndex
from ragforge.indices.vector import VectorIndex


class TerminalColors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    # Check if stdout is a tty or supports colors
    if os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return text
    return f"{color}{text}{TerminalColors.RESET}"


def cmd_chunk(args: argparse.Namespace) -> None:
    if not os.path.exists(args.file):
        print(colorize(f"Error: File '{args.file}' not found.", TerminalColors.RED))
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Select chunker
    strategy = args.strategy.lower()
    if strategy == "fixed":
        chunker = FixedSizeChunker(chunk_size=args.size, chunk_overlap=args.overlap)
    elif strategy == "token":
        chunker = TokenChunker(target_tokens=args.size, overlap_tokens=args.overlap)
    elif strategy == "semantic":
        chunker = SemanticChunker(target_chunk_size=args.size)
    elif strategy == "markdown":
        chunker = MarkdownChunker(max_chunk_size=args.size)
    elif strategy == "code":
        chunker = CodeChunker(max_lines=max(10, args.size // 20))
    else:
        print(colorize(f"Unknown strategy '{strategy}'.", TerminalColors.RED))
        sys.exit(1)

    chunks = chunker.chunk(content, doc_id=os.path.basename(args.file))

    if args.output == "json":
        data = [c.to_dict() for c in chunks]
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(colorize(f"=== RAGForge Chunking Report ===", TerminalColors.CYAN + TerminalColors.BOLD))
        print(f"File: {args.file} | Strategy: {strategy} | Total Chunks: {len(chunks)}\n")
        for i, c in enumerate(chunks, 1):
            strategy_tag = c.metadata.get("chunk_strategy", "default")
            heading = c.metadata.get("current_heading")
            heading_info = f" [{heading}]" if heading else ""
            print(colorize(f"--- Chunk #{i} ({len(c.text)} chars, ~{c.token_count} tokens) {strategy_tag}{heading_info} ---", TerminalColors.GREEN))
            print(c.text)
            print()


def cmd_search(args: argparse.Namespace) -> None:
    if not os.path.exists(args.file):
        print(colorize(f"Error: File '{args.file}' not found.", TerminalColors.RED))
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    chunker = SemanticChunker(target_chunk_size=args.chunk_size)
    chunks = chunker.chunk(content, doc_id=os.path.basename(args.file))

    mode = args.mode.lower()
    if mode == "bm25":
        index = BM25Index()
        index.add_chunks(chunks)
        results = index.search(args.query, top_k=args.top_k)
    elif mode == "vector":
        index = VectorIndex()
        index.add_chunks(chunks)
        results = index.search(args.query, top_k=args.top_k)
    elif mode == "hybrid":
        index = HybridIndex()
        index.add_chunks(chunks)
        results = index.search(args.query, top_k=args.top_k)
    else:
        print(colorize(f"Unknown mode '{mode}'. Use 'bm25', 'vector', or 'hybrid'.", TerminalColors.RED))
        sys.exit(1)

    print(colorize(f"=== RAGForge Search Results ({mode.upper()}) ===", TerminalColors.CYAN + TerminalColors.BOLD))
    print(f"Query: \"{args.query}\" | Indexed Chunks: {len(chunks)} | Top Matches: {len(results)}\n")

    for r in results:
        print(colorize(f"Rank #{r.rank} | Score: {r.score:.4f} | Chunk ID: {r.chunk.chunk_id[:8]}", TerminalColors.YELLOW))
        print(r.chunk.text)
        print("-" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="RAGForge: Lightweight, modular toolkit for RAG pipelines and semantic chunking.",
    )
    parser.add_argument("--version", "-v", action="version", version=f"ragforge {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: chunk
    p_chunk = subparsers.add_parser("chunk", help="Chunk a text or markdown file")
    p_chunk.add_argument("file", help="Path to text or markdown file")
    p_chunk.add_argument("--strategy", "-s", default="semantic", choices=["fixed", "token", "semantic", "markdown", "code"])
    p_chunk.add_argument("--size", type=int, default=400, help="Target chunk size in characters or tokens")
    p_chunk.add_argument("--overlap", type=int, default=40, help="Overlap size")
    p_chunk.add_argument("--output", "-o", default="text", choices=["text", "json"], help="Output format")

    # Command: search
    p_search = subparsers.add_parser("search", help="Perform keyword, vector, or hybrid search over a file")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--file", "-f", required=True, help="Path to file to index and search")
    p_search.add_argument("--mode", "-m", default="hybrid", choices=["hybrid", "bm25", "vector"])
    p_search.add_argument("--top-k", "-k", type=int, default=3, help="Number of results to return")
    p_search.add_argument("--chunk-size", type=int, default=400, help="Chunk size for indexation")

    return parser


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "chunk":
        cmd_chunk(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
