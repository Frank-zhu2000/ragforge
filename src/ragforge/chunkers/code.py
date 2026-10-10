"""Code-aware chunking preserving function/class boundaries and signatures."""

import re
from typing import Any, Dict, List, Optional
from ragforge.chunkers.base import BaseChunker, Chunk


class CodeChunker(BaseChunker):
    """
    Code-aware chunker that splits source code by functions, classes,
    or statement blocks, avoiding mid-statement cuts.
    """

    # Regex patterns for Python, JS/TS, Go, Java, Rust
    DEF_PATTERNS = [
        # Python
        r"^(?:async\s+)?def\s+([a-zA-Z0-9_]+)",
        r"^class\s+([a-zA-Z0-9_]+)",
        # JS/TS
        r"^(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)",
        r"^(?:export\s+)?class\s+([a-zA-Z0-9_]+)",
        r"^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?\([^\)]*\)\s*=>",
        # Go
        r"^func\s+(?:\([^\)]+\)\s+)?([a-zA-Z0-9_]+)",
        r"^type\s+([a-zA-Z0-9_]+)\s+struct",
        # Java / C++
        r"^(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(",
    ]

    def __init__(self, max_lines: int = 60, min_lines: int = 5):
        self.max_lines = max_lines
        self.min_lines = min_lines
        self._compiled_patterns = [re.compile(p, re.MULTILINE) for p in self.DEF_PATTERNS]

    def _find_symbol_name(self, line: str) -> Optional[str]:
        for pattern in self._compiled_patterns:
            m = pattern.search(line.strip())
            if m and m.groups():
                return m.group(1)
        return None

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        meta = dict(metadata or {})
        lines = text.splitlines(keepends=True)
        chunks: List[Chunk] = []

        current_block: List[str] = []
        current_symbol: Optional[str] = None
        current_start_line = 1
        current_start_char = 0
        running_char = 0

        for line_idx, line in enumerate(lines, 1):
            symbol = self._find_symbol_name(line)

            # Check if this line starts a new definition and the current block is big enough
            if symbol and len(current_block) >= self.min_lines:
                chunk_str = "".join(current_block)
                chunk_meta = dict(meta)
                chunk_meta["chunk_index"] = len(chunks)
                chunk_meta["chunk_strategy"] = "code_boundary"
                chunk_meta["start_line"] = current_start_line
                chunk_meta["end_line"] = line_idx - 1
                if current_symbol:
                    chunk_meta["symbol_name"] = current_symbol

                chunks.append(
                    Chunk(
                        text=chunk_str.rstrip(),
                        doc_id=doc_id,
                        start_char=current_start_char,
                        end_char=current_start_char + len(chunk_str),
                        token_count=max(1, len(chunk_str.split())),
                        metadata=chunk_meta,
                    )
                )

                current_start_char = running_char
                current_start_line = line_idx
                current_block = [line]
                current_symbol = symbol
            else:
                if not current_block:
                    current_start_char = running_char
                    current_start_line = line_idx
                if symbol and not current_symbol:
                    current_symbol = symbol

                current_block.append(line)

                # Hard limit check
                if len(current_block) >= self.max_lines:
                    chunk_str = "".join(current_block)
                    chunk_meta = dict(meta)
                    chunk_meta["chunk_index"] = len(chunks)
                    chunk_meta["chunk_strategy"] = "code_boundary"
                    chunk_meta["start_line"] = current_start_line
                    chunk_meta["end_line"] = line_idx
                    if current_symbol:
                        chunk_meta["symbol_name"] = current_symbol

                    chunks.append(
                        Chunk(
                            text=chunk_str.rstrip(),
                            doc_id=doc_id,
                            start_char=current_start_char,
                            end_char=current_start_char + len(chunk_str),
                            token_count=max(1, len(chunk_str.split())),
                            metadata=chunk_meta,
                        )
                    )
                    current_start_char = running_char + len(line)
                    current_start_line = line_idx + 1
                    current_block = []
                    current_symbol = None

            running_char += len(line)

        if current_block:
            chunk_str = "".join(current_block)
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = len(chunks)
            chunk_meta["chunk_strategy"] = "code_boundary"
            chunk_meta["start_line"] = current_start_line
            chunk_meta["end_line"] = len(lines)
            if current_symbol:
                chunk_meta["symbol_name"] = current_symbol

            chunks.append(
                Chunk(
                    text=chunk_str.rstrip(),
                    doc_id=doc_id,
                    start_char=current_start_char,
                    end_char=current_start_char + len(chunk_str),
                    token_count=max(1, len(chunk_str.split())),
                    metadata=chunk_meta,
                )
            )

        return chunks
