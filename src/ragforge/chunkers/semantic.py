"""Semantic and hierarchical Markdown-aware document chunking."""

import re
from typing import Any, Dict, List, Optional
from ragforge.chunkers.base import BaseChunker, Chunk


class MarkdownChunker(BaseChunker):
    """
    Hierarchical Markdown chunker that preserves heading breadcrumbs,
    code blocks, and table integrity in chunk metadata.
    """

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 1000, preserve_breadcrumbs: bool = True):
        self.max_chunk_size = max_chunk_size
        self.preserve_breadcrumbs = preserve_breadcrumbs

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        meta = dict(metadata or {})
        lines = text.splitlines(keepends=True)
        chunks: List[Chunk] = []

        current_headings: Dict[int, str] = {}  # level -> title
        current_section_lines: List[str] = []
        section_start_char = 0
        current_char_offset = 0

        def flush_current_section():
            nonlocal current_section_lines, section_start_char
            if not current_section_lines:
                return

            section_text = "".join(current_section_lines).strip()
            if not section_text:
                current_section_lines = []
                return

            # Construct breadcrumb hierarchy (e.g., ["Chapter 1", "Section 1.2"])
            active_breadcrumbs = [
                current_headings[lvl]
                for lvl in sorted(current_headings.keys())
                if current_headings[lvl]
            ]

            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = len(chunks)
            chunk_meta["chunk_strategy"] = "markdown_hierarchy"
            if self.preserve_breadcrumbs:
                chunk_meta["breadcrumbs"] = list(active_breadcrumbs)
                chunk_meta["current_heading"] = active_breadcrumbs[-1] if active_breadcrumbs else None

            # Sub-split if section exceeds max_chunk_size
            if len(section_text) > self.max_chunk_size:
                sub_parts = self._split_large_section(section_text, self.max_chunk_size)
                sub_offset = section_start_char
                for sub in sub_parts:
                    sub_meta = dict(chunk_meta)
                    sub_meta["chunk_index"] = len(chunks)
                    chunks.append(
                        Chunk(
                            text=sub,
                            doc_id=doc_id,
                            start_char=sub_offset,
                            end_char=sub_offset + len(sub),
                            token_count=max(1, len(sub.split())),
                            metadata=sub_meta,
                        )
                    )
                    sub_offset += len(sub)
            else:
                chunks.append(
                    Chunk(
                        text=section_text,
                        doc_id=doc_id,
                        start_char=section_start_char,
                        end_char=section_start_char + len(section_text),
                        token_count=max(1, len(section_text.split())),
                        metadata=chunk_meta,
                    )
                )

            current_section_lines = []

        for line in lines:
            line_len = len(line)
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())

            if m:
                # We encountered a markdown heading
                flush_current_section()
                level = len(m.group(1))
                title = m.group(2).strip()

                # Clear deeper headings
                keys_to_remove = [k for k in current_headings if k >= level]
                for k in keys_to_remove:
                    del current_headings[k]
                current_headings[level] = title

                section_start_char = current_char_offset
                current_section_lines.append(line)
            else:
                if not current_section_lines:
                    section_start_char = current_char_offset
                current_section_lines.append(line)

            current_char_offset += line_len

        flush_current_section()
        return chunks

    def _split_large_section(self, text: str, max_size: int) -> List[str]:
        paragraphs = text.split("\n\n")
        parts: List[str] = []
        buf = []
        buf_len = 0

        for p in paragraphs:
            if buf_len + len(p) > max_size and buf:
                parts.append("\n\n".join(buf).strip())
                buf = [p]
                buf_len = len(p)
            else:
                buf.append(p)
                buf_len += len(p) + 2

        if buf:
            parts.append("\n\n".join(buf).strip())

        return parts or [text]


class SemanticChunker(BaseChunker):
    """
    Splits text along natural semantic boundaries (paragraphs, topic markers,
    transitions) aiming to preserve contextual integrity.
    """

    def __init__(self, target_chunk_size: int = 500, min_chunk_size: int = 80):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        meta = dict(metadata or {})
        # Split on double line breaks, bullet lists, or section dividers
        raw_paragraphs = re.split(r"(\n\s*\n|(?<=\n)(?=[-*#\d]+\.\s+))", text)
        paragraphs: List[str] = []
        for p in raw_paragraphs:
            clean_p = p.strip()
            if clean_p:
                paragraphs.append(clean_p)

        chunks: List[Chunk] = []
        buffer: List[str] = []
        buf_len = 0
        start_char = 0

        for p in paragraphs:
            p_len = len(p)
            if buf_len + p_len > self.target_chunk_size and buf_len >= self.min_chunk_size:
                combined = "\n\n".join(buffer)
                chunk_meta = dict(meta)
                chunk_meta["chunk_index"] = len(chunks)
                chunk_meta["chunk_strategy"] = "semantic_paragraphs"

                chunks.append(
                    Chunk(
                        text=combined,
                        doc_id=doc_id,
                        start_char=start_char,
                        end_char=start_char + len(combined),
                        token_count=max(1, len(combined.split())),
                        metadata=chunk_meta,
                    )
                )
                start_char += len(combined)
                buffer = [p]
                buf_len = p_len
            else:
                buffer.append(p)
                buf_len += p_len + 2

        if buffer:
            combined = "\n\n".join(buffer)
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = len(chunks)
            chunk_meta["chunk_strategy"] = "semantic_paragraphs"
            chunks.append(
                Chunk(
                    text=combined,
                    doc_id=doc_id,
                    start_char=start_char,
                    end_char=start_char + len(combined),
                    token_count=max(1, len(combined.split())),
                    metadata=chunk_meta,
                )
            )

        return chunks
