"""Token and character-based chunking with sliding overlap windows."""

import re
from typing import Any, Dict, List, Optional
from ragforge.chunkers.base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed character-length segments with configurable overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        if not text:
            return []

        meta = dict(metadata or {})
        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            sub_text = text[start:end]

            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = len(chunks)
            chunk_meta["chunk_strategy"] = "fixed_size"

            chunks.append(
                Chunk(
                    text=sub_text,
                    doc_id=doc_id,
                    start_char=start,
                    end_char=end,
                    token_count=max(1, len(sub_text.split())),
                    metadata=chunk_meta,
                )
            )

            if end == total_len:
                break
            start += step

        return chunks


class TokenChunker(BaseChunker):
    """
    Chunks text based on token/word estimation while respecting
    natural sentence and paragraph boundaries whenever possible.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]

    def __init__(
        self,
        target_tokens: int = 256,
        overlap_tokens: int = 32,
        separators: Optional[List[str]] = None,
    ):
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive.")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative.")
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be strictly less than target_tokens.")

        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (supports English words and CJK characters)."""
        # CJK characters count as roughly 1 token each; English words count by spaces
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        non_cjk_words = len(re.findall(r"[a-zA-Z0-9_\-]+", text))
        return max(1, cjk_chars + non_cjk_words)

    def _split_into_units(self, text: str) -> List[str]:
        """Split text recursively into small coherent units."""
        # Split by paragraphs, CJK punctuation, and English sentence enders
        pattern = r"(\n\n|\r\n\r\n|[\n。！？!?]|\.\s+|;\s+)"
        parts = re.split(pattern, text)
        units: List[str] = []
        for i in range(0, len(parts), 2):
            unit = parts[i]
            if i + 1 < len(parts):
                unit += parts[i + 1]
            if unit.strip():
                units.append(unit)
        if not units and text.strip():
            units = [text]
        return units

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        if not text.strip():
            return []

        meta = dict(metadata or {})
        units = self._split_into_units(text)
        chunks: List[Chunk] = []

        current_units: List[str] = []
        current_unit_indices: List[int] = []
        current_tokens = 0

        unit_char_offsets = []
        offset = 0
        for u in units:
            pos = text.find(u, offset)
            if pos == -1:
                pos = offset
            unit_char_offsets.append(pos)
            offset = pos + len(u)

        i = 0
        while i < len(units):
            unit = units[i]
            u_tokens = self._estimate_tokens(unit)

            if current_units and (current_tokens + u_tokens > self.target_tokens):
                chunk_str = "".join(current_units)
                start_c = unit_char_offsets[current_unit_indices[0]]
                end_c = start_c + len(chunk_str)

                chunk_meta = dict(meta)
                chunk_meta["chunk_index"] = len(chunks)
                chunk_meta["chunk_strategy"] = "token_boundary"

                chunks.append(
                    Chunk(
                        text=chunk_str.strip(),
                        doc_id=doc_id,
                        start_char=start_c,
                        end_char=end_c,
                        token_count=current_tokens,
                        metadata=chunk_meta,
                    )
                )

                # Overlap step: keep trailing units that fit within overlap_tokens
                overlap_units: List[str] = []
                overlap_indices: List[int] = []
                overlap_t = 0
                for back_u, back_idx in zip(reversed(current_units), reversed(current_unit_indices)):
                    bt = self._estimate_tokens(back_u)
                    if overlap_t + bt <= self.overlap_tokens:
                        overlap_units.insert(0, back_u)
                        overlap_indices.insert(0, back_idx)
                        overlap_t += bt
                    else:
                        break

                current_units = list(overlap_units)
                current_unit_indices = list(overlap_indices)
                current_tokens = overlap_t

            current_units.append(unit)
            current_unit_indices.append(i)
            current_tokens += u_tokens
            i += 1

        if current_units:
            chunk_str = "".join(current_units)
            start_c = unit_char_offsets[current_unit_indices[0]]
            end_c = start_c + len(chunk_str)

            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = len(chunks)
            chunk_meta["chunk_strategy"] = "token_boundary"

            chunks.append(
                Chunk(
                    text=chunk_str.strip(),
                    doc_id=doc_id,
                    start_char=start_c,
                    end_char=end_c,
                    token_count=current_tokens,
                    metadata=chunk_meta,
                )
            )

        return chunks
