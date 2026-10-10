"""Plain text cleaning and normalization extractor."""

import re
from typing import Any, Dict, Optional
from ragforge.chunkers.base import Document
from ragforge.extractors.base import BaseExtractor


class TextExtractor(BaseExtractor):
    """
    Cleans raw text by stripping extraneous zero-width chars,
    normalizing whitespace, removing bad control codes, and extracting metrics.
    """

    def __init__(self, strip_excess_blank_lines: bool = True, remove_control_chars: bool = True):
        self.strip_excess_blank_lines = strip_excess_blank_lines
        self.remove_control_chars = remove_control_chars

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove zero-width characters and BOM
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

        # Remove non-printable control characters except \n, \t
        if self.remove_control_chars:
            text = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)

        # Collapse 3+ consecutive newlines down to 2
        if self.strip_excess_blank_lines:
            text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def extract_from_string(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        meta = dict(metadata or {})
        cleaned = self.clean_text(content)

        meta.update({
            "char_count": len(cleaned),
            "line_count": len(cleaned.splitlines()),
            "word_count": len(cleaned.split()),
            "format": "text",
        })

        return Document(content=cleaned, metadata=meta)
