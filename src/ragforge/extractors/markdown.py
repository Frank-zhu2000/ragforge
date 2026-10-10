"""Markdown-specific extractor with frontmatter and structural parsing."""

import re
from typing import Any, Dict, List, Optional, Tuple
from ragforge.chunkers.base import Document
from ragforge.extractors.base import BaseExtractor


class MarkdownExtractor(BaseExtractor):
    """
    Parses Markdown files, extracting YAML frontmatter, headers,
    outbound links, and structured section outlines.
    """

    FRONTMATTER_REGEX = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    LINK_REGEX = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    IMAGE_REGEX = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def __init__(self, parse_frontmatter: bool = True, strip_html_tags: bool = True):
        self.parse_frontmatter = parse_frontmatter
        self.strip_html_tags = strip_html_tags

    def _extract_frontmatter(self, text: str) -> Tuple[Dict[str, Any], str]:
        """Simple standard library YAML frontmatter parser."""
        metadata: Dict[str, Any] = {}
        cleaned_text = text

        m = self.FRONTMATTER_REGEX.match(text)
        if m:
            raw_yaml = m.group(1)
            cleaned_text = text[m.end():]
            for line in raw_yaml.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip("\"'")

        return metadata, cleaned_text

    def extract_from_string(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        meta = dict(metadata or {})
        working_text = content

        if self.parse_frontmatter:
            fm_meta, working_text = self._extract_frontmatter(working_text)
            meta.update(fm_meta)

        # Extract links
        links = self.LINK_REGEX.findall(working_text)
        meta["links"] = [{"text": text, "url": url} for text, url in links[:20]]

        # Extract document title (first H1)
        h1_match = re.search(r"^#\s+(.+)$", working_text, re.MULTILINE)
        if h1_match:
            meta["title"] = h1_match.group(1).strip()

        # Clean text
        if self.strip_html_tags:
            working_text = re.sub(r"<[^>]+>", "", working_text)

        meta.update({
            "format": "markdown",
            "char_count": len(working_text),
            "line_count": len(working_text.splitlines()),
        })

        return Document(content=working_text.strip(), metadata=meta)
