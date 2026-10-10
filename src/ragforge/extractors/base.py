"""Base interface for document extractors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ragforge.chunkers.base import Document


class BaseExtractor(ABC):
    """Abstract base class for extracting content into structured Document objects."""

    @abstractmethod
    def extract_from_string(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Extract and clean raw string content into a Document."""
        pass

    def extract_from_file(self, file_path: str, encoding: str = "utf-8") -> Document:
        """Extract and clean document content directly from a local file path."""
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()
        metadata = {
            "source_file": file_path,
            "encoding": encoding,
        }
        return self.extract_from_string(content, metadata=metadata)
