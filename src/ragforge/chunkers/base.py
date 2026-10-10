"""Base classes and data structures for document chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Chunk:
    """Represents a text chunk produced by a chunker."""
    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.end_char and self.text:
            self.end_char = self.start_char + len(self.text)
        if not self.token_count and self.text:
            # Simple whitespace estimation if token_count wasn't computed
            self.token_count = max(1, len(self.text.split()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data.get("chunk_id", str(uuid.uuid4())),
            doc_id=data.get("doc_id"),
            text=data["text"],
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document:
    """Represents a source document."""
    content: str
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            doc_id=data.get("doc_id", str(uuid.uuid4())),
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class BaseChunker(ABC):
    """Abstract base class for all chunking algorithms."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        """Chunk a single raw text string."""
        pass

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a Document object, propagating its metadata and doc_id."""
        return self.chunk(
            text=document.content,
            metadata=document.metadata,
            doc_id=document.doc_id,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Batch chunk multiple documents."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
