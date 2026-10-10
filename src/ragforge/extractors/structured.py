"""Structured JSON and JSONL data extractor."""

import json
from typing import Any, Callable, Dict, List, Optional, Union
from ragforge.chunkers.base import Document
from ragforge.extractors.base import BaseExtractor


class JSONExtractor(BaseExtractor):
    """
    Extracts structured JSON or JSONL records into text-embeddable Document objects.
    Allows template-based string formatting or key-value serialization.
    """

    def __init__(
        self,
        content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        formatter: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.content_key = content_key
        self.metadata_keys = metadata_keys
        self.formatter = formatter

    def extract_from_string(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        meta = dict(metadata or {})
        try:
            parsed = json.loads(content.strip())
        except json.JSONDecodeError:
            # Fallback for plain text or malformed JSON
            return Document(content=content, metadata={"format": "raw_json_fallback"})

        if isinstance(parsed, dict):
            return self._record_to_document(parsed, meta)
        elif isinstance(parsed, list):
            # Aggregate multiple records into a readable string
            docs = [self._record_to_document(item, {}).content for item in parsed if isinstance(item, dict)]
            aggregated_content = "\n\n---\n\n".join(docs)
            meta["record_count"] = len(parsed)
            meta["format"] = "json_array"
            return Document(content=aggregated_content, metadata=meta)

        return Document(content=str(parsed), metadata=meta)

    def _record_to_document(self, record: Dict[str, Any], base_meta: Dict[str, Any]) -> Document:
        meta = dict(base_meta)

        # Extract selected metadata keys
        if self.metadata_keys:
            for k in self.metadata_keys:
                if k in record:
                    meta[k] = record[k]
        else:
            # By default copy scalar keys to metadata
            for k, v in record.items():
                if isinstance(v, (str, int, float, bool)) and k != self.content_key:
                    meta[k] = v

        if self.formatter:
            text = self.formatter(record)
        elif self.content_key and self.content_key in record:
            text = str(record[self.content_key])
        else:
            # Format all key-values cleanly
            lines = [f"{k}: {v}" for k, v in record.items()]
            text = "\n".join(lines)

        meta["format"] = "json_record"
        return Document(content=text, metadata=meta)

    def extract_jsonl(self, jsonl_content: str) -> List[Document]:
        """Extract lines of JSON objects into a list of Documents."""
        docs: List[Document] = []
        for line in jsonl_content.splitlines():
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        docs.append(self._record_to_document(record, {}))
                except json.JSONDecodeError:
                    continue
        return docs
