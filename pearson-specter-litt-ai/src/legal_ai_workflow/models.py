"""Typed data models used across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import stable_id, utc_now_iso


@dataclass
class ExtractionWarning:
    message: str
    severity: str = "warning"
    page: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageText:
    page_number: int
    text: str
    confidence: float
    extraction_method: str


@dataclass
class ProcessedDocument:
    document_id: str
    source_path: str
    file_name: str
    raw_text: str
    structured_data: dict[str, Any]
    pages: list[PageText] = field(default_factory=list)
    warnings: list[ExtractionWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source_path: Path,
        raw_text: str,
        structured_data: dict[str, Any],
        pages: list[PageText],
        warnings: list[ExtractionWarning],
        metadata: dict[str, Any] | None = None,
    ) -> "ProcessedDocument":
        original_source_path = source_path
        resolved_source_path = source_path.resolve()
        document_id = stable_id(str(resolved_source_path) + raw_text[:500])
        base_metadata = {
            "processed_at": utc_now_iso(),
            "source_suffix": original_source_path.suffix.lower(),
        }
        if metadata:
            base_metadata.update(metadata)
        return cls(
            document_id=document_id,
            source_path=str(original_source_path),
            file_name=original_source_path.name,
            raw_text=raw_text,
            structured_data=structured_data,
            pages=pages,
            warnings=warnings,
            metadata=base_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessedDocument":
        pages = [PageText(**item) for item in data.get("pages", [])]
        warnings = [ExtractionWarning(**item) for item in data.get("warnings", [])]
        return cls(
            document_id=data["document_id"],
            source_path=data["source_path"],
            file_name=data["file_name"],
            raw_text=data.get("raw_text", ""),
            structured_data=data.get("structured_data", {}),
            pages=pages,
            warnings=warnings,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source_path=data["source_path"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float

    def citation(self) -> str:
        page = self.chunk.metadata.get("page")
        if page:
            return f"[{self.chunk.chunk_id}; p. {page}]"
        return f"[{self.chunk.chunk_id}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "citation": self.citation(),
        }
