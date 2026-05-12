"""Chunking, embedding, and retrieval for grounded generation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import Chunk, ProcessedDocument, RetrievalResult
from .utils import ensure_dir, normalize_text, stable_id, tokenize


class Chunker:
    """Splits processed documents into paragraph/window chunks with metadata."""

    def __init__(self, target_tokens: int = 120, overlap_tokens: int = 25) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, document: ProcessedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        counter = 1
        pages = document.pages or []
        if not pages and document.raw_text:
            pages = []
        for page in pages:
            for text in self._chunk_text(page.text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-C{counter:03d}",
                        document_id=document.document_id,
                        source_path=document.source_path,
                        text=text,
                        metadata={
                            "file_name": document.file_name,
                            "page": page.page_number,
                            "confidence": page.confidence,
                            "extraction_method": page.extraction_method,
                            "document_type": document.structured_data.get("document_type"),
                        },
                    )
                )
                counter += 1
        if not chunks and document.raw_text:
            for text in self._chunk_text(document.raw_text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-C{counter:03d}",
                        document_id=document.document_id,
                        source_path=document.source_path,
                        text=text,
                        metadata={"file_name": document.file_name, "document_type": document.structured_data.get("document_type")},
                    )
                )
                counter += 1
        return chunks

    def chunk_documents(self, documents: Iterable[ProcessedDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current_tokens: list[str] = []
        current_text: list[str] = []

        for paragraph in paragraphs:
            paragraph_tokens = tokenize(paragraph)
            if not paragraph_tokens:
                continue
            if current_tokens and len(current_tokens) + len(paragraph_tokens) > self.target_tokens:
                chunks.append("\n\n".join(current_text))
                current_text, current_tokens = self._paragraph_overlap(current_text)
            current_tokens.extend(paragraph_tokens)
            current_text.append(paragraph)

        if current_text:
            chunks.append("\n\n".join(current_text))
        return [chunk for chunk in chunks if tokenize(chunk)]

    def _paragraph_overlap(self, paragraphs: list[str]) -> tuple[list[str], list[str]]:
        if not self.overlap_tokens:
            return [], []
        kept_text: list[str] = []
        kept_tokens: list[str] = []
        for paragraph in reversed(paragraphs):
            paragraph_tokens = tokenize(paragraph)
            if kept_tokens and len(kept_tokens) + len(paragraph_tokens) > self.overlap_tokens:
                break
            kept_text.insert(0, paragraph)
            kept_tokens = paragraph_tokens + kept_tokens
            if len(kept_tokens) >= self.overlap_tokens:
                break
        return kept_text, kept_tokens


class TfidfEmbeddingModel:
    """Dependency-free sparse TF-IDF embeddings for local review/demo runs."""

    name = "local-tfidf"

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        doc_count = max(len(texts), 1)
        document_frequency: dict[str, int] = defaultdict(int)
        for text in texts:
            for token in set(tokenize(text)):
                document_frequency[token] += 1
        self.idf = {
            token: math.log((1 + doc_count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def embed(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        max_tf = max(counts.values())
        vector = {
            token: (0.5 + 0.5 * (count / max_tf)) * self.idf.get(token, 1.0)
            for token, count in counts.items()
        }
        return vector

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "idf": self.idf}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TfidfEmbeddingModel":
        model = cls()
        model.idf = {str(key): float(value) for key, value in dict(data.get("idf", {})).items()}
        return model


def sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorStore:
    """Small inspectable vector store backed by JSON files."""

    def __init__(self, embedding_model: TfidfEmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or TfidfEmbeddingModel()
        self.chunks: list[Chunk] = []
        self.vectors: list[dict[str, float]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.embedding_model.fit([chunk.text for chunk in chunks])
        self.vectors = [self.embedding_model.embed(chunk.text) for chunk in chunks]

    def query(self, query: str, top_k: int = 6, min_score: float = 0.0) -> list[RetrievalResult]:
        query_vector = self.embedding_model.embed(query)
        scored = [
            RetrievalResult(chunk=chunk, score=sparse_cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [item for item in scored if item.score >= min_score][:top_k]

    def save(self, index_dir: str | Path) -> None:
        index_dir = ensure_dir(Path(index_dir))
        payload = {
            "embedding_model": self.embedding_model.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": self.vectors,
        }
        (index_dir / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        path = Path(index_dir) / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = TfidfEmbeddingModel.from_dict(payload["embedding_model"])
        store = cls(model)
        store.chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        store.vectors = [{str(k): float(v) for k, v in vector.items()} for vector in payload.get("vectors", [])]
        return store


def load_processed_documents(processed_dir: str | Path) -> list[ProcessedDocument]:
    documents: list[ProcessedDocument] = []
    for path in sorted(Path(processed_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        documents.append(ProcessedDocument.from_dict(data))
    return documents


def build_index_from_processed(processed_dir: str | Path, index_dir: str | Path) -> VectorStore:
    documents = load_processed_documents(processed_dir)
    chunks = Chunker().chunk_documents(documents)
    store = VectorStore()
    store.build(chunks)
    store.save(index_dir)
    manifest = {
        "index_id": stable_id("".join(chunk.chunk_id for chunk in chunks)),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_model": store.embedding_model.name,
    }
    (Path(index_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store
