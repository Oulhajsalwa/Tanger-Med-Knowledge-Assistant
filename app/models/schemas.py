
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    ANNUAL_REPORT = "annual_report"
    RSE_REPORT = "rse_report"


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    GENERAL_SYNTHESIS = "general_synthesis"


class ChunkMetadata(BaseModel):
   

    document_name: str
    document_type: DocumentType
    year: int
    page_start: int
    page_end: int
    section: Optional[str] = None
    source_url: Optional[str] = None
    source_file: str
    chunk_index: int
    chunk_id: str

    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"p. {self.page_start}-{self.page_end}"

    def to_chroma_metadata(self) -> dict:
        """Aplatit en types primitifs, seuls acceptés par ChromaDB."""
        return {
            "document_name": self.document_name,
            "document_type": self.document_type.value,
            "year": self.year,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section or "",
            "source_url": self.source_url or "",
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_chroma_metadata(cls, meta: dict) -> "ChunkMetadata":
        return cls(
            document_name=meta["document_name"],
            document_type=DocumentType(meta["document_type"]),
            year=int(meta["year"]),
            page_start=int(meta["page_start"]),
            page_end=int(meta["page_end"]),
            section=meta.get("section") or None,
            source_url=meta.get("source_url") or None,
            source_file=meta["source_file"],
            chunk_index=int(meta["chunk_index"]),
            chunk_id=meta["chunk_id"],
        )


class Chunk(BaseModel):
    """Chunk de texte issu de l'ingestion, avant calcul de son embedding."""

    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """Chunk remonté par la recherche, avec ses scores de pertinence."""

    text: str
    metadata: ChunkMetadata
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    fused_score: float = 0.0


class Source(BaseModel):
    """Citation affichée à l'utilisateur à côté de la réponse."""

    document: str
    document_type: DocumentType
    year: int
    page: str
    section: Optional[str] = None
    source_url: Optional[str] = None
    excerpt: str
    relevance_score: float

    @classmethod
    def from_retrieved_chunk(cls, chunk: RetrievedChunk) -> "Source":
        from app.utils.text import truncate

        return cls(
            document=chunk.metadata.document_name,
            document_type=chunk.metadata.document_type,
            year=chunk.metadata.year,
            page=chunk.metadata.page_label(),
            section=chunk.metadata.section,
            source_url=chunk.metadata.source_url,
            excerpt=truncate(chunk.text, 320),
            relevance_score=round(chunk.fused_score, 4),
        )


class QueryAnalysis(BaseModel):
    """Résultat de l'analyse de la question."""

    raw_question: str
    years: list[int] = Field(default_factory=list)
    document_type: Optional[DocumentType] = None
    topics: list[str] = Field(default_factory=list)
    intent: QueryIntent = QueryIntent.FACTUAL
    is_comparison: bool = False


class QueryRequest(BaseModel):
    """Charge utile acceptée par l'API et par le pipeline."""

    question: str = Field(..., min_length=3)
    years: Optional[list[int]] = None
    document_type: Optional[DocumentType] = None


class QueryResponse(BaseModel):
    """Réponse finale renvoyée à l'appelant, API ou interface."""

    answer: str
    sources: list[Source]
    abstained: bool
    query_analysis: QueryAnalysis
    retrieved_count: int
    timings_ms: dict[str, float]
    context_texts: list[str] = Field(
        default_factory=list,
        description=(
            "Texte intégral des chunks envoyés au modèle, non tronqué "
            "contrairement à Source.excerpt. Sert à l'évaluation de fidélité "
            "et au débogage."
        ),
    )


class DocumentSummary(BaseModel):
    """Ligne de l'inventaire renvoyé par /api/documents."""

    document_name: str
    document_type: DocumentType
    year: int
    source_file: str
    chunk_count: int
