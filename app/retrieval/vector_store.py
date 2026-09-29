"""Stockage vectoriel persistant (ChromaDB).
"""

from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings, get_settings
from app.models import Chunk, ChunkMetadata, DocumentType, RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_where_filter(
    years: Optional[list[int]] = None,
    document_type: Optional[DocumentType] = None,
) -> Optional[dict]:
    clauses: list[dict] = []
    if years:
        clauses.append({"year": years[0]} if len(years) == 1 else {"year": {"$in": years}})
    if document_type is not None:
        clauses.append({"document_type": document_type.value})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = chromadb.PersistentClient(
            path=str(self.settings.vectorstore_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks et embeddings doivent avoir la même longueur")

        self._collection.upsert(
            ids=[c.metadata.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata.to_chroma_metadata() for c in chunks],
        )
        logger.info("%d chunks insérés dans le vector store.", len(chunks))

    def delete_by_source_file(self, source_file: str) -> None:
        """Supprime les chunks d'un PDF donné (utilisé lors d'une ré-ingestion)."""
        self._collection.delete(where={"source_file": source_file})

    def all_metadata(self) -> list[ChunkMetadata]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        return [ChunkMetadata.from_chroma_metadata(m) for m in result["metadatas"]]

    def all_chunks_raw(self) -> tuple[list[str], list[ChunkMetadata], list[str]]:
        """(textes, métadonnées, ids) de tous les chunks, pour reconstruire BM25."""
        if self.count() == 0:
            return [], [], []
        result = self._collection.get(include=["documents", "metadatas"])
        metadatas = [ChunkMetadata.from_chroma_metadata(m) for m in result["metadatas"]]
        return result["documents"], metadatas, result["ids"]

    def reset(self) -> None:
        self._client.delete_collection(self.settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection du vector store réinitialisée.")

    def semantic_search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        years: Optional[list[int]] = None,
        document_type: Optional[DocumentType] = None,
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=build_where_filter(years, document_type),
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        retrieved = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
           
            similarity = max(0.0, min(1.0, 1.0 - distance))
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    metadata=ChunkMetadata.from_chroma_metadata(meta),
                    semantic_score=similarity,
                )
            )
        return retrieved
