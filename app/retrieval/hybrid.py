
from __future__ import annotations

from typing import Optional

from app.config import Settings, get_settings
from app.models import DocumentType, RetrievedChunk
from app.retrieval.bm25 import Bm25Index
from app.retrieval.vector_store import VectorStore
from app.utils.logging import get_logger
from app.utils.openrouter_client import OpenRouterClient

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: Bm25Index,
        llm_client: OpenRouterClient,
        *,
        settings: Optional[Settings] = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.llm_client = llm_client
        self.settings = settings or get_settings()

    def search(
        self,
        query: str,
        *,
        years: Optional[list[int]] = None,
        document_type: Optional[DocumentType] = None,
        top_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
       
        top_k = top_k or self.settings.top_k
        final_k = final_k or self.settings.final_context_k

        query_embedding = self.llm_client.embed_query(query)

        semantic_results = self.vector_store.semantic_search(
            query_embedding, top_k, years=years, document_type=document_type
        )
        lexical_results = self.bm25_index.search(
            query, top_k, years=years, document_type=document_type
        )

        return self._fuse(semantic_results, lexical_results)[:final_k]

    def _fuse(
        self,
        semantic_results: list[RetrievedChunk],
        lexical_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        alpha = self.settings.retrieval_alpha
        merged: dict[str, RetrievedChunk] = {}

        for chunk in semantic_results:
            merged[chunk.metadata.chunk_id] = chunk

        for chunk in lexical_results:
            key = chunk.metadata.chunk_id
            if key in merged:
                merged[key].lexical_score = chunk.lexical_score
            else:
                merged[key] = chunk

        for chunk in merged.values():
            chunk.fused_score = alpha * chunk.semantic_score + (1 - alpha) * chunk.lexical_score

        ranked = sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
        logger.debug(
            "Fusion : %d sémantiques + %d lexicaux -> %d candidats uniques.",
            len(semantic_results),
            len(lexical_results),
            len(ranked),
        )
        return ranked
