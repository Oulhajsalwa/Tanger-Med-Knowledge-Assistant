
from __future__ import annotations

import math
import time
from collections import defaultdict
from functools import lru_cache
from typing import Optional

from app.config import Settings, get_settings
from app.models import (
    DocumentSummary,
    DocumentType,
    QueryAnalysis,
    QueryIntent,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    Source,
)
from app.rag.generator import AnswerGenerator
from app.rag.query_analyzer import analyze_query
from app.retrieval.bm25 import load_or_build
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStore
from app.utils.logging import get_logger
from app.utils.openrouter_client import OpenRouterClient, OpenRouterError

logger = get_logger(__name__)

MULTI_YEAR_INTENTS = {QueryIntent.COMPARISON, QueryIntent.TEMPORAL_EVOLUTION}

ABSTENTION_ANSWER = (
    "L'information demandée n'est pas disponible dans le corpus documentaire "
    "fourni. Le corpus actuellement indexé ne contient peut-être pas cette "
    "donnée, cette année, ou ce sujet. Essayez de reformuler la question, de "
    "changer le filtre d'année, ou vérifiez que le rapport concerné a bien "
    "été ajouté dans `data/annual_reports` ou `data/rse_reports` puis ingéré."
)


class RAGPipeline:
    """Service enveloppant le vector store, l'index BM25, le retriever et le générateur."""

    def __init__(
        self,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.settings = settings

    def answer(self, request: QueryRequest) -> QueryResponse:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        analysis = analyze_query(request.question)
        timings["query_analysis_ms"] = elapsed_ms(t0)

        effective_years = request.years if request.years else (analysis.years or None)
        effective_doc_type = request.document_type or analysis.document_type

        t0 = time.perf_counter()
        chunks = self._retrieve(request.question, analysis, effective_years, effective_doc_type)
        timings["retrieval_ms"] = elapsed_ms(t0)

        should_abstain, reason, max_score = self._should_abstain(chunks)
        if should_abstain:
            logger.info(
                "Abstention sur %r (récupérés=%d, score max=%.3f, motif=%s).",
                request.question,
                len(chunks),
                max_score,
                reason,
            )
            timings["generation_ms"] = 0.0
            timings["total_ms"] = elapsed_ms(total_start)
            return QueryResponse(
                answer=ABSTENTION_ANSWER,
                sources=[],
                abstained=True,
                query_analysis=analysis,
                retrieved_count=len(chunks),
                timings_ms=timings,
            )

        t0 = time.perf_counter()
        try:
            answer_text = self.generator.generate(request.question, analysis, chunks)
        except OpenRouterError as exc:
            logger.error("Génération en échec : %s", exc)
            timings["generation_ms"] = elapsed_ms(t0)
            timings["total_ms"] = elapsed_ms(total_start)
            return QueryResponse(
                answer=(
                    "Une erreur est survenue lors de la génération de la réponse "
                    f"(fournisseur LLM indisponible) : {exc}"
                ),
                sources=[Source.from_retrieved_chunk(c) for c in chunks],
                abstained=False,
                query_analysis=analysis,
                retrieved_count=len(chunks),
                timings_ms=timings,
            )
        timings["generation_ms"] = elapsed_ms(t0)
        timings["total_ms"] = elapsed_ms(total_start)

        return QueryResponse(
            answer=answer_text,
            sources=[Source.from_retrieved_chunk(c) for c in chunks],
            abstained=False,
            query_analysis=analysis,
            retrieved_count=len(chunks),
            timings_ms=timings,
            context_texts=[c.text for c in chunks],
        )

    def _should_abstain(self, chunks: list[RetrievedChunk]) -> tuple[bool, str, float]:

        if not chunks:
            return True, "aucun_chunk", 0.0

        max_fused = max(c.fused_score for c in chunks)
        max_semantic = max(c.semantic_score for c in chunks)

        if max_fused < self.settings.relevance_threshold:
            return True, f"fusionné<{self.settings.relevance_threshold}", max_fused

        if max_semantic > 0.0 and max_semantic < self.settings.min_semantic_score:
            return True, f"sémantique<{self.settings.min_semantic_score}", max_fused

        return False, "", max_fused

    def _retrieve(
        self,
        question: str,
        analysis: QueryAnalysis,
        years: Optional[list[int]],
        document_type: Optional[DocumentType],
    ) -> list[RetrievedChunk]:
        multi_year = analysis.intent in MULTI_YEAR_INTENTS and years and len(years) >= 2

        if not multi_year:
            return self.retriever.search(
                question,
                years=years,
                document_type=document_type,
                top_k=self.settings.top_k,
                final_k=self.settings.final_context_k,
            )


        per_year_k = max(2, math.ceil(self.settings.final_context_k / len(years)))
        results: list[RetrievedChunk] = []
        for year in years:
            year_chunks = self.retriever.search(
                question,
                years=[year],
                document_type=document_type,
                top_k=self.settings.top_k,
                final_k=per_year_k,
            )
            if not year_chunks:
                logger.info("Aucun passage pertinent pour %d (« %s »).", year, question)
            results.extend(year_chunks)

        results.sort(key=lambda c: (c.metadata.year, -c.fused_score))
        return results[: max(self.settings.final_context_k, per_year_k * len(years))]

    def list_documents(self) -> list[DocumentSummary]:
        counts: dict[tuple, int] = defaultdict(int)
        for m in self.vector_store.all_metadata():
            counts[(m.document_name, m.document_type, m.year, m.source_file)] += 1

        summaries = [
            DocumentSummary(
                document_name=name,
                document_type=doc_type,
                year=year,
                source_file=source_file,
                chunk_count=count,
            )
            for (name, doc_type, year, source_file), count in counts.items()
        ]
        return sorted(summaries, key=lambda s: (s.year, s.document_type.value))

    def health(self) -> dict:
        return {
            "status": "ok",
            "chunks_indexed": self.vector_store.count(),
            "openrouter_key_configured": self.settings.has_api_key(),
            "chat_model": self.settings.openrouter_model,
            "embedding_model": self.settings.openrouter_embedding_model,
        }


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@lru_cache(maxsize=1)
def build_pipeline(_cache_token: Optional[str] = None) -> RAGPipeline:
  
    settings = get_settings()
    vector_store = VectorStore(settings)
    bm25_index = load_or_build(settings)
    llm_client = OpenRouterClient(settings)
    retriever = HybridRetriever(vector_store, bm25_index, llm_client, settings=settings)
    return RAGPipeline(vector_store, retriever, AnswerGenerator(llm_client), settings)
