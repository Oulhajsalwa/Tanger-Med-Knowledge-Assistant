"""Tests du RAG : analyse de la question, prompts, abstention, multi-années."""

from __future__ import annotations

import pytest

from app.models import (
    DocumentType,
    QueryIntent,
    QueryRequest,
    RetrievedChunk,
    Source,
)
from app.rag.generator import AnswerGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import SYSTEM_PROMPT, build_messages, format_context
from app.rag.query_analyzer import analyze_query
from tests.factories import make_metadata


# analyse de la question : les exemples du cahier des charges
class TestQueryAnalyzer:
    def test_single_year_factual(self):
        analysis = analyze_query("Quel était le trafic en 2024 ?")
        assert analysis.years == [2024]
        assert analysis.intent is QueryIntent.FACTUAL
        assert analysis.is_comparison is False

    def test_comparison_keeps_only_cited_years(self):
        """« Comparez … entre 2022 et 2024 » désigne 2022 et 2024, pas 2023."""
        analysis = analyze_query("Comparez le trafic entre 2022 et 2024.")
        assert analysis.years == [2022, 2024]
        assert analysis.intent is QueryIntent.COMPARISON
        assert analysis.is_comparison is True

    def test_topic_only_question(self):
        analysis = analyze_query("Quelles sont les principales actions environnementales de Tanger Med ?")
        assert analysis.years == []
        assert "environnement" in analysis.topics
        assert analysis.intent is QueryIntent.GENERAL_SYNTHESIS

    def test_evolution_expands_the_full_year_range(self):
        analysis = analyze_query("Comment les performances ont-elles évolué entre 2018 et 2024 ?")
        assert analysis.years == [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        assert analysis.intent is QueryIntent.TEMPORAL_EVOLUTION

    def test_out_of_corpus_year_is_still_extracted(self):
        """L'extraction ne doit écarter aucune année : l'abstention vient après."""
        analysis = analyze_query("Quel était le chiffre d'affaires de Tanger Med en 2030 ?")
        assert analysis.years == [2030]

    def test_detects_rse_document_type(self):
        analysis = analyze_query("Que dit le rapport RSE sur la biodiversité ?")
        assert analysis.document_type is DocumentType.RSE_REPORT

    def test_detects_annual_document_type(self):
        analysis = analyze_query("Quel chiffre d'affaires figure dans le rapport annuel ?")
        assert analysis.document_type is DocumentType.ANNUAL_REPORT

    def test_ambiguous_type_is_left_unfiltered(self):
        analysis = analyze_query("Combien de collaborateurs compte le Groupe ?")
        assert analysis.document_type is None

    def test_multiple_topics_detected(self):
        analysis = analyze_query("Quelle est la gouvernance du Groupe et sa politique environnementale ?")
        assert "gouvernance" in analysis.topics
        assert "environnement" in analysis.topics

    def test_is_deterministic(self):
        q = "Comparez le trafic entre 2022 et 2024."
        assert analyze_query(q) == analyze_query(q)


# prompts
class TestPrompts:
    def test_system_prompt_states_the_anti_hallucination_rules(self):
        assert "EXCLUSIVEMENT" in SYSTEM_PROMPT
        assert "n'est pas disponible dans le corpus" in SYSTEM_PROMPT
        assert "Cite systématiquement tes sources" in SYSTEM_PROMPT

    def test_context_blocks_carry_citable_metadata(self):
        chunk = RetrievedChunk(
            text="Le tonnage global atteint 209 millions de tonnes.",
            metadata=make_metadata(year=2025, page_start=9, section="NOS CHIFFRES CLEFS"),
        )
        context = format_context([chunk])
        assert "Rapport Annuel 2025" in context
        assert "Année : 2025" in context
        assert "p. 9" in context
        assert "NOS CHIFFRES CLEFS" in context
        assert "209 millions de tonnes" in context

    def test_empty_context_is_explicit(self):
        assert "Aucun extrait pertinent" in format_context([])

    def test_user_message_includes_years_and_comparison_hint(self):
        analysis = analyze_query("Comparez le trafic entre 2023 et 2024.")
        chunk = RetrievedChunk(text="texte", metadata=make_metadata())
        messages = build_messages(analysis.raw_question, analysis, [chunk])
        user_content = messages[1]["content"]
        assert "2023, 2024" in user_content
        assert "comparaison" in user_content.lower()

    def test_messages_structure(self):
        analysis = analyze_query("Question simple ?")
        messages = build_messages("Question simple ?", analysis, [])
        assert [m["role"] for m in messages] == ["system", "user"]


# generation
class TestGenerator:
    def test_generate_passes_prompt_and_returns_answer(self, fake_llm_client):
        generator = AnswerGenerator(fake_llm_client)
        analysis = analyze_query("Quel était le trafic en 2024 ?")
        chunk = RetrievedChunk(text="Trafic de 187 MT en 2024.", metadata=make_metadata(year=2024))

        answer = generator.generate(analysis.raw_question, analysis, [chunk])

        assert answer == fake_llm_client.answer
        assert len(fake_llm_client.chat_calls) == 1
        sent_user_msg = fake_llm_client.chat_calls[0][1]["content"]
        assert "187 MT" in sent_user_msg


# pipeline : abstention, filtres, equilibrage par annee, citations
class StubRetriever:
    """Enregistre chaque appel de recherche et renvoie des résultats fixés."""

    def __init__(self, results_by_year=None, default_results=None):
        self.results_by_year = results_by_year or {}
        self.default_results = default_results or []
        self.calls: list[dict] = []

    def search(self, query, *, years=None, document_type=None, top_k=None, final_k=None):
        self.calls.append(
            {"query": query, "years": years, "document_type": document_type,
             "top_k": top_k, "final_k": final_k}
        )
        if years and len(years) == 1 and years[0] in self.results_by_year:
            return self.results_by_year[years[0]][: (final_k or 99)]
        return self.default_results[: (final_k or 99)]


def _chunk(year: int, score: float, text: str = "extrait", page: int = 10) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=make_metadata(year=year, page_start=page, chunk_id=f"c-{year}-{page}-{score}"),
        fused_score=score,
    )


def _pipeline(settings, fake_llm_client, retriever) -> RAGPipeline:
    from app.retrieval.vector_store import VectorStore

    return RAGPipeline(
        vector_store=VectorStore(settings),
        retriever=retriever,
        generator=AnswerGenerator(fake_llm_client),
        settings=settings,
    )


class TestAbstention:
    def test_abstains_when_nothing_is_retrieved(self, settings, fake_llm_client):
        retriever = StubRetriever(default_results=[])
        pipeline = _pipeline(settings, fake_llm_client, retriever)

        response = pipeline.answer(QueryRequest(question="Quel trafic en 2030 ?"))

        assert response.abstained is True
        assert "n'est pas disponible dans le corpus" in response.answer
        assert response.sources == []
        assert fake_llm_client.chat_calls == [], "the LLM must not be called when abstaining"

    def test_abstains_when_best_score_is_below_threshold(self, settings, fake_llm_client):
        weak = _chunk(2024, score=settings.relevance_threshold - 0.01)
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[weak]))

        response = pipeline.answer(QueryRequest(question="Question hors sujet ?"))

        assert response.abstained is True
        assert fake_llm_client.chat_calls == []

    def test_answers_when_score_reaches_threshold(self, settings, fake_llm_client):
        strong = _chunk(2024, score=settings.relevance_threshold + 0.4)
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[strong]))

        response = pipeline.answer(QueryRequest(question="Quel était le trafic en 2024 ?"))

        assert response.abstained is False
        assert response.answer == fake_llm_client.answer
        assert len(fake_llm_client.chat_calls) == 1

    def test_threshold_is_configurable(self, settings, fake_llm_client):
        settings.relevance_threshold = 0.9
        borderline = _chunk(2024, score=0.5)
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[borderline]))
        assert pipeline.answer(QueryRequest(question="Trafic 2024 ?")).abstained is True


class TestSemanticAbstentionGuard:
    """Abstention thématique : un bon score lexical ne doit pas sauver une question hors sujet.

    Sur ce corpus, « le cours de l'action Apple » décroche un BM25 honorable
    parce que « action » et « cours » sont partout dans les rapports. Seul le
    score sémantique révèle que la question est hors sujet.
    """

    def _off_topic_chunk(self, *, semantic: float, lexical: float, settings) -> RetrievedChunk:
        chunk = RetrievedChunk(
            text="Le Groupe a mené plusieurs actions au cours de l'exercice.",
            metadata=make_metadata(year=2024, chunk_id="offtopic"),
            semantic_score=semantic,
            lexical_score=lexical,
        )
        chunk.fused_score = (
            settings.retrieval_alpha * semantic + (1 - settings.retrieval_alpha) * lexical
        )
        return chunk

    def test_abstains_when_semantically_off_topic_despite_lexical_match(
        self, settings, fake_llm_client
    ):
        settings.min_semantic_score = 0.25
        settings.relevance_threshold = 0.10
        chunk = self._off_topic_chunk(semantic=0.12, lexical=0.95, settings=settings)
        # Le lexical seul franchit le seuil fusionne...
        assert chunk.fused_score > settings.relevance_threshold
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[chunk]))

        response = pipeline.answer(QueryRequest(question="Quel est le cours de l'action Apple ?"))

        # ...mais le garde semantique doit quand meme declencher l'abstention.
        assert response.abstained is True
        assert fake_llm_client.chat_calls == []

    def test_answers_when_semantically_on_topic(self, settings, fake_llm_client):
        settings.min_semantic_score = 0.25
        chunk = self._off_topic_chunk(semantic=0.62, lexical=0.5, settings=settings)
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[chunk]))

        assert pipeline.answer(QueryRequest(question="Quel trafic en 2024 ?")).abstained is False

    def test_guard_is_skipped_when_no_semantic_score_available(self, settings, fake_llm_client):
        """En mode lexical seul, le système ne doit pas s'abstenir systématiquement."""
        chunk = RetrievedChunk(
            text="extrait",
            metadata=make_metadata(year=2024, chunk_id="lexonly"),
            semantic_score=0.0,
            lexical_score=0.9,
        )
        chunk.fused_score = 0.9
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[chunk]))

        assert pipeline.answer(QueryRequest(question="Quel trafic en 2024 ?")).abstained is False

    def test_semantic_guard_is_configurable(self, settings, fake_llm_client):
        chunk = self._off_topic_chunk(semantic=0.40, lexical=0.5, settings=settings)
        settings.min_semantic_score = 0.30
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[chunk]))
        assert pipeline.answer(QueryRequest(question="Trafic 2024 ?")).abstained is False

        settings.min_semantic_score = 0.60
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[chunk]))
        assert pipeline.answer(QueryRequest(question="Trafic 2024 ?")).abstained is True


class TestMultiYearRetrieval:
    def test_comparison_triggers_one_search_per_year(self, settings, fake_llm_client):
        retriever = StubRetriever(
            results_by_year={
                2023: [_chunk(2023, 0.8, "Trafic 2023", page=37)],
                2024: [_chunk(2024, 0.8, "Trafic 2024", page=41)],
            }
        )
        pipeline = _pipeline(settings, fake_llm_client, retriever)

        response = pipeline.answer(QueryRequest(question="Comparez le trafic entre 2023 et 2024."))

        searched_years = [call["years"] for call in retriever.calls]
        assert searched_years == [[2023], [2024]], "each year must get its own balanced search"
        assert response.abstained is False
        assert {s.year for s in response.sources} == {2023, 2024}

    def test_single_year_uses_one_global_search(self, settings, fake_llm_client):
        retriever = StubRetriever(default_results=[_chunk(2024, 0.8)])
        pipeline = _pipeline(settings, fake_llm_client, retriever)

        pipeline.answer(QueryRequest(question="Quel était le trafic en 2024 ?"))

        assert len(retriever.calls) == 1
        assert retriever.calls[0]["years"] == [2024]

    def test_missing_year_does_not_break_comparison(self, settings, fake_llm_client):
        """Une année présente, une absente : on répond pour celle qui existe."""
        retriever = StubRetriever(
            results_by_year={2024: [_chunk(2024, 0.8)], 2030: []}
        )
        pipeline = _pipeline(settings, fake_llm_client, retriever)

        response = pipeline.answer(QueryRequest(question="Comparez le trafic entre 2024 et 2030."))

        assert response.abstained is False
        assert {s.year for s in response.sources} == {2024}

    def test_explicit_request_years_override_the_question(self, settings, fake_llm_client):
        retriever = StubRetriever(default_results=[_chunk(2023, 0.8)])
        pipeline = _pipeline(settings, fake_llm_client, retriever)

        pipeline.answer(QueryRequest(question="Quel était le trafic en 2024 ?", years=[2023]))

        assert retriever.calls[0]["years"] == [2023], "UI/API filter must win over the parsed year"

    def test_explicit_document_type_is_forwarded(self, settings, fake_llm_client):
        retriever = StubRetriever(default_results=[_chunk(2024, 0.8)])
        pipeline = _pipeline(settings, fake_llm_client, retriever)

        pipeline.answer(
            QueryRequest(question="Trafic ?", document_type=DocumentType.RSE_REPORT)
        )
        assert retriever.calls[0]["document_type"] is DocumentType.RSE_REPORT


class TestCitations:
    def test_sources_expose_document_year_and_page(self, settings, fake_llm_client):
        chunk = _chunk(2024, 0.8, "Le tonnage global atteint 187 millions de tonnes.", page=41)
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[chunk]))

        response = pipeline.answer(QueryRequest(question="Quel tonnage en 2024 ?"))

        assert len(response.sources) == 1
        source = response.sources[0]
        assert source.document == "Rapport Annuel 2024"
        assert source.year == 2024
        assert source.page == "p. 41"
        assert source.source_url == "https://www.tangermed.ma/fr/documentation/"
        assert "187 millions" in source.excerpt

    def test_source_from_chunk_preserves_page_range(self):
        chunk = RetrievedChunk(
            text="texte", metadata=make_metadata(page_start=90, page_end=91), fused_score=0.5
        )
        assert Source.from_retrieved_chunk(chunk).page == "p. 90-91"

    def test_excerpt_is_truncated(self):
        chunk = RetrievedChunk(text="A" * 1000, metadata=make_metadata(), fused_score=0.5)
        assert len(Source.from_retrieved_chunk(chunk).excerpt) <= 321


class TestObservability:
    def test_timings_are_reported(self, settings, fake_llm_client):
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[_chunk(2024, 0.8)]))
        response = pipeline.answer(QueryRequest(question="Trafic 2024 ?"))
        for key in ("query_analysis_ms", "retrieval_ms", "generation_ms", "total_ms"):
            assert key in response.timings_ms
        assert response.retrieved_count == 1

    def test_llm_failure_is_surfaced_without_crashing(self, settings, fake_llm_client):
        from app.utils.openrouter_client import OpenRouterError

        def boom(*args, **kwargs):
            raise OpenRouterError("quota exceeded")

        fake_llm_client.chat_completion = boom
        pipeline = _pipeline(settings, fake_llm_client, StubRetriever(default_results=[_chunk(2024, 0.8)]))

        response = pipeline.answer(QueryRequest(question="Trafic 2024 ?"))

        assert "erreur" in response.answer.lower()
        assert response.sources, "sources are still returned so the user can check them manually"


class TestDocumentInventory:
    def test_groups_chunks_by_document(self, settings, sample_chunks, fake_llm_client):
        from app.retrieval.vector_store import VectorStore

        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))
        pipeline = RAGPipeline(store, StubRetriever(), AnswerGenerator(fake_llm_client), settings)

        documents = pipeline.list_documents()

        assert {d.year for d in documents} == {2024, 2025}
        assert sum(d.chunk_count for d in documents) == len(sample_chunks)
        assert documents == sorted(documents, key=lambda d: (d.year, d.document_type.value))

    def test_health_reports_index_state(self, settings, fake_llm_client):
        from app.retrieval.vector_store import VectorStore

        pipeline = RAGPipeline(
            VectorStore(settings), StubRetriever(), AnswerGenerator(fake_llm_client), settings
        )
        health = pipeline.health()
        assert health["status"] == "ok"
        assert health["chunks_indexed"] == 0
        assert health["openrouter_key_configured"] is True

    def test_question_too_short_is_rejected_by_validation(self):
        with pytest.raises(Exception):
            QueryRequest(question="ab")
