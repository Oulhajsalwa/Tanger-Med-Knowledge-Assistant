"""Tests de la recherche : vector store, BM25, fusion hybride."""

from __future__ import annotations

import pytest

from app.models import DocumentType, RetrievedChunk
from app.retrieval.bm25 import Bm25Index, tokenize
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStore, build_where_filter
from tests.factories import make_metadata


# traduction des filtres de metadonnees
class TestWhereFilter:
    def test_no_filter(self):
        assert build_where_filter(None, None) is None

    def test_single_year_uses_equality(self):
        assert build_where_filter([2024], None) == {"year": 2024}

    def test_multiple_years_use_in_operator(self):
        assert build_where_filter([2022, 2024], None) == {"year": {"$in": [2022, 2024]}}

    def test_document_type_only(self):
        assert build_where_filter(None, DocumentType.RSE_REPORT) == {"document_type": "rse_report"}

    def test_year_and_type_are_combined_with_and(self):
        where = build_where_filter([2024], DocumentType.ANNUAL_REPORT)
        assert where == {"$and": [{"year": 2024}, {"document_type": "annual_report"}]}


# BM25
class TestTokenizer:
    def test_strips_accents_and_lowercases(self):
        assert "decarbonation" in tokenize("Décarbonation")

    def test_removes_french_stopwords(self):
        tokens = tokenize("Quel est le trafic de la plateforme ?")
        assert "trafic" in tokens
        assert "le" not in tokens and "la" not in tokens

    def test_keeps_figures_and_acronyms(self):
        tokens = tokenize("209 millions de tonnes et 12,42 MEVP en 2025")
        assert "209" in tokens and "2025" in tokens and "mevp" in tokens


class TestBm25Index:
    @pytest.fixture
    def index(self, sample_chunks) -> Bm25Index:
        return Bm25Index.build(
            [c.text for c in sample_chunks],
            [c.metadata for c in sample_chunks],
            [c.metadata.chunk_id for c in sample_chunks],
        )

    def test_finds_exact_figures(self, index):
        results = index.search("209 millions de tonnes", top_k=3)
        assert results
        assert "209 millions" in results[0].text

    def test_scores_are_bounded_in_zero_one(self, index):
        results = index.search("tonnage global", top_k=5)
        assert results
        assert all(0.0 < r.lexical_score < 1.0 for r in results)

    def test_normalization_is_absolute_not_relative(self, index):
        """Le meilleur résultat ne doit pas être figé à 1.0.

        Une normalisation relative au lot de résultats donne toujours un score
        parfait au premier, même sur une question étrangère au corpus, et prive
        le seuil d'abstention de tout signal absolu.
        """
        strong = index.search("tonnage global traite millions de tonnes 2025", top_k=3)
        weak = index.search("tonnage", top_k=3)
        assert strong and weak
        assert strong[0].lexical_score != pytest.approx(1.0)
        # Plus de termes de la question retrouvés => score absolu plus élevé.
        assert strong[0].lexical_score > weak[0].lexical_score

    def test_saturation_constant_is_configurable(self, sample_chunks):
        args = (
            [c.text for c in sample_chunks],
            [c.metadata for c in sample_chunks],
            [c.metadata.chunk_id for c in sample_chunks],
        )
        low_k = Bm25Index.build(*args, saturation_k=1.0)
        high_k = Bm25Index.build(*args, saturation_k=100.0)
        query = "tonnage global tonnes"
        # Un k plus petit sature plus vite => score normalise plus eleve.
        assert low_k.search(query, top_k=1)[0].lexical_score > high_k.search(query, top_k=1)[0].lexical_score

    def test_scores_are_monotonically_decreasing(self, index):
        results = index.search("tonnage millions tonnes 2024", top_k=5)
        scores = [r.lexical_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_year_filter_is_enforced(self, index):
        results = index.search("tonnage", top_k=5, years=[2024])
        assert results
        assert all(r.metadata.year == 2024 for r in results)

    def test_document_type_filter_is_enforced(self, index):
        results = index.search("dechets femmes carbone", top_k=5, document_type=DocumentType.RSE_REPORT)
        assert results
        assert all(r.metadata.document_type is DocumentType.RSE_REPORT for r in results)

    def test_combined_filters(self, index):
        results = index.search(
            "neutralite carbone", top_k=5, years=[2024], document_type=DocumentType.RSE_REPORT
        )
        assert all(
            r.metadata.year == 2024 and r.metadata.document_type is DocumentType.RSE_REPORT
            for r in results
        )

    def test_returns_empty_for_irrelevant_query(self, index):
        assert index.search("zzzzz qqqqq inexistant", top_k=5) == []

    def test_returns_empty_when_filter_excludes_everything(self, index):
        assert index.search("tonnage", top_k=5, years=[1999]) == []

    def test_persistence_round_trip(self, index, tmp_path):
        path = tmp_path / "bm25.pkl"
        index.save(path)
        loaded = Bm25Index.load(path)
        assert loaded is not None
        assert loaded.texts == index.texts
        assert [m.chunk_id for m in loaded.metadatas] == [m.chunk_id for m in index.metadatas]

    def test_load_missing_file_returns_none(self, tmp_path):
        assert Bm25Index.load(tmp_path / "does_not_exist.pkl") is None


# vector store : vrai ChromaDB, repertoire temporaire, embeddings simules
class TestVectorStore:
    def test_upsert_and_count(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        embeddings = fake_llm_client.embed_texts([c.text for c in sample_chunks])
        store.upsert_chunks(sample_chunks, embeddings)
        assert store.count() == len(sample_chunks)

    def test_upsert_is_idempotent_on_same_ids(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        embeddings = fake_llm_client.embed_texts([c.text for c in sample_chunks])
        store.upsert_chunks(sample_chunks, embeddings)
        store.upsert_chunks(sample_chunks, embeddings)
        assert store.count() == len(sample_chunks)

    def test_rejects_mismatched_lengths(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        with pytest.raises(ValueError):
            store.upsert_chunks(sample_chunks, [fake_llm_client.embed_query("x")])

    def test_semantic_search_returns_scored_chunks(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))

        query_vec = fake_llm_client.embed_query(sample_chunks[0].text)
        results = store.semantic_search(query_vec, top_k=3)
        assert results
        assert all(0.0 <= r.semantic_score <= 1.0 for r in results)
        # Le texte identique doit arriver premier, avec un score quasi parfait.
        assert results[0].text == sample_chunks[0].text
        assert results[0].semantic_score > 0.99

    def test_semantic_search_respects_year_filter(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))
        results = store.semantic_search(fake_llm_client.embed_query("tonnage"), top_k=5, years=[2024])
        assert results
        assert all(r.metadata.year == 2024 for r in results)

    def test_search_on_empty_store_returns_empty(self, settings, fake_llm_client):
        store = VectorStore(settings)
        assert store.semantic_search(fake_llm_client.embed_query("x"), top_k=3) == []

    def test_delete_by_source_file(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))
        target = sample_chunks[0].metadata.source_file
        expected_remaining = sum(1 for c in sample_chunks if c.metadata.source_file != target)
        store.delete_by_source_file(target)
        assert store.count() == expected_remaining

    def test_all_metadata_returns_every_chunk(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))
        assert len(store.all_metadata()) == len(sample_chunks)

    def test_reset_empties_the_collection(self, settings, sample_chunks, fake_llm_client):
        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))
        store.reset()
        assert store.count() == 0


# fusion hybride
class TestHybridFusion:
    def _retriever(self, settings, fake_llm_client, sample_chunks):
        store = VectorStore(settings)
        store.upsert_chunks(sample_chunks, fake_llm_client.embed_texts([c.text for c in sample_chunks]))
        bm25 = Bm25Index.build(
            [c.text for c in sample_chunks],
            [c.metadata for c in sample_chunks],
            [c.metadata.chunk_id for c in sample_chunks],
        )
        return HybridRetriever(store, bm25, fake_llm_client, settings=settings)

    def test_fusion_applies_configured_alpha(self, settings, fake_llm_client, sample_chunks):
        retriever = self._retriever(settings, fake_llm_client, sample_chunks)
        semantic = [RetrievedChunk(text="a", metadata=make_metadata(chunk_id="x"), semantic_score=1.0)]
        lexical = [RetrievedChunk(text="a", metadata=make_metadata(chunk_id="x"), lexical_score=0.5)]

        fused = retriever._fuse(semantic, lexical)
        assert len(fused) == 1
        expected = settings.retrieval_alpha * 1.0 + (1 - settings.retrieval_alpha) * 0.5
        assert fused[0].fused_score == pytest.approx(expected)

    def test_fusion_deduplicates_by_chunk_id(self, settings, fake_llm_client, sample_chunks):
        retriever = self._retriever(settings, fake_llm_client, sample_chunks)
        semantic = [RetrievedChunk(text="a", metadata=make_metadata(chunk_id="dup"), semantic_score=0.8)]
        lexical = [RetrievedChunk(text="a", metadata=make_metadata(chunk_id="dup"), lexical_score=0.9)]
        fused = retriever._fuse(semantic, lexical)
        assert len(fused) == 1
        assert fused[0].semantic_score == 0.8 and fused[0].lexical_score == 0.9

    def test_fusion_keeps_single_mode_hits(self, settings, fake_llm_client, sample_chunks):
        """Un chunk trouvé par BM25 seul, un chiffre exact par exemple, ne doit pas disparaître."""
        retriever = self._retriever(settings, fake_llm_client, sample_chunks)
        semantic = [RetrievedChunk(text="a", metadata=make_metadata(chunk_id="s"), semantic_score=0.6)]
        lexical = [RetrievedChunk(text="b", metadata=make_metadata(chunk_id="l"), lexical_score=1.0)]
        fused = retriever._fuse(semantic, lexical)
        assert {c.metadata.chunk_id for c in fused} == {"s", "l"}

    def test_fusion_is_sorted_desc(self, settings, fake_llm_client, sample_chunks):
        retriever = self._retriever(settings, fake_llm_client, sample_chunks)
        semantic = [
            RetrievedChunk(text="a", metadata=make_metadata(chunk_id="a"), semantic_score=0.2),
            RetrievedChunk(text="b", metadata=make_metadata(chunk_id="b"), semantic_score=0.9),
        ]
        fused = retriever._fuse(semantic, [])
        assert [c.metadata.chunk_id for c in fused] == ["b", "a"]

    def test_search_truncates_to_final_k(self, settings, fake_llm_client, sample_chunks):
        retriever = self._retriever(settings, fake_llm_client, sample_chunks)
        results = retriever.search("tonnage global tonnes", final_k=2)
        assert len(results) <= 2

    def test_search_applies_filters_end_to_end(self, settings, fake_llm_client, sample_chunks):
        retriever = self._retriever(settings, fake_llm_client, sample_chunks)
        results = retriever.search(
            "tonnage", years=[2025], document_type=DocumentType.ANNUAL_REPORT, final_k=5
        )
        assert results
        assert all(
            r.metadata.year == 2025 and r.metadata.document_type is DocumentType.ANNUAL_REPORT
            for r in results
        )
