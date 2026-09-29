"""Fixtures partagées.

Les tests tournent hors ligne : le client OpenRouter est remplacé par un
double déterministe et le vector store pointe vers un répertoire temporaire,
jamais vers le vrai `vectorstore/`.
"""

from __future__ import annotations

import hashlib

import pytest

from app.config.settings import Settings
from app.ingestion.loader import PageContent, RawDocument
from app.models import Chunk, DocumentType
from tests.factories import make_metadata

EMBEDDING_DIM = 16  # petit mais non trivial ; 3072 en production


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Configuration isolée, pointant vers des répertoires temporaires."""
    return Settings(
        openrouter_api_key="test-key-not-real",
        openrouter_embedding_dim=EMBEDDING_DIM,
        chunk_size=400,
        chunk_overlap=50,
        top_k=5,
        final_context_k=3,
        retrieval_alpha=0.7,
        relevance_threshold=0.12,
        data_dir=str(tmp_path / "data"),
        vectorstore_dir=str(tmp_path / "vectorstore"),
        collection_name="test_collection",
    )


class FakeOpenRouterClient:
    """Remplaçant déterministe du client OpenRouter, sans appel réseau."""

    def __init__(self, answer: str = "Réponse de test [Rapport Annuel 2024, 2024, p. 12]") -> None:
        self.answer = answer
        self.chat_calls: list[list[dict]] = []
        self.embed_calls: list[list[str]] = []

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(EMBEDDING_DIM)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str, *, model: str | None = None) -> list[float]:
        return self._vector_for(text)

    def chat_completion(self, messages, *, temperature=0.0, max_tokens=1200, model=None):
        from app.utils.openrouter_client import ChatResult

        self.chat_calls.append(messages)
        return ChatResult(content=self.answer, model="fake/model", finish_reason="stop")


@pytest.fixture
def fake_llm_client() -> FakeOpenRouterClient:
    return FakeOpenRouterClient()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Petit corpus multi-années et multi-types pour les tests de recherche."""
    specs = [
        ("Le tonnage global traite par le complexe portuaire s'eleve a 209 millions de tonnes en 2025.", 2025, DocumentType.ANNUAL_REPORT, 9),
        ("Le tonnage global traite par le complexe portuaire s'eleve a 187 millions de tonnes en 2024.", 2024, DocumentType.ANNUAL_REPORT, 41),
        ("Le chiffre d'affaires consolide atteint 13 031 millions de dirhams en 2025, en hausse de 16%.", 2025, DocumentType.ANNUAL_REPORT, 11),
        ("La part de femmes managers atteint 21% en 2025 contre 17% en 2024 au sein du Groupe.", 2025, DocumentType.RSE_REPORT, 56),
        ("Un centre de tri et de valorisation des dechets industriels de 29 000 tonnes par an a ete mis en service.", 2025, DocumentType.RSE_REPORT, 94),
        ("La feuille de route de decarbonation vise la neutralite carbone a l'horizon 2030.", 2024, DocumentType.RSE_REPORT, 74),
    ]
    chunks = []
    for i, (text, year, doc_type, page) in enumerate(specs):
        chunks.append(Chunk(text=text, metadata=make_metadata(
            year=year, document_type=doc_type, page_start=page, chunk_index=i,
            chunk_id=f"chunk-{i}",
        )))
    return chunks


@pytest.fixture
def raw_document() -> RawDocument:
    """Document synthétique de plusieurs pages, avec pieds de page répétés."""
    from pathlib import Path

    footer = "Rapport RSE 2025 Preambule Gouvernance Social Environnement Annexes"
    sections = [
        "GOUVERNANCE ET GESTION DES RISQUES",
        "ETHIQUE DES AFFAIRES",
        "SURETE ET SECURITE DES OPERATIONS",
        "ACHATS RESPONSABLES",
    ]
    pages = []
    for i in range(1, 9):
        section = sections[(i - 1) % len(sections)]
        body = (
            f"Le Groupe Tanger Med a tenu {i * 10} reunions en {2017 + i}, dans le cadre de la "
            f"section {section.lower()}. Cet exercice a permis de consolider les indicateurs "
            f"extra-financiers sur le perimetre du Groupe avec {i * 37} initiatives menees."
        )
        pages.append(PageContent(page_number=i, text=f"{footer}\n{section}\n{body}\n{footer}"))

    return RawDocument(
        path=Path("data/rse_reports/rapport_rse_2025.pdf"),
        document_type=DocumentType.RSE_REPORT,
        year=2025,
        document_name="Rapport RSE 2025",
        file_hash="0" * 64,
        pages=pages,
    )
