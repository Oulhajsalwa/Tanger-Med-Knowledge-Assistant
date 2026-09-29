"""Tests de la couche FastAPI.

L'API n'est qu'un adaptateur : ces tests vérifient le contrat — codes de
statut, forme des réponses, traduction des erreurs — avec un pipeline
bouchonné, donc sans réseau, sans vector store et sans LLM.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.models import (
    DocumentSummary,
    DocumentType,
    QueryAnalysis,
    QueryIntent,
    QueryResponse,
    Source,
)
from app.utils.openrouter_client import OpenRouterError


class StubPipeline:
    """Pipeline minimal implémentant ce que les routes attendent."""

    def __init__(self, *, raise_error: Exception | None = None) -> None:
        self.raise_error = raise_error
        self.received_requests: list = []

    def health(self) -> dict:
        return {
            "status": "ok",
            "chunks_indexed": 1647,
            "openrouter_key_configured": True,
            "chat_model": "openai/gpt-4o",
            "embedding_model": "openai/text-embedding-3-large",
        }

    def list_documents(self) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                document_name="Rapport Annuel 2024",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2024,
                source_file="Rapport-Annuel-2024-.pdf",
                chunk_count=555,
            ),
            DocumentSummary(
                document_name="Rapport RSE 2024",
                document_type=DocumentType.RSE_REPORT,
                year=2024,
                source_file="Rapport-RSE-2024-.pdf",
                chunk_count=389,
            ),
        ]

    def answer(self, request) -> QueryResponse:
        self.received_requests.append(request)
        if self.raise_error:
            raise self.raise_error
        return QueryResponse(
            answer="Le tonnage global a atteint 209 millions de tonnes en 2025.",
            sources=[
                Source(
                    document="Rapport Annuel 2025",
                    document_type=DocumentType.ANNUAL_REPORT,
                    year=2025,
                    page="p. 9",
                    section="NOS CHIFFRES CLEFS",
                    source_url="https://www.tangermed.ma/fr/documentation/",
                    excerpt="209 MT de marchandises traitees.",
                    relevance_score=0.83,
                )
            ],
            abstained=False,
            query_analysis=QueryAnalysis(
                raw_question=request.question,
                years=[2025],
                intent=QueryIntent.FACTUAL,
            ),
            retrieved_count=5,
            timings_ms={"total_ms": 1234.5},
        )


@pytest.fixture
def client_and_stub(monkeypatch):
    stub = StubPipeline()
    monkeypatch.setattr("app.api.routes.build_pipeline", lambda *a, **k: stub)
    return TestClient(create_app()), stub


class TestHealthEndpoint:
    def test_returns_index_state(self, client_and_stub):
        client, _ = client_and_stub
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["chunks_indexed"] == 1647
        assert body["chat_model"] == "openai/gpt-4o"


class TestDocumentsEndpoint:
    def test_lists_corpus_inventory_with_totals(self, client_and_stub):
        client, _ = client_and_stub
        response = client.get("/api/documents")
        assert response.status_code == 200
        body = response.json()
        assert body["total_documents"] == 2
        assert body["total_chunks"] == 944
        assert body["documents"][0]["document_name"] == "Rapport Annuel 2024"


class TestQueryEndpoint:
    def test_returns_answer_and_sources(self, client_and_stub):
        client, _ = client_and_stub
        response = client.post(
            "/api/query",
            json={"question": "Quel a été le tonnage global en 2025 ?", "years": [2025]},
        )
        assert response.status_code == 200
        body = response.json()
        assert "209 millions de tonnes" in body["answer"]
        assert body["abstained"] is False
        assert body["sources"][0]["page"] == "p. 9"
        assert body["sources"][0]["year"] == 2025

    def test_forwards_filters_to_the_pipeline(self, client_and_stub):
        client, stub = client_and_stub
        client.post(
            "/api/query",
            json={
                "question": "Actions environnementales ?",
                "years": [2023, 2024],
                "document_type": "rse_report",
            },
        )
        request = stub.received_requests[-1]
        assert request.years == [2023, 2024]
        assert request.document_type is DocumentType.RSE_REPORT

    def test_years_are_optional(self, client_and_stub):
        client, stub = client_and_stub
        response = client.post("/api/query", json={"question": "Combien de collaborateurs ?"})
        assert response.status_code == 200
        assert stub.received_requests[-1].years is None

    def test_rejects_too_short_question(self, client_and_stub):
        client, _ = client_and_stub
        assert client.post("/api/query", json={"question": "ab"}).status_code == 422

    def test_rejects_missing_question(self, client_and_stub):
        client, _ = client_and_stub
        assert client.post("/api/query", json={}).status_code == 422

    def test_rejects_invalid_document_type(self, client_and_stub):
        client, _ = client_and_stub
        response = client.post(
            "/api/query", json={"question": "Question valide ?", "document_type": "unknown_type"}
        )
        assert response.status_code == 422

    def test_llm_provider_failure_maps_to_502(self, monkeypatch):
        stub = StubPipeline(raise_error=OpenRouterError("quota exceeded"))
        monkeypatch.setattr("app.api.routes.build_pipeline", lambda *a, **k: stub)
        client = TestClient(create_app())

        response = client.post("/api/query", json={"question": "Question valide ?"})

        assert response.status_code == 502
        assert "quota" in response.json()["detail"]

    def test_unexpected_failure_maps_to_500_without_leaking_details(self, monkeypatch):
        stub = StubPipeline(raise_error=RuntimeError("internal boom with secrets"))
        monkeypatch.setattr("app.api.routes.build_pipeline", lambda *a, **k: stub)
        client = TestClient(create_app())

        response = client.post("/api/query", json={"question": "Question valide ?"})

        assert response.status_code == 500
        assert "boom" not in response.json()["detail"]


class TestRootAndDocs:
    def test_root_points_to_docs(self, client_and_stub):
        client, _ = client_and_stub
        body = client.get("/").json()
        assert body["service"] == "tanger-med-rag"

    def test_openapi_schema_is_generated(self, client_and_stub):
        client, _ = client_and_stub
        schema = client.get("/openapi.json").json()
        assert "/api/query" in schema["paths"]
        assert "/api/health" in schema["paths"]
        assert "/api/documents" in schema["paths"]
