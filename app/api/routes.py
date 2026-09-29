

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import DocumentsResponse, HealthResponse
from app.models import QueryRequest, QueryResponse
from app.rag.pipeline import build_pipeline
from app.utils.logging import get_logger
from app.utils.openrouter_client import OpenRouterError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = build_pipeline()
    return HealthResponse(**pipeline.health())


@router.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    pipeline = build_pipeline()
    documents = pipeline.list_documents()
    return DocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.chunk_count for d in documents),
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = build_pipeline()
    try:
        return pipeline.answer(request)
    except OpenRouterError as exc:
        logger.error("Erreur OpenRouter pendant la réponse : %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - pas de trace d'exécution côté client
        logger.exception("Erreur inattendue pendant la réponse.")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from exc
