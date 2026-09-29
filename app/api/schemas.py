
from __future__ import annotations

from pydantic import BaseModel

from app.models import DocumentSummary


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    openrouter_key_configured: bool
    chat_model: str
    embedding_model: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_documents: int
    total_chunks: int
