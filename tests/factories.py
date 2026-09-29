"""Constructeurs d'objets de test, pour éviter de répéter les métadonnées."""

from __future__ import annotations

from app.models import ChunkMetadata, DocumentType


def make_metadata(
    *,
    year: int = 2024,
    document_type: DocumentType = DocumentType.ANNUAL_REPORT,
    page_start: int = 10,
    page_end: int | None = None,
    section: str | None = "PERFORMANCE OPERATIONNELLE",
    chunk_index: int = 0,
    chunk_id: str | None = None,
) -> ChunkMetadata:
    label = "Rapport RSE" if document_type is DocumentType.RSE_REPORT else "Rapport Annuel"
    return ChunkMetadata(
        document_name=f"{label} {year}",
        document_type=document_type,
        year=year,
        page_start=page_start,
        page_end=page_end if page_end is not None else page_start,
        section=section,
        source_url="https://www.tangermed.ma/fr/documentation/",
        source_file=f"rapport_{document_type.value}_{year}.pdf",
        chunk_index=chunk_index,
        chunk_id=chunk_id or f"chunk-{document_type.value}-{year}-{chunk_index}",
    )
