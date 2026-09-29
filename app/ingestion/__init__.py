"""Ingestion : lecture des PDF, nettoyage, découpage, indexation."""

from app.ingestion.pipeline import IngestionReport, run_ingestion

__all__ = ["IngestionReport", "run_ingestion"]
