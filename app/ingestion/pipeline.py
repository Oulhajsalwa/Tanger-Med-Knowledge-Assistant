
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.chunker import chunk_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.loader import RawDocument, discover_pdfs, extract_pdf
from app.models import Chunk
from app.retrieval.bm25 import build_and_persist
from app.retrieval.vector_store import VectorStore
from app.utils.logging import get_logger
from app.utils.openrouter_client import OpenRouterClient

logger = get_logger(__name__)

OFFICIAL_SOURCE_URL = "https://www.tangermed.ma/fr/documentation/"


@dataclass
class IngestionReport:
    processed_files: list[str] = field(default_factory=list)
    skipped_unchanged_files: list[str] = field(default_factory=list)
    skipped_duplicate_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    total_chunks_created: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Ingestion terminée en {self.duration_seconds:.1f}s | "
            f"traités={len(self.processed_files)} "
            f"inchangés={len(self.skipped_unchanged_files)} "
            f"doublons={len(self.skipped_duplicate_files)} "
            f"échecs={len(self.failed_files)} "
            f"chunks={self.total_chunks_created}"
        )


def _load_manifest(settings: Settings) -> dict:
    path = settings.ingestion_manifest_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Manifeste illisible, on repart de zéro : %s", exc)
        return {}


def _save_manifest(settings: Settings, manifest: dict) -> None:
    settings.ingestion_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_ingestion(*, force: bool = False, settings: Settings | None = None) -> IngestionReport:
   
    settings = settings or get_settings()
    start = time.perf_counter()
    report = IngestionReport()

    vector_store = VectorStore(settings)
    llm_client = OpenRouterClient(settings)

    if force:
        logger.info("--force : vector store réinitialisé avant ingestion.")
        vector_store.reset()
        manifest: dict = {}
    else:
        manifest = _load_manifest(settings)

    discovered = discover_pdfs(settings)
    if not discovered:
        logger.warning(
            "Aucun PDF dans '%s' ni '%s' : rien à ingérer.",
            settings.annual_reports_path,
            settings.rse_reports_path,
        )
        report.duration_seconds = time.perf_counter() - start
        return report

    seen_hashes: dict[str, str] = {}  # empreinte -> premier fichier rencontré

    for path, doc_type in discovered:
        try:
            raw = extract_pdf(path, doc_type)
        except Exception as exc:  # noqa: BLE001 - un fichier ne doit pas tout arrêter
            logger.error("Extraction de '%s' en échec : %s", path.name, exc)
            report.failed_files.append(path.name)
            continue

        if raw is None:
            report.failed_files.append(path.name)
            continue

        if raw.file_hash in seen_hashes:
            logger.warning(
                "'%s' (%s) ignoré : copie identique de '%s', déjà ingéré.",
                raw.source_file,
                path.parent.name,
                seen_hashes[raw.file_hash],
            )
            report.skipped_duplicate_files.append(f"{path.parent.name}/{raw.source_file}")
            continue
        seen_hashes[raw.file_hash] = raw.source_file

        if not force and manifest.get(raw.source_file) == raw.file_hash:
            logger.info("'%s' inchangé depuis la dernière ingestion.", raw.source_file)
            report.skipped_unchanged_files.append(raw.source_file)
            continue

        try:
            chunks = _process_document(raw, settings)
            embeddings = llm_client.embed_texts([c.text for c in chunks])
            vector_store.delete_by_source_file(raw.source_file)  # retire les chunks périmés
            vector_store.upsert_chunks(chunks, embeddings)

            manifest[raw.source_file] = raw.file_hash
            report.processed_files.append(raw.source_file)
            report.total_chunks_created += len(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion de '%s' en échec : %s", raw.source_file, exc)
            report.failed_files.append(raw.source_file)

    _save_manifest(settings, manifest)

    if report.processed_files:
        logger.info("Reconstruction de l'index BM25.")
        build_and_persist(settings)

    report.duration_seconds = time.perf_counter() - start
    logger.info(report.summary())
    return report


def _process_document(raw: RawDocument, settings: Settings) -> list[Chunk]:
    raw.pages = clean_pages(raw.pages)
    return chunk_document(raw, source_url=OFFICIAL_SOURCE_URL, settings=settings)
