
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

from app.config import Settings, get_settings
from app.models import DocumentType
from app.utils.logging import get_logger
from app.utils.text import file_sha256

logger = get_logger(__name__)

YEAR_RE = re.compile(r"(20[1-3][0-9])")
MIN_YEAR, MAX_YEAR = 2000, 2035


RSE_HINTS = ("rse", "csr", "durable")
ANNUAL_HINTS = ("annuel", "annual")


@dataclass
class PageContent:
    page_number: int  
    text: str


@dataclass
class RawDocument:
    path: Path
    document_type: DocumentType
    year: int
    document_name: str
    file_hash: str
    pages: list[PageContent] = field(default_factory=list)

    @property
    def source_file(self) -> str:
        return self.path.name

    @property
    def page_count(self) -> int:
        return len(self.pages)


def infer_year(file_name: str) -> int | None:
    match = YEAR_RE.search(file_name)
    if not match:
        return None
    year = int(match.group(1))
    return year if MIN_YEAR <= year <= MAX_YEAR else None


def resolve_document_type(path: Path, folder_type: DocumentType) -> DocumentType:
  
    name = path.name.lower()
    says_rse = any(hint in name for hint in RSE_HINTS)
    says_annual = any(hint in name for hint in ANNUAL_HINTS)

    if says_rse and not says_annual:
        inferred = DocumentType.RSE_REPORT
    elif says_annual and not says_rse:
        inferred = DocumentType.ANNUAL_REPORT
    else:
        return folder_type  # nom ambigu : on garde le dossier

    if inferred is not folder_type:
        logger.warning(
            "'%s' est dans un dossier '%s' mais son nom indique '%s' : on retient '%s'.",
            path.name,
            folder_type.value,
            inferred.value,
            inferred.value,
        )
    return inferred


def infer_document_name(path: Path, document_type: DocumentType, year: int) -> str:

    label = "Rapport RSE" if document_type is DocumentType.RSE_REPORT else "Rapport Annuel"
    return f"{label} {year}"


def discover_pdfs(settings: Settings | None = None) -> list[tuple[Path, DocumentType]]:
    settings = settings or get_settings()
    discovered: list[tuple[Path, DocumentType]] = []

    for folder, doc_type in (
        (settings.annual_reports_path, DocumentType.ANNUAL_REPORT),
        (settings.rse_reports_path, DocumentType.RSE_REPORT),
    ):
        if not folder.exists():
            continue
        for pdf_path in sorted(folder.glob("*.pdf")):
            discovered.append((pdf_path, doc_type))

    return discovered


def extract_pdf(path: Path, document_type: DocumentType) -> RawDocument | None:
  
    document_type = resolve_document_type(path, document_type)
    year = infer_year(path.name)
    if year is None:
        logger.warning(
            "'%s' ignoré : aucune année dans le nom de fichier "
            "(attendu 4 chiffres, ex. 'rapport_annuel_2024.pdf').",
            path.name,
        )
        return None

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ouverture impossible de '%s' : %s", path, exc)
        return None

    pages: list[PageContent] = []
    try:
        for index in range(doc.page_count):
            text = doc.load_page(index).get_text("text") or ""
            pages.append(PageContent(page_number=index + 1, text=text))
    except Exception as exc:  # noqa: BLE001
        logger.error("Extraction interrompue sur '%s' : %s", path, exc)
        return None
    finally:
        doc.close()

    if not pages:
        logger.warning("'%s' ne contient aucune page exploitable.", path.name)
        return None

    logger.info(
        "%d pages extraites de '%s' (%s, %d).",
        len(pages),
        path.name,
        document_type.value,
        year,
    )

    return RawDocument(
        path=path,
        document_type=document_type,
        year=year,
        document_name=infer_document_name(path, document_type, year),
        file_hash=file_sha256(str(path)),
        pages=pages,
    )
