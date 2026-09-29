"""Tests de l'ingestion : lecture, nettoyage, découpage, métadonnées."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.cleaner import clean_pages, detect_boilerplate_lines
from app.ingestion.loader import infer_document_name, infer_year, resolve_document_type
from app.models import ChunkMetadata, DocumentType
from app.utils.text import dehyphenate, looks_like_heading, normalize_whitespace
from tests.factories import make_metadata


# loader
class TestYearInference:
    def test_extracts_year_from_typical_filenames(self):
        assert infer_year("rapport_annuel_2024.pdf") == 2024
        assert infer_year("Rapport-RSE-2024-.pdf") == 2024
        assert infer_year("RAPPORT-RSE-TANGER-MED-2023.pdf") == 2023
        assert infer_year("Rapport_Annuel_Tanger_Med_2025.pdf") == 2025

    def test_returns_none_when_no_year(self):
        assert infer_year("rapport_annuel.pdf") is None
        assert infer_year("document.pdf") is None

    def test_rejects_implausible_years(self):
        # 1999 et 2999 sortent de la fenetre acceptee.
        assert infer_year("rapport_1999.pdf") is None
        assert infer_year("rapport_2999.pdf") is None


class TestDocumentTypeResolution:
    def test_folder_type_is_kept_when_filename_is_consistent(self):
        path = Path("data/annual_reports/Rapport-Annuel-2024.pdf")
        assert resolve_document_type(path, DocumentType.ANNUAL_REPORT) is DocumentType.ANNUAL_REPORT

    def test_filename_wins_over_misleading_folder(self):
        """Cas réel : un rapport RSE rangé par erreur dans annual_reports."""
        path = Path("data/annual_reports/RAPPORT-RSE-TANGER-MED-2023.pdf")
        assert resolve_document_type(path, DocumentType.ANNUAL_REPORT) is DocumentType.RSE_REPORT

    def test_ambiguous_filename_falls_back_to_folder(self):
        path = Path("data/rse_reports/document_2024.pdf")
        assert resolve_document_type(path, DocumentType.RSE_REPORT) is DocumentType.RSE_REPORT

    def test_document_name_is_human_readable(self):
        name = infer_document_name(Path("x_2024.pdf"), DocumentType.RSE_REPORT, 2024)
        assert name == "Rapport RSE 2024"
        name = infer_document_name(Path("x_2024.pdf"), DocumentType.ANNUAL_REPORT, 2024)
        assert name == "Rapport Annuel 2024"


# cleaner
class TestCleaner:
    def test_detects_repeated_footer_as_boilerplate(self, raw_document):
        boilerplate = detect_boilerplate_lines(raw_document.pages)
        assert any("preambule gouvernance social" in line for line in boilerplate)

    def test_removes_boilerplate_but_keeps_content(self, raw_document):
        cleaned = clean_pages(raw_document.pages)
        joined = "\n".join(p.text for p in cleaned)
        assert "Preambule Gouvernance Social Environnement Annexes" not in joined
        assert "Le Groupe Tanger Med a tenu" in joined
        # Les vrais titres de section changent d'une page a l'autre : ils doivent survivre.
        assert "GOUVERNANCE ET GESTION DES RISQUES" in joined

    def test_keeps_page_numbers_aligned(self, raw_document):
        cleaned = clean_pages(raw_document.pages)
        assert [p.page_number for p in cleaned] == [p.page_number for p in raw_document.pages]

    def test_short_documents_are_not_boilerplate_stripped(self):
        from app.ingestion.loader import PageContent

        pages = [PageContent(page_number=i, text="Ligne identique\nContenu") for i in range(1, 4)]
        # En dessous de 4 pages, la repetition n'est pas un signal fiable.
        assert detect_boilerplate_lines(pages) == set()


class TestTextUtils:
    def test_dehyphenate_rejoins_split_words(self):
        assert dehyphenate("environ-\nnement") == "environnement"

    def test_normalize_whitespace(self):
        assert normalize_whitespace("a   b\n\n\n\nc") == "a b\n\nc"

    def test_heading_detection_accepts_real_headings(self):
        assert looks_like_heading("GOUVERNANCE ET GESTION DES RISQUES")
        assert looks_like_heading("ENVIRONNEMENT")
        assert looks_like_heading("Nos Engagements ESG")

    def test_heading_detection_rejects_prose_and_kpi_labels(self):
        assert not looks_like_heading(
            "Le Groupe Tanger Med a tenu 44 reunions de conseil en 2025, un niveau eleve."
        )
        assert not looks_like_heading("11 237 M DH")
        assert not looks_like_heading("107M(T)")
        assert not looks_like_heading("+16%")


# chunker
class TestChunker:
    def test_produces_chunks_within_configured_size(self, raw_document, settings):
        raw_document.pages = clean_pages(raw_document.pages)
        chunks = chunk_document(raw_document, settings=settings)
        assert chunks
        for chunk in chunks:
            # Le recouvrement s'ajoute a chunk_size.
            assert len(chunk.text) <= settings.chunk_size + settings.chunk_overlap + 200

    def test_tracks_page_range_for_citations(self, raw_document, settings):
        raw_document.pages = clean_pages(raw_document.pages)
        chunks = chunk_document(raw_document, settings=settings)
        for chunk in chunks:
            assert chunk.metadata.page_start >= 1
            assert chunk.metadata.page_end >= chunk.metadata.page_start
            assert chunk.metadata.page_end <= raw_document.page_count

    def test_captures_section_titles(self, raw_document, settings):
        raw_document.pages = clean_pages(raw_document.pages)
        chunks = chunk_document(raw_document, settings=settings)
        sections = {c.metadata.section for c in chunks if c.metadata.section}
        assert sections, "no section metadata captured at all"
        assert any("GOUVERNANCE" in s or "ETHIQUE" in s or "SURETE" in s or "ACHATS" in s
                   for s in sections)

    def test_metadata_is_complete_and_propagated(self, raw_document, settings):
        raw_document.pages = clean_pages(raw_document.pages)
        url = "https://www.tangermed.ma/fr/documentation/"
        chunks = chunk_document(raw_document, source_url=url, settings=settings)
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            assert meta.document_name == "Rapport RSE 2025"
            assert meta.document_type is DocumentType.RSE_REPORT
            assert meta.year == 2025
            assert meta.source_file == "rapport_rse_2025.pdf"
            assert meta.source_url == url
            assert meta.chunk_index == i
            assert meta.chunk_id

    def test_chunk_ids_are_unique_and_deterministic(self, raw_document, settings):
        pages_copy = list(raw_document.pages)
        raw_document.pages = clean_pages(pages_copy)
        first = chunk_document(raw_document, settings=settings)
        raw_document.pages = clean_pages(pages_copy)
        second = chunk_document(raw_document, settings=settings)

        ids = [c.metadata.chunk_id for c in first]
        assert len(ids) == len(set(ids)), "chunk ids must be unique"
        assert ids == [c.metadata.chunk_id for c in second], "chunk ids must be deterministic"

    def test_splits_paragraph_longer_than_chunk_size(self, settings):
        from app.ingestion.loader import PageContent, RawDocument

        long_paragraph = " ".join(
            f"Phrase numero {i} du rapport annuel decrivant les performances." for i in range(80)
        )
        doc = RawDocument(
            path=Path("data/annual_reports/rapport_annuel_2024.pdf"),
            document_type=DocumentType.ANNUAL_REPORT,
            year=2024,
            document_name="Rapport Annuel 2024",
            file_hash="a" * 64,
            pages=[PageContent(page_number=1, text=long_paragraph)],
        )
        chunks = chunk_document(doc, settings=settings)
        assert len(chunks) > 1
        assert all(c.metadata.page_start == 1 for c in chunks)


# metadonnees
class TestChunkMetadata:
    def test_page_label_single_and_range(self):
        assert make_metadata(page_start=45).page_label() == "p. 45"
        assert make_metadata(page_start=45, page_end=46).page_label() == "p. 45-46"

    def test_chroma_round_trip_preserves_values(self):
        original = make_metadata(year=2023, document_type=DocumentType.RSE_REPORT, page_start=7)
        restored = ChunkMetadata.from_chroma_metadata(original.to_chroma_metadata())
        assert restored == original

    def test_chroma_metadata_contains_only_primitives(self):
        flat = make_metadata().to_chroma_metadata()
        assert all(isinstance(v, (str, int, float, bool)) for v in flat.values())
