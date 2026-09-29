
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.ingestion.loader import PageContent, RawDocument
from app.models import Chunk, ChunkMetadata
from app.utils.logging import get_logger
from app.utils.text import looks_like_heading, stable_id

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_TRAILING_CHUNK_CHARS = 40


@dataclass
class _Segment:
    page: int
    text: str
    is_heading: bool


def _page_to_segments(page: PageContent) -> list[_Segment]:
   
    segments: list[_Segment] = []
    body_lines: list[str] = []

    def flush_body() -> None:
        if body_lines:
            joined = " ".join(line.strip() for line in body_lines if line.strip())
            if joined:
                segments.append(_Segment(page=page.page_number, text=joined, is_heading=False))
            body_lines.clear()

    for line in page.text.splitlines():
        if not line.strip():
            continue
        if looks_like_heading(line):
            flush_body()
            segments.append(_Segment(page=page.page_number, text=line.strip(), is_heading=True))
        else:
            body_lines.append(line)

    flush_body()
    return segments


def _split_long_paragraph(segment: _Segment, chunk_size: int) -> list[_Segment]:
   
    if len(segment.text) <= chunk_size:
        return [segment]

    sentences = _SENTENCE_SPLIT_RE.split(segment.text)
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > chunk_size and buffer:
            pieces.append(buffer.strip())
            buffer = sentence
        else:
            buffer = candidate
    if buffer.strip():
        pieces.append(buffer.strip())

    
    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            final_pieces.append(piece)
        else:
            for i in range(0, len(piece), chunk_size):
                final_pieces.append(piece[i : i + chunk_size])

    return [_Segment(page=segment.page, text=p, is_heading=False) for p in final_pieces]


def _document_segments(pages: list[PageContent], chunk_size: int) -> list[_Segment]:
    segments: list[_Segment] = []
    for page in pages:
        for seg in _page_to_segments(page):
            if seg.is_heading:
                segments.append(seg)
            else:
                segments.extend(_split_long_paragraph(seg, chunk_size))
    return segments


class _ChunkBuilder:
    """Accumule les segments en chunks, en respectant taille et recouvrement."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._buffer = ""
        self._page_start: int | None = None
        self._page_end: int | None = None
        self._section: str | None = None
        self.flushed: list[tuple[str, int, int, str | None]] = []

    def _reset(self) -> None:
        self._buffer = ""
        self._page_start = None
        self._page_end = None

    def _flush(self) -> None:
        text = self._buffer.strip()
        if text and self._page_start is not None and self._page_end is not None:
            self.flushed.append((text, self._page_start, self._page_end, self._section))
        self._reset()

    def add(self, segment: _Segment, current_section: str | None) -> None:
        piece = segment.text
        prospective_len = len(self._buffer) + (1 if self._buffer else 0) + len(piece)

        if self._buffer and prospective_len > self.chunk_size:
            self._flush()
            if self.chunk_overlap > 0 and self.flushed:
                tail_text = self.flushed[-1][0][-self.chunk_overlap :]
                self._buffer = tail_text + "\n"

        if self._page_start is None:
            self._page_start = segment.page
            self._section = current_section
        self._page_end = segment.page
        self._buffer = f"{self._buffer}{piece}\n" if self._buffer else f"{piece}\n"

    def finalize(self) -> None:
        text = self._buffer.strip()
        if text and len(text) >= _MIN_TRAILING_CHUNK_CHARS and self._page_start is not None:
            self.flushed.append((text, self._page_start, self._page_end or self._page_start, self._section))
        self._reset()


def chunk_document(
    document: RawDocument,
    *,
    source_url: str | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    """Transforme un document nettoyé en liste de chunks prêts à indexer."""
    settings = settings or get_settings()
    segments = _document_segments(document.pages, settings.chunk_size)

    builder = _ChunkBuilder(settings.chunk_size, settings.chunk_overlap)
    current_section: str | None = None

    for segment in segments:
        if segment.is_heading:
            current_section = segment.text
        builder.add(segment, current_section)

    builder.finalize()

    chunks: list[Chunk] = []
    for index, (text, page_start, page_end, section) in enumerate(builder.flushed):
        metadata = ChunkMetadata(
            document_name=document.document_name,
            document_type=document.document_type,
            year=document.year,
            page_start=page_start,
            page_end=page_end,
            section=section,
            source_url=source_url,
            source_file=document.source_file,
            chunk_index=index,
            chunk_id=stable_id(document.source_file, str(index), text[:64]),
        )
        chunks.append(Chunk(text=text, metadata=metadata))

    logger.info(
        "'%s' découpé en %d chunks (taille=%d, recouvrement=%d).",
        document.source_file,
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
