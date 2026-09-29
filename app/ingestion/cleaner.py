
from __future__ import annotations

from collections import Counter

from app.ingestion.loader import PageContent
from app.utils.logging import get_logger
from app.utils.text import dehyphenate, normalize_whitespace

logger = get_logger(__name__)


MIN_RATIO = 0.35
MIN_COUNT = 3
MAX_LINE_LEN = 140


def _normalized_line(line: str) -> str:
    return " ".join(line.split()).strip().lower()


def detect_boilerplate_lines(pages: list[PageContent]) -> set[str]:
    """Repère les lignes qui se répètent sur de nombreuses pages du document."""
    if len(pages) < 4:
        return set() 
    counts: Counter[str] = Counter()
    for page in pages:
        seen_this_page: set[str] = set()
        for raw_line in page.text.splitlines():
            norm = _normalized_line(raw_line)
            if not norm or len(norm) > MAX_LINE_LEN:
                continue
            if norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            counts[norm] += 1

    threshold = max(MIN_COUNT, int(len(pages) * MIN_RATIO))
    boilerplate = {line for line, count in counts.items() if count >= threshold}
    if boilerplate:
        logger.debug("%d lignes de gabarit détectées.", len(boilerplate))
    return boilerplate


def clean_pages(pages: list[PageContent]) -> list[PageContent]:
   
    boilerplate = detect_boilerplate_lines(pages)

    cleaned: list[PageContent] = []
    for page in pages:
        kept_lines = [
            line for line in page.text.splitlines() if _normalized_line(line) not in boilerplate
        ]
        text = "\n".join(kept_lines)
        text = dehyphenate(text)
        text = normalize_whitespace(text)
        cleaned.append(PageContent(page_number=page.page_number, text=text))

    return cleaned
