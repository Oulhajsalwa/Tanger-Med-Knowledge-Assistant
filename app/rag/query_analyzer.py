from __future__ import annotations

import re

from app.models import DocumentType, QueryAnalysis, QueryIntent
from app.utils.text import strip_accents_lower

_YEAR_RE = re.compile(r"\b(20[0-3][0-9])\b")
_RANGE_RE = re.compile(
    r"\bentre\s+(20[0-3][0-9])\s+et\s+(20[0-3][0-9])\b"
    r"|\bde\s+(20[0-3][0-9])\s+(?:a|à)\s+(20[0-3][0-9])\b",
)

_COMPARISON_KEYWORDS = [
    "compar", "versus", " vs ", "par rapport a", "differenc", "ecart entre",
]
_EVOLUTION_KEYWORDS = [
    "evolu", "tendance", "progression", "croissance", "depuis", "au fil des annees",
    "au cours des annees",
]
_RSE_KEYWORDS = [
    "rapport rse", "responsabilite societale", "rse ", " rse", "developpement durable",
    "environnement", "rse.", "csr",
]
_ANNUAL_KEYWORDS = [
    "rapport annuel", "resultats financiers", "chiffre d'affaires", "chiffre daffaires",
    "ebe", "trafic portuaire", "resultat net",
]


TOPIC_KEYWORDS: dict[str, list[str]] = {
    "gouvernance": ["gouvernance", "conseil de surveillance", "directoire", "conformite", "ethique"],
    "social": ["social", "collaborateurs", "ressources humaines", "emploi", "diversite", "formation"],
    "environnement": [
        "environnement", "carbone", "decarbonation", "emissions", "energie", "biodiversite",
        "dechets", "eau", "climat", "durable",
    ],
    "trafic_portuaire": ["trafic", "conteneurs", "evp", "port", "navires", "manutention"],
    "finance": ["chiffre d'affaires", "chiffre daffaires", "resultat net", "ebe", "investissement", "financier"],
    "digital": ["digital", "numerique", "intelligence artificielle", "ia ", "technologie", "cybersecurite"],
    "communautes_locales": ["fondation", "communautes locales", "education", "sante", "inclusion"],
    "international": ["international", "afrique", "export", "international"],
}


def _normalize(text: str) -> str:
    return strip_accents_lower(text)


def _extract_years(normalized_question: str, *, expand_range: bool) -> list[int]:
  
    if expand_range:
        range_match = _RANGE_RE.search(normalized_question)
        if range_match:
            groups = [g for g in range_match.groups() if g]
            if len(groups) == 2:
                start, end = sorted(int(g) for g in groups)
                if end - start <= 20:
                    return list(range(start, end + 1))

    years = sorted({int(y) for y in _YEAR_RE.findall(normalized_question)})
    return years


def _matches_any(normalized_question: str, keywords: list[str]) -> bool:
    return any(kw in normalized_question for kw in keywords)


def _extract_topics(normalized_question: str) -> list[str]:
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if _matches_any(normalized_question, [_normalize(k) for k in keywords]):
            topics.append(topic)
    return topics


def _extract_document_type(normalized_question: str) -> DocumentType | None:
    is_rse = _matches_any(normalized_question, [_normalize(k) for k in _RSE_KEYWORDS])
    is_annual = _matches_any(normalized_question, [_normalize(k) for k in _ANNUAL_KEYWORDS])
    if is_rse and not is_annual:
        return DocumentType.RSE_REPORT
    if is_annual and not is_rse:
        return DocumentType.ANNUAL_REPORT
    return None  # ambigu ou non précisé : on cherche dans les deux


def analyze_query(question: str) -> QueryAnalysis:
    
    normalized = _normalize(question)

    is_comparison = _matches_any(normalized, _COMPARISON_KEYWORDS)
    is_evolution = _matches_any(normalized, _EVOLUTION_KEYWORDS)

    years = _extract_years(normalized, expand_range=is_evolution and not is_comparison)
    topics = _extract_topics(normalized)
    document_type = _extract_document_type(normalized)

    if is_comparison and len(years) >= 2:
        intent = QueryIntent.COMPARISON
    elif is_evolution and len(years) >= 2:
        intent = QueryIntent.TEMPORAL_EVOLUTION
    elif is_comparison or (is_evolution and len(years) <= 1):
      
        intent = QueryIntent.TEMPORAL_EVOLUTION if is_evolution else QueryIntent.COMPARISON
    elif not years and not topics:
        intent = QueryIntent.FACTUAL
    elif not years and topics:
        intent = QueryIntent.GENERAL_SYNTHESIS
    else:
        intent = QueryIntent.FACTUAL

    return QueryAnalysis(
        raw_question=question,
        years=years,
        document_type=document_type,
        topics=topics,
        intent=intent,
        is_comparison=is_comparison or intent == QueryIntent.COMPARISON,
    )
