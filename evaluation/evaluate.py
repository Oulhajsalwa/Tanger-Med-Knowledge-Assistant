
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.models import DocumentType, QueryRequest  # noqa: E402
from app.rag.pipeline import build_pipeline  # noqa: E402
from app.utils.logging import get_logger  # noqa: E402
from app.utils.text import strip_accents_lower  # noqa: E402

logger = get_logger(__name__)

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

NUMBER_RE = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


CITATION_RE = re.compile(r"\[[^\]]*\]")
PAGE_REF_RE = re.compile(r"\bp(?:ages?|p)?\.?\s*\d+(?:\s*[-–]\s*\d+)?", re.IGNORECASE)
SOURCES_TAIL_RE = re.compile(r"\n\s*sources?\s*:.*$", re.IGNORECASE | re.DOTALL)
YEAR_RE = re.compile(r"^(19|20)\d{2}$")


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    answer: str = ""
    abstained: bool = False
    should_abstain: bool = False
    retrieved_count: int = 0
    cited_years: list[int] = field(default_factory=list)
    expected_years: list[int] = field(default_factory=list)
    retrieval_hit: bool | None = None
    year_coverage: bool | None = None
    abstention_correct: bool = False
    citation_valid: bool | None = None
    faithfulness_numeric: bool | None = None
    answer_keywords: bool | None = None
    unsupported_numbers: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    error: str | None = None


def extract_numbers(text: str) -> set[str]:
    
    numbers = set()
    for match in NUMBER_RE.finditer(text):
        normalized = re.sub(r"[^\d]", "", match.group())
        if not normalized or YEAR_RE.match(normalized):
            continue
        
        is_percentage = text[match.end() : match.end() + 1] == "%"
        if len(normalized) < 3 and not is_percentage:
            continue
        numbers.add(normalized)
    return numbers


def strip_citations(answer: str) -> str:
    cleaned = SOURCES_TAIL_RE.sub("", answer)
    cleaned = CITATION_RE.sub(" ", cleaned)
    return PAGE_REF_RE.sub(" ", cleaned)


def contains(haystack: str, keyword: str) -> bool:
    return strip_accents_lower(keyword) in strip_accents_lower(haystack)


def evaluate_question(pipeline, spec: dict, corpus_years: set[int]) -> QuestionResult:
    result = QuestionResult(
        id=spec["id"],
        category=spec["category"],
        question=spec["question"],
        should_abstain=spec.get("should_abstain", False),
        expected_years=spec.get("expected_years", []),
    )
    keywords = spec.get("expected_keywords", [])
    doc_type = spec.get("expected_document_type")

    try:
        response = pipeline.answer(
            QueryRequest(
                question=spec["question"],
                document_type=DocumentType(doc_type) if doc_type else None,
            )
        )
    except Exception as exc:  # une question en échec ne doit pas arrêter le run
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error("Question %s en échec : %s", result.id, result.error)
        return result

    result.answer = response.answer
    result.abstained = response.abstained
    result.retrieved_count = response.retrieved_count
    result.sources = [s.model_dump(mode="json") for s in response.sources]
    result.cited_years = sorted({s.year for s in response.sources})
    # Textes complets des chunks, pas les extraits tronqués destinés à l'affichage.
    result.contexts = response.context_texts

    result.abstention_correct = result.abstained == result.should_abstain
    if result.abstained:
        result.citation_valid = not result.sources
        return result

    context = "\n".join(result.contexts)

    result.retrieval_hit = (
        any(contains(context, kw) for kw in keywords) if keywords else result.retrieved_count > 0
    )

    if result.expected_years:
        available = [y for y in result.expected_years if y in corpus_years]
        result.year_coverage = all(y in result.cited_years for y in available) if available else None

    result.citation_valid = bool(result.sources) and all(
        s["document"] and s["year"] and s["page"] and s["year"] in corpus_years
        for s in result.sources
    )

    unsupported = extract_numbers(strip_citations(result.answer)) - extract_numbers(context)
    result.unsupported_numbers = sorted(unsupported)
    result.faithfulness_numeric = not unsupported

    if keywords:
        result.answer_keywords = any(contains(result.answer, kw) for kw in keywords)

    return result


def rate(values: list[bool | None]) -> str:
    considered = [v for v in values if v is not None]
    if not considered:
        return "n/a"
    ok = sum(considered)
    return f"{ok}/{len(considered)} ({100 * ok / len(considered):.0f}%)"


def print_report(results: list[QuestionResult]) -> None:
    print("\n" + "=" * 90)
    print("RÉSULTATS PAR QUESTION")
    print("=" * 90)

    for r in results:
        status = "ERREUR" if r.error else ("ABSTENTION" if r.abstained else "RÉPONSE")
        flag = "OK" if r.abstention_correct and not r.error else "KO"
        print(f"\n[{flag}] {r.id} ({r.category}) - {status}")
        print(f"     Q: {r.question}")

        if r.error:
            print(f"     Erreur: {r.error}")
            continue
        if r.should_abstain:
            print(f"     Abstention attendue: oui | obtenue: {'oui' if r.abstained else 'NON'}")
        if r.abstained:
            continue

        print(f"     Chunks: {r.retrieved_count} | années citées: {r.cited_years or '-'}")
        if r.expected_years:
            print(f"     Années attendues: {r.expected_years} | couverture: {r.year_coverage}")
        print(f"     Fidélité numérique: {r.faithfulness_numeric}", end="")
        print(f" (non sourcés: {r.unsupported_numbers})" if r.unsupported_numbers else "")
        print(f"     Mots-clés attendus: {r.answer_keywords}")
        print(f"     Réponse: {r.answer[:200].strip()}...")

    print("\n" + "=" * 90)
    print("SYNTHÈSE")
    print("=" * 90)
    errors = sum(1 for r in results if r.error)
    print(f"Questions évaluées        : {len(results)} ({errors} erreur(s))")
    print(f"Abstention correcte       : {rate([r.abstention_correct for r in results])}")
    print(f"Retrieval hit             : {rate([r.retrieval_hit for r in results])}")
    print(f"Couverture des années     : {rate([r.year_coverage for r in results])}")
    print(f"Citations valides         : {rate([r.citation_valid for r in results])}")
    print(f"Fidélité numérique        : {rate([r.faithfulness_numeric for r in results])}")
    print(f"Mots-clés dans la réponse : {rate([r.answer_keywords for r in results])}")

    out_of_corpus = [r for r in results if r.category == "out_of_corpus"]
    if out_of_corpus:
        ok = sum(1 for r in out_of_corpus if r.abstained)
        print(f"Hors corpus (abstentions) : {ok}/{len(out_of_corpus)}")
    print("=" * 90 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG Tanger Med.")
    parser.add_argument("--limit", type=int, help="N'évaluer que les N premières questions.")
    parser.add_argument("--output", help="Chemin d'export JSON des résultats.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.has_api_key():
        print("OPENROUTER_API_KEY n'est pas configurée (voir .env).", file=sys.stderr)
        return 2

    pipeline = build_pipeline()
    if pipeline.vector_store.count() == 0:
        print("Vector store vide. Lancez d'abord : python -m app.ingestion", file=sys.stderr)
        return 2

    specs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        specs = specs[: args.limit]

    corpus_years = {d.year for d in pipeline.list_documents()}
    print(f"Corpus : {pipeline.vector_store.count()} chunks | années : {sorted(corpus_years)}")

    results = [evaluate_question(pipeline, spec, corpus_years) for spec in specs]
    print_report(results)

    if args.output:
        payload = {"corpus_years": sorted(corpus_years), "results": [asdict(r) for r in results]}
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Résultats exportés vers {args.output}")

    failed = sum(1 for r in results if r.error or not r.abstention_correct)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
