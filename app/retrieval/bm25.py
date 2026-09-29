"""Recherche lexicale BM25, complémentaire de la recherche vectorielle.

Les rapports sont denses en chiffres, acronymes et noms propres ("209 millions
de tonnes", "EVP", "Nador West Med") que la recherche sémantique sous-pondère.
L'index est construit à l'ingestion sur tout le corpus et persisté sur disque.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.models import ChunkMetadata, DocumentType, RetrievedChunk
from app.utils.logging import get_logger
from app.utils.text import strip_accents_lower

logger = get_logger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")

FR_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "a", "au",
    "aux", "en", "pour", "par", "sur", "dans", "est", "sont", "que", "qui",
    "quel", "quelle", "quels", "quelles", "avec", "sans", "ce", "cette",
    "ces", "il", "elle", "ils", "elles", "nous", "vous", "je", "tu", "se",
    "sa", "son", "ses", "leur", "leurs", "comment", "quand", "entre",
}


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(strip_accents_lower(text))
    return [t for t in tokens if t not in FR_STOPWORDS]


@dataclass
class Bm25Index:
    bm25: BM25Okapi
    texts: list[str]
    metadatas: list[ChunkMetadata]
    ids: list[str]
    # Constante de saturation de la normalisation (voir `search`). Stockée sur
    # l'index pour qu'un index persisté score toujours comme à sa construction.
    saturation_k: float = 15.0

    @classmethod
    def build(
        cls,
        texts: list[str],
        metadatas: list[ChunkMetadata],
        ids: list[str],
        *,
        saturation_k: float = 15.0,
    ) -> "Bm25Index":
        bm25 = BM25Okapi([tokenize(t) for t in texts])
        logger.info("Index BM25 construit sur %d chunks.", len(texts))
        return cls(bm25, texts, metadatas, ids, saturation_k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Index BM25 (%d documents) enregistré dans '%s'.", len(self.texts), path)

    @classmethod
    def load(cls, path: Path) -> Optional["Bm25Index"]:
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        years: Optional[list[int]] = None,
        document_type: Optional[DocumentType] = None,
    ) -> list[RetrievedChunk]:
        if not self.texts:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)

        # rank_bm25 ne gère pas les filtres : on masque après scoring.
        mask = np.ones(len(self.texts), dtype=bool)
        if years:
            year_set = set(years)
            mask &= np.array([m.year in year_set for m in self.metadatas])
        if document_type is not None:
            mask &= np.array([m.document_type == document_type for m in self.metadatas])
        scores = np.where(mask, scores, -np.inf)

        if not np.any(np.isfinite(scores)) or scores.max() <= 0:
            return []

        top_indices = [
            i for i in np.argsort(scores)[::-1][:top_k]
            if np.isfinite(scores[i]) and scores[i] > 0
        ]
        if not top_indices:
            return []

        return [
            RetrievedChunk(
                text=self.texts[i],
                metadata=self.metadatas[i],
                lexical_score=float(scores[i]) / (float(scores[i]) + self.saturation_k),
            )
            for i in top_indices
        ]


def build_and_persist(settings: Optional[Settings] = None) -> Bm25Index:
    from app.retrieval.vector_store import VectorStore  # import local : évite un cycle

    settings = settings or get_settings()
    texts, metadatas, ids = VectorStore(settings).all_chunks_raw()
    index = Bm25Index.build(texts, metadatas, ids, saturation_k=settings.bm25_saturation_k)
    index.save(settings.bm25_index_path)
    return index


def load_or_build(settings: Optional[Settings] = None) -> Bm25Index:
    settings = settings or get_settings()
    index = Bm25Index.load(settings.bm25_index_path)
    if index is not None:
        return index
    logger.warning("Aucun index BM25 persisté : reconstruction à la volée.")
    return build_and_persist(settings)
