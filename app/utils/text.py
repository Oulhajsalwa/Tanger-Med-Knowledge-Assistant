"""Utilitaires de texte partagés par le reste du pipeline."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_whitespace(text: str) -> str:
    """Réduit les espaces répétés et les lignes vides en trop."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Recolle les mots coupés par un tiret en fin de ligne ("environ-\\nnement")."""
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def strip_accents_lower(text: str) -> str:
    """Minuscules sans accents, pour comparer des mots-clés français."""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return without_accents.lower()


def looks_like_heading(line: str) -> bool:
    """Cette ligne est-elle un titre de section plutôt que du texte courant ?

    L'extraction PDF perd la taille et la graisse des polices : on se rabat
    sur des indices de surface fréquents dans les rapports d'entreprise —
    ligne courte, pas de ponctuation finale, forte proportion de majuscules.
    """
    stripped = line.strip()
    if not (3 <= len(stripped) <= 90):
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False

    # Les libellés de KPI dont ces rapports très infographiques sont remplis
    # ("107M(T)", "11 237 M DH", "+16%") sont courts et quasi tout en
    # majuscules : sans ce filtre ils passeraient pour des titres et
    # pollueraient la métadonnée `section`. Un titre d'un seul mot
    # ("GOUVERNANCE") reste légitime, d'où l'exigence d'un vrai mot plutôt
    # que d'une longueur minimale.
    digits = sum(1 for c in stripped if c.isdigit())
    if digits / len(stripped) > 0.35:
        return False
    if not any(len(w) > 2 and any(c.isalpha() for c in w) for w in stripped.split()):
        return False

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio > 0.75:
        return True
    # Ligne courte en casse de titre : titre plausible aussi.
    if len(stripped.split()) <= 8 and stripped[0].isupper() and upper_ratio > 0.25:
        return True
    return False


def stable_id(*parts: str) -> str:
    """Identifiant court et déterministe dérivé des parties fournies."""
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: str) -> str:
    """Empreinte du fichier, pour détecter qu'un PDF a changé."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate(text: str, max_chars: int = 220) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
