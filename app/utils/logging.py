"""Configuration de la journalisation.

Tous les modules passent par `get_logger(__name__)`. Aucun message ne
contient de clé API ni de valeur d'environnement brute.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure le logger racine, une seule fois par processus."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Bibliothèques tierces bavardes, sauf si l'application est elle-même en DEBUG.
    if resolved_level != "DEBUG":
        for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Logger de module ; configure la journalisation au premier appel."""
    try:
        from app.config import get_settings

        configure_logging(get_settings().log_level)
    except Exception:  # pragma: no cover - les settings peuvent ne pas être prêts
        configure_logging("INFO")
    return logging.getLogger(name)
