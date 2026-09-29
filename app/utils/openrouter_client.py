"""Accès au fournisseur LLM via l'API OpenRouter (compatible OpenAI).

Seul module du projet qui parle au fournisseur : changer de modèle ou de
fournisseur ne demande de toucher qu'à ce fichier et au `.env`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)
EMBEDDING_BATCH_SIZE = 64


class OpenRouterError(RuntimeError):
    """Appel au fournisseur en échec, tentatives épuisées."""


@dataclass
class ChatResult:
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None


class OpenRouterClient:
    """Génération et embeddings, avec retries bornés et timeouts."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

        if not self.settings.has_api_key():
            logger.warning(
                "OPENROUTER_API_KEY absente : les appels échoueront. "
                "Renseignez-la dans .env (voir .env.example)."
            )

        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key or "missing-api-key",
            base_url=self.settings.openrouter_base_url,
            timeout=self.settings.request_timeout_seconds,
            default_headers={
                "HTTP-Referer": self.settings.openrouter_site_url,
                "X-Title": self.settings.openrouter_app_name,
            },
        )

        # Les embeddings peuvent vivre sur un autre fournisseur compatible
        # OpenAI : tous les routeurs n'exposent pas /embeddings. Sans
        # surcharge dans .env, c'est le même client.
        if self.settings.embeddings_use_separate_provider:
            self._embeddings_client = OpenAI(
                api_key=self.settings.resolved_embeddings_api_key or "missing-api-key",
                base_url=self.settings.resolved_embeddings_base_url,
                timeout=self.settings.request_timeout_seconds,
            )
            logger.info(
                "Embeddings servis par %s", self.settings.resolved_embeddings_base_url
            )
        else:
            self._embeddings_client = self._client

    def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        model: Optional[str] = None,
    ) -> ChatResult:
        target_model = model or self.settings.openrouter_model

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(RETRYABLE),
        )
        def call():
            return self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        try:
            response = call()
        except RETRYABLE as exc:
            logger.error("Génération en échec après retries : %s", exc)
            raise OpenRouterError(f"Génération en échec : {exc}") from exc
        except APIStatusError as exc:
            logger.error("Génération : statut d'erreur %s", exc.status_code)
            raise OpenRouterError(
                f"Erreur API génération (statut {exc.status_code}) : {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - toute erreur est normalisée
            logger.error("Erreur inattendue à la génération : %s", exc)
            raise OpenRouterError(f"Erreur inattendue à la génération : {exc}") from exc

        if not response.choices:
            raise OpenRouterError("Réponse sans aucun choix.")

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            raise OpenRouterError("Réponse vide du modèle.")

        usage = getattr(response, "usage", None)
        return ChatResult(
            content=content,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage_prompt_tokens=getattr(usage, "prompt_tokens", None),
            usage_completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def embed_texts(self, texts: list[str], *, model: Optional[str] = None) -> list[list[float]]:
        if not texts:
            return []

        target_model = model or self.settings.openrouter_embedding_model
        vectors: list[list[float]] = []
        for batch in batched(texts, EMBEDDING_BATCH_SIZE):
            vectors.extend(self._embed_batch(batch, target_model))
        return vectors

    def embed_query(self, text: str, *, model: Optional[str] = None) -> list[float]:
        return self.embed_texts([text], model=model)[0]

    def _embed_batch(self, batch: list[str], model: str) -> list[list[float]]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(RETRYABLE),
        )
        def call():
            return self._embeddings_client.embeddings.create(model=model, input=batch)

        try:
            response = call()
        except RETRYABLE as exc:
            logger.error("Embeddings en échec après retries : %s", exc)
            raise OpenRouterError(f"Embeddings en échec : {exc}") from exc
        except APIStatusError as exc:
            logger.error("Embeddings : statut d'erreur %s", exc.status_code)
            if exc.status_code in (400, 404, 405, 501):
                raise OpenRouterError(
                    f"L'endpoint /embeddings a répondu {exc.status_code} sur "
                    f"{self.settings.resolved_embeddings_base_url} pour le modèle "
                    f"'{model}'.\n"
                    "Ce fournisseur ne sert probablement pas les embeddings. "
                    "Pointez-les vers un fournisseur compatible OpenAI en renseignant "
                    "EMBEDDINGS_BASE_URL et EMBEDDINGS_API_KEY dans votre .env "
                    f"(aucun changement de code nécessaire). Détail : {exc}"
                ) from exc
            raise OpenRouterError(
                f"Erreur API embeddings (statut {exc.status_code}) : {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur inattendue aux embeddings : %s", exc)
            raise OpenRouterError(f"Erreur inattendue aux embeddings : {exc}") from exc

        # L'API conserve l'ordre d'entrée, mais on trie sur `.index` par sécurité.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]

        expected_dim = self.settings.openrouter_embedding_dim
        for vec in vectors:
            if len(vec) != expected_dim:
                logger.warning(
                    "Dimension d'embedding %d au lieu de %d : vérifiez que "
                    "OPENROUTER_EMBEDDING_DIM correspond au modèle.",
                    len(vec),
                    expected_dim,
                )
                break
        return vectors


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
