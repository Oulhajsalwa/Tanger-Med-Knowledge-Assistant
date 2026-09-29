

from __future__ import annotations

from app.models import QueryAnalysis, RetrievedChunk
from app.rag.prompts import build_messages
from app.utils.logging import get_logger
from app.utils.openrouter_client import OpenRouterClient

logger = get_logger(__name__)


class AnswerGenerator:
    

    def __init__(self, llm_client: OpenRouterClient) -> None:
        self.llm_client = llm_client

    def generate(self, question: str, analysis: QueryAnalysis, chunks: list[RetrievedChunk]) -> str:
        messages = build_messages(question, analysis, chunks)
        result = self.llm_client.chat_completion(messages, temperature=0.0, max_tokens=1200)
        logger.debug(
            "Réponse générée (modèle=%s, fin=%s, tokens prompt=%s, tokens réponse=%s).",
            result.model,
            result.finish_reason,
            result.usage_prompt_tokens,
            result.usage_completion_tokens,
        )
        return result.content
