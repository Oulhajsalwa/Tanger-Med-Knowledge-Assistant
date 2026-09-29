
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.utils.openrouter_client import OpenRouterClient, OpenRouterError  # noqa: E402


def _mask(key: str) -> str:
    if not key:
        return "(vide)"
    return f"{key[:8]}...{key[-4:]} ({len(key)} caractères)"


def main() -> int:
    settings = get_settings()

    print("=" * 78)
    print("DIAGNOSTIC - Tanger Med Knowledge Assistant")
    print("=" * 78)
    print(f"Chat     : {settings.openrouter_model}")
    print(f"           base_url = {settings.openrouter_base_url}")
    print(f"           clé      = {_mask(settings.openrouter_api_key)}")
    print(f"Embeddings: {settings.openrouter_embedding_model} (dim {settings.openrouter_embedding_dim})")
    print(f"           base_url = {settings.resolved_embeddings_base_url}")
    print(f"           clé      = {_mask(settings.resolved_embeddings_api_key)}")
    print("=" * 78)

    if not settings.has_api_key():
        print(
            "\nAucune clé API configurée.\n"
            "Ouvrez le fichier .env et renseignez OPENROUTER_API_KEY=sk-or-v1-...\n",
            file=sys.stderr,
        )
        return 2

    client = OpenRouterClient(settings)
    chat_ok = embeddings_ok = False

    print("\n[1/2] Test du endpoint chat/completions...")
    try:
        result = client.chat_completion(
            [{"role": "user", "content": "Réponds uniquement par le mot: OK"}],
            max_tokens=10,
        )
        print(f"      OK  - modèle réellement utilisé : {result.model}")
        print(f"      Réponse : {result.content[:60]!r}")
        print(f"      Tokens  : prompt={result.usage_prompt_tokens} completion={result.usage_completion_tokens}")
        chat_ok = True
    except OpenRouterError as exc:
        print(f"      ÉCHEC - {exc}")

    print("\n[2/2] Test du endpoint embeddings...")
    try:
        vector = client.embed_query("Tanger Med, premier port de Méditerranée.")
        print(f"      OK  - dimension retournée : {len(vector)}")
        if len(vector) != settings.openrouter_embedding_dim:
            print(
                f"      ATTENTION : la dimension diffère de OPENROUTER_EMBEDDING_DIM "
                f"({settings.openrouter_embedding_dim}). Mettez cette variable à {len(vector)} "
                f"dans .env avant l'ingestion."
            )
        embeddings_ok = True
    except OpenRouterError as exc:
        print(f"      ÉCHEC - {exc}")

    print("\n" + "=" * 78)
    if chat_ok and embeddings_ok:
        print("TOUT EST OPÉRATIONNEL. Prochaine étape :")
        print("    python -m app.ingestion --force")
        print("=" * 78)
        return 0

    if chat_ok and not embeddings_ok:
        print("Le chat fonctionne mais PAS les embeddings.")
        print("Ce fournisseur ne sert pas /embeddings. Dans .env, ajoutez un")
        print("fournisseur d'embeddings compatible OpenAI, par exemple :")
        print("    EMBEDDINGS_BASE_URL=https://api.openai.com/v1")
        print("    EMBEDDINGS_API_KEY=sk-...")
        print("Puis relancez ce script. Aucune modification de code n'est nécessaire.")
    elif not chat_ok:
        print("Le endpoint chat a échoué : vérifiez la validité de la clé, le crédit")
        print("disponible sur le compte, et le nom exact du modèle.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
