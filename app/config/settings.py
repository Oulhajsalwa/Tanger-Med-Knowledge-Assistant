
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenRouter
    openrouter_api_key: str = Field(default="", description="Clé API OpenRouter")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="openai/gpt-4o")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-large")
    openrouter_embedding_dim: int = Field(default=3072, gt=0)
    openrouter_site_url: str = Field(default="https://github.com/your-org/tanger-med-rag")
    openrouter_app_name: str = Field(default="Tanger Med Knowledge Assistant")


    embeddings_base_url: str = Field(default="")
    embeddings_api_key: str = Field(default="")

    # Découpage
    chunk_size: int = Field(default=1000, gt=100, description="Taille cible d'un chunk, en caractères")
    chunk_overlap: int = Field(default=150, ge=0, description="Recouvrement entre chunks consécutifs")

    # Recherche
    top_k: int = Field(default=8, gt=0, description="Candidats remontés par CHAQUE retriever avant fusion")
    final_context_k: int = Field(default=5, gt=0, description="Chunks conservés dans le contexte final")
    retrieval_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Poids du sémantique face au lexical")
    relevance_threshold: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Score fusionné minimal pour répondre au lieu de s'abstenir"
    )
    min_semantic_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Similarité cosinus minimale du meilleur passage pour juger la question "
            "dans le sujet. Le lexical seul ne peut pas franchir cette barre, ce qui "
            "rend l'abstention thématique possible sur un corpus saturé de vocabulaire "
            "économique courant."
        ),
    )
    bm25_saturation_k: float = Field(
        default=15.0,
        gt=0.0,
        description=(
            "Constante de saturation de la normalisation BM25 raw/(raw+k). "
            "Calibrée sur ce corpus ; plus bas = saturation plus rapide."
        ),
    )

    # Stockage
    data_dir: str = Field(default="data")
    vectorstore_dir: str = Field(default="vectorstore")
    collection_name: str = Field(default="tanger_med_reports")

  
    request_timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)
    log_level: str = Field(default="INFO")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
        return v

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def annual_reports_path(self) -> Path:
        return self.data_path / "annual_reports"

    @property
    def rse_reports_path(self) -> Path:
        return self.data_path / "rse_reports"

    @property
    def vectorstore_path(self) -> Path:
        return self._resolve(self.vectorstore_dir)

    @property
    def bm25_index_path(self) -> Path:
        return self.vectorstore_path / "bm25_index.pkl"

    @property
    def ingestion_manifest_path(self) -> Path:
        """Trace les fichiers déjà ingérés et leur empreinte."""
        return self.vectorstore_path / "ingestion_manifest.json"

    def _resolve(self, relative_or_absolute: str) -> Path:
        p = Path(relative_or_absolute)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def ensure_directories(self) -> None:
        for path in (
            self.annual_reports_path,
            self.rse_reports_path,
            self.vectorstore_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip())

    @property
    def resolved_embeddings_base_url(self) -> str:
        """Endpoint d'embeddings : la surcharge dédiée, sinon OpenRouter."""
        return (self.embeddings_base_url or self.openrouter_base_url).strip()

    @property
    def resolved_embeddings_api_key(self) -> str:
        return (self.embeddings_api_key or self.openrouter_api_key).strip()

    @property
    def embeddings_use_separate_provider(self) -> bool:
        return self.resolved_embeddings_base_url != self.openrouter_base_url.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance unique par processus, mise en cache."""
    settings = Settings()
    settings.ensure_directories()
    return settings
