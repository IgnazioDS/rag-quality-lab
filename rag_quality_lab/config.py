from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LabConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key for embeddings and judge")
    database_url: str = Field(..., description="PostgreSQL connection string with pgvector")

    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="LAB_EMBEDDING_MODEL",
        description="OpenAI embedding model name",
    )
    judge_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="LAB_JUDGE_MODEL",
        description="OpenAI model for LLM-as-judge scoring",
    )
    judge_cache_path: Path = Field(
        default=Path(".cache/judge.json"),
        validation_alias="LAB_JUDGE_CACHE_PATH",
        description="Path for LLM judge result cache file",
    )
    default_top_k: int = Field(
        default=10,
        validation_alias="LAB_DEFAULT_TOP_K",
        description="Maximum K for retrieval during benchmark",
    )
    results_dir: Path = Field(
        default=Path("results"),
        validation_alias="LAB_RESULTS_DIR",
        description="Directory for benchmark result artifacts",
    )


config = LabConfig()
