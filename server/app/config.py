"""Application configuration — Pydantic Settings, env validation.

Loads and validates environment variables (from the process environment or a
.env file). Required variables raise a clear error at import/startup time if
missing so the app fails fast rather than failing deep inside a request.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REQUIRED_VARS = (
    "GOOGLE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required — no defaults, must be provided via environment or .env
    GOOGLE_API_KEY: str = Field(default="")
    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")
    SUPABASE_URL: str = Field(default="")
    SUPABASE_KEY: str = Field(default="")

    # Optional — sensible defaults matching .env.example
    GEMINI_MODEL: str = "gemini-3.1-pro"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 768
    QDRANT_COLLECTION: str = "enterprise_knowledge"
    PORT: int = 5001
    ENVIRONMENT: str = "development"
    MAX_AGENT_STEPS: int = 6

    @field_validator(*REQUIRED_VARS)
    @classmethod
    def _not_empty(cls, value: str, info):
        if not value or not value.strip():
            raise ValueError(
                f"Missing required environment variable: {info.field_name}. "
                f"Copy server/.env.example to server/.env and fill it in."
            )
        return value


def _load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        missing = [name for name in REQUIRED_VARS if name in str(exc)]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                f"{', '.join(missing)}. Copy server/.env.example to "
                "server/.env and fill in the required values."
            ) from exc
        raise


settings = _load_settings()
