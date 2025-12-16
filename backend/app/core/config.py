from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application configuration.

    Values are sourced from environment variables or an optional ``.env`` file.
    """

    app_name: str = "CalAI"
    environment: str = "development"
    api_prefix: str = "/api"
    version: str = "0.1.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()