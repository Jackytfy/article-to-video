"""Application configuration loaded from environment via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NLPBackend = Literal["ollama", "llm", "local"]
AspectRatio = Literal["16:9", "9:16", "1:1"]


class Settings(BaseSettings):
    """Runtime config; values come from .env or process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    # Pipeline defaults
    nlp_backend: NLPBackend = "ollama"
    aspect_ratio: AspectRatio = "16:9"
    source_lang: str = "zh"
    translate_to: str | None = None
    voice_primary: str = "zh-CN-XiaoxiaoNeural"
    voice_secondary: str | None = "en-US-AriaNeural"
    bgm_enabled: bool = True
    burn_subtitles: bool = True
    use_gpu: bool = False

    # Media providers
    pexels_api_key: str | None = None
    pixabay_api_key: str | None = None
    unsplash_access_key: str | None = None

    # Music providers
    jamendo_client_id: str | None = None

    # LLM
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-4-5"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Paths
    cache_dir: Path = Field(default=Path("./assets/cache"))
    output_dir: Path = Field(default=Path("./assets/output"))

    @field_validator("translate_to", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("cache_dir", "output_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton settings."""
    return Settings()
