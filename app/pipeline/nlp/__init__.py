"""NLP stage: pluggable backends + factory.

Selects an NLPBackend based on settings.nlp_backend.
Heavy backend imports are lazy so the unused ones never load.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.pipeline.nlp.base import NLPBackend


def make_backend(
    backend: str | None = None,
    settings: Settings | None = None,
) -> NLPBackend:
    """Build the configured NLPBackend.

    Args:
        backend: optional override ("ollama" | "llm" | "local"). Falls back to settings.
        settings: optional injected settings (testing).
    """
    settings = settings or get_settings()
    backend = backend or settings.nlp_backend

    if backend == "ollama":
        from app.pipeline.nlp.ollama import OllamaNLPBackend

        return OllamaNLPBackend(
            host=settings.ollama_host,
            model=settings.ollama_model,
        )

    if backend == "local":
        from app.pipeline.nlp.local import LocalNLPBackend

        return LocalNLPBackend()

    if backend == "llm":
        from app.pipeline.nlp.llm import LLMNLPBackend

        return LLMNLPBackend(
            model=settings.llm_model,
            anthropic_key=settings.anthropic_api_key,
            openai_key=settings.openai_api_key,
        )

    raise ValueError(f"Unknown NLP backend: {backend!r}")


__all__ = ["make_backend"]
