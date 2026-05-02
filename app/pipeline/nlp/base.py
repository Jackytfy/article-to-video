"""NLP backend Protocol — implemented by ollama.py, llm.py, local.py."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.pipeline.models import Segment


@runtime_checkable
class NLPBackend(Protocol):
    """Pluggable NLP layer."""

    async def segment(
        self, article: str, target_seconds_per_seg: int = 8
    ) -> list[Segment]:
        """Split article into narration-sized segments."""
        ...

    async def keywords(self, segment: Segment, top_k: int = 5) -> list[str]:
        """Extract top-k keywords for stock-media search."""
        ...

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate text. Return source unchanged if langs equal."""
        ...

    async def detect_mood(self, article: str) -> str:
        """Single mood label: calm | energetic | sad | positive."""
        ...
