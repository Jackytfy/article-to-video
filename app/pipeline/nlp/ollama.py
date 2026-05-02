"""Ollama-backed NLP impl.

Uses ollama.AsyncClient + JSON-schema structured outputs for deterministic parsing.
Default backend (free, local).
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from ollama import AsyncClient
from pydantic import BaseModel, Field

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


# ---- Structured output schemas ----------------------------------------------


class _SegmentOut(BaseModel):
    text: str = Field(..., description="One narration-sized chunk of the article.")
    keywords: list[str] = Field(
        default_factory=list,
        description="3-5 visually concrete English keywords for stock-image search.",
    )


class _SegmentList(BaseModel):
    segments: list[_SegmentOut]


class _KeywordList(BaseModel):
    keywords: list[str]


class _Translation(BaseModel):
    translation: str


Mood = Literal["calm", "energetic", "sad", "positive"]


class _Mood(BaseModel):
    mood: Mood


# ---- Backend impl ------------------------------------------------------------


class OllamaNLPBackend:
    """NLPBackend impl backed by a local Ollama server."""

    def __init__(self, host: str, model: str) -> None:
        self._client = AsyncClient(host=host)
        self._model = model

    async def segment(
        self, article: str, target_seconds_per_seg: int = 8
    ) -> list[Segment]:
        prompt = (
            "You split articles into narration-sized chunks for a video voiceover.\n"
            f"Target: each chunk reads aloud in roughly {target_seconds_per_seg} seconds "
            "(about 25-40 Chinese characters or 15-25 English words).\n"
            "Preserve original wording — do not paraphrase. Maintain order.\n"
            "For each chunk extract 3-5 visually concrete English keywords usable for "
            "stock-image search (nouns, scenes, objects — not abstract emotions).\n\n"
            f"Article:\n```\n{article}\n```"
        )
        response = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=_SegmentList.model_json_schema(),
            options={"temperature": 0},
        )
        parsed = _SegmentList.model_validate_json(response["message"]["content"])
        return [
            Segment(
                index=i,
                text=s.text.strip(),
                keywords=tuple(s.keywords),
            )
            for i, s in enumerate(parsed.segments)
            if s.text.strip()
        ]

    async def keywords(self, segment: Segment, top_k: int = 5) -> list[str]:
        if segment.keywords:
            return list(segment.keywords[:top_k])
        prompt = (
            f"Extract {top_k} visually concrete English keywords from this text "
            "for stock-image search. Prefer concrete nouns, scenes, objects.\n\n"
            f"Text:\n```\n{segment.text}\n```"
        )
        response = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=_KeywordList.model_json_schema(),
            options={"temperature": 0},
        )
        parsed = _KeywordList.model_validate_json(response["message"]["content"])
        return parsed.keywords[:top_k]

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if source_lang == target_lang or not target_lang:
            return text
        prompt = (
            f"Translate from {source_lang} to {target_lang}. Preserve tone and meaning. "
            "Return only the translation, no commentary.\n\n"
            f"Source:\n```\n{text}\n```"
        )
        response = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=_Translation.model_json_schema(),
            options={"temperature": 0.2},
        )
        parsed = _Translation.model_validate_json(response["message"]["content"])
        return parsed.translation.strip()

    async def detect_mood(self, article: str) -> str:
        prompt = (
            "Classify the dominant mood of this article into exactly one of: "
            "calm, energetic, sad, positive.\n\n"
            f"Article:\n```\n{article}\n```"
        )
        response = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            format=_Mood.model_json_schema(),
            options={"temperature": 0},
        )
        try:
            parsed = _Mood.model_validate_json(response["message"]["content"])
            return parsed.mood
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Mood parse failed: %s. Defaulting to 'calm'.", exc)
            return "calm"
