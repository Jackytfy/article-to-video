"""智谱 GLM-backed NLP impl.

Uses Zhipu AI Open API (OpenAI-compatible).
Docs: https://open.bigmodel.cn/dev/api
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


class ZhipuNLPBackend:
    """NLPBackend impl backed by Zhipu AI (智谱 GLM)."""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4",
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        self._model = model

    async def segment(
        self, article: str, target_seconds_per_seg: int = 8
    ) -> list[Segment]:
        prompt = (
            "You split articles into narration-sized chunks for a video voiceover.\n"
            f"Target: each chunk reads aloud in roughly {target_seconds_per_seg} seconds "
            "(about 25-40 Chinese characters or 15-25 English words).\n"
            "Preserve original wording — do not paraphrase. Maintain order.\n"
            "For each chunk extract 3-5 visually concrete English keywords for "
            "stock-image search (nouns, scenes, objects — not abstract emotions).\n\n"
            f"Article:\n```\n{article}\n```"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = _safe_json(response.choices[0].message.content or "{}")
        segs_data: list[dict[str, Any]] = raw.get("segments", [])
        return [
            Segment(
                index=i,
                text=s.get("text", "").strip(),
                keywords=tuple(s.get("keywords", [])),
            )
            for i, s in enumerate(segs_data)
            if s.get("text", "").strip()
        ]

    async def keywords(self, segment: Segment, top_k: int = 5) -> list[str]:
        if segment.keywords:
            return list(segment.keywords[:top_k])
        prompt = (
            f"Extract {top_k} visually concrete English keywords from this text "
            "for stock-image search. Prefer concrete nouns, scenes, objects.\n\n"
            f"Text:\n```\n{segment.text}\n```"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = _safe_json(response.choices[0].message.content or "{}")
        return list(raw.get("keywords", []))[:top_k]

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
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = _safe_json(response.choices[0].message.content or "{}")
        return str(raw.get("translation", text)).strip()

    async def detect_mood(self, article: str) -> str:
        prompt = (
            "Classify the dominant mood of this article into exactly one of: "
            "calm, energetic, sad, positive.\n\n"
            f"Article:\n```\n{article}\n```"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = _safe_json(response.choices[0].message.content or "{}")
        mood = str(raw.get("mood", "calm")).lower()
        return mood if mood in {"calm", "energetic", "sad", "positive"} else "calm"


def _safe_json(text: str) -> dict[str, Any]:
    """Parse JSON, stripping common wrappers if model added them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Model returned non-JSON: %s", exc)
        return {}
