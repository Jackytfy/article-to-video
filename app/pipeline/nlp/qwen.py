"""通义千问 (Qwen) backed NLP impl.

Uses Alibaba Cloud DashScope API.
Docs: https://help.aliyun.com/zh/dashscope/
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


class QwenNLPBackend:
    """NLPBackend impl backed by Alibaba Cloud DashScope (通义千问)."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-turbo",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = "https://dashscope.aliyuncs.com/api/v1"

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
        response = await self._chat_json(prompt)
        segs_data: list[dict[str, Any]] = response.get("segments", [])
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
        response = await self._chat_json(prompt)
        return list(response.get("keywords", []))[:top_k]

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
        response = await self._chat_json(prompt)
        return str(response.get("translation", text)).strip()

    async def detect_mood(self, article: str) -> str:
        prompt = (
            "Classify the dominant mood of this article into exactly one of: "
            "calm, energetic, sad, positive.\n\n"
            f"Article:\n```\n{article}\n```"
        )
        response = await self._chat_json(prompt)
        mood = str(response.get("mood", "calm")).lower()
        return mood if mood in {"calm", "energetic", "sad", "positive"} else "calm"

    async def _chat_json(self, prompt: str) -> dict[str, Any]:
        """Call DashScope chat API with JSON output."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/services/aigc/text-generation/generation",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": {"messages": [{"role": "user", "content": prompt}]},
                    "parameters": {
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code"):
                raise RuntimeError(f"DashScope API error: {data.get('message')}")

            output = data.get("output", {})
            text = output.get("text", "{}")
            return _safe_json(text)


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
