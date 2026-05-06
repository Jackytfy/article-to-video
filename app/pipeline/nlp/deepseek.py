"""DeepSeek-backed NLP impl.

Uses DeepSeek's OpenAI-compatible API for structured JSON outputs.
Free tier available at https://platform.deepseek.com/
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


class DeepSeekNLPBackend:
    """NLPBackend impl backed by DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._model = model

    async def segment(
        self, article: str, target_seconds_per_seg: int = 8
    ) -> list[Segment]:
        prompt = (
            'Please respond ONLY with valid JSON in this exact format:\n'
            '{"segments": [{"text": "chunk text here", "keywords": ["keyword1", "keyword2"]}]}\n\n'
            "Split this article into narration-sized chunks for a video voiceover.\n"
            f"Target: each chunk reads aloud in roughly {target_seconds_per_seg} seconds "
            "(about 25-40 Chinese characters or 15-25 English words).\n"
            "Preserve original wording. Maintain order.\n\n"
            "IMPORTANT for keywords:\n"
            "- Extract 3-5 visually concrete ENGLISH keywords per segment\n"
            "- Keywords should describe scenes, objects, environments that can be "
            "found as stock video footage\n"
            "- Think about what VIDEO CLIP would match this text\n"
            "- Examples: for '秦始皇统一六国' use ['qin dynasty', 'ancient china war', "
            "'emperor', 'terracotta warriors']\n"
            "- For '宇宙中的黑洞' use ['black hole', 'space', 'galaxy', 'cosmos']\n"
            "- For '人工智能改变生活' use ['artificial intelligence', 'robot', "
            "'smart technology', 'futuristic']\n"
            "- Avoid abstract concepts; prefer concrete visual terms\n\n"
            f"Article:\n{article}"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        logger.info("DeepSeek raw response: %s", content[:500] if content else "None")
        raw = _safe_json(content)
        
        # Handle case where raw is a list
        if isinstance(raw, list):
            segs_data = raw
        elif isinstance(raw, dict):
            segs_data = raw.get("segments", [])
        else:
            logger.error("Unexpected response type: %s", type(raw))
            segs_data = []
        
        segments = []
        for i, s in enumerate(segs_data):
            if isinstance(s, dict):
                text = s.get("text", "").strip()
                if text:
                    segments.append(Segment(
                        index=i,
                        text=text,
                        keywords=tuple(s.get("keywords", [])),
                    ))
        return segments

    async def keywords(self, segment: Segment, top_k: int = 5) -> list[str]:
        if segment.keywords:
            return list(segment.keywords[:top_k])
        prompt = (
            f'Please respond ONLY with valid JSON: {{"keywords": ["kw1", "kw2"]}}\n\n'
            f"Extract {top_k} visually concrete English keywords from this text "
            "for stock-video search.\n"
            "Keywords should describe specific scenes, objects, or environments "
            "that would appear as video footage.\n"
            "Think about what REAL VIDEO CLIP would match this text.\n"
            "Prefer concrete visual terms over abstract concepts.\n\n"
            f"Text: {segment.text}"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = _safe_json(response.choices[0].message.content)
        if isinstance(raw, dict):
            return list(raw.get("keywords", []))[:top_k]
        return []

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if source_lang == target_lang or not target_lang:
            return text
        prompt = (
            f'Please respond ONLY with valid JSON: {{"translation": "translated text"}}\n\n'
            f"Translate from {source_lang} to {target_lang}. Preserve tone and meaning.\n\n"
            f"Source: {text}"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = _safe_json(response.choices[0].message.content)
        if isinstance(raw, dict):
            return str(raw.get("translation", text)).strip()
        return text

    async def detect_mood(self, article: str) -> str:
        prompt = (
            'Please respond ONLY with valid JSON: {"mood": "calm"}\n\n'
            "Classify the dominant mood into exactly one of: calm, energetic, sad, positive.\n\n"
            f"Article: {article}"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = _safe_json(response.choices[0].message.content)
        mood = "calm"
        if isinstance(raw, dict):
            mood = str(raw.get("mood", "calm")).lower()
        return mood if mood in {"calm", "energetic", "sad", "positive"} else "calm"


def _safe_json(text: str | None) -> dict[str, Any] | list[Any]:
    """Parse JSON, stripping common wrappers if model added them."""
    if not text:
        return {}
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
