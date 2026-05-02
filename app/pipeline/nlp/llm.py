"""LLM NLP backend.

Dispatches to Anthropic Claude or OpenAI based on which API key is set.
Uses structured JSON output. Highest quality but paid.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "keywords"],
            },
        }
    },
    "required": ["segments"],
}


def _segment_prompt(article: str, target_seconds: int) -> str:
    return (
        "You split articles into narration-sized chunks for a video voiceover.\n"
        f"Target: each chunk reads aloud in roughly {target_seconds} seconds "
        "(about 25-40 Chinese characters or 15-25 English words).\n"
        "Preserve original wording — do not paraphrase. Maintain order.\n"
        "For each chunk extract 3-5 visually concrete English keywords for "
        "stock-image search.\n\n"
        f"Article:\n```\n{article}\n```\n\n"
        'Return JSON: {"segments": [{"text": "...", "keywords": ["..."]}]}'
    )


class LLMNLPBackend:
    """LLM backend (Anthropic-preferred, OpenAI fallback)."""

    def __init__(
        self,
        model: str,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
    ) -> None:
        self._model = model
        self._anthropic_key = anthropic_key
        self._openai_key = openai_key

        if not (anthropic_key or openai_key):
            raise ValueError(
                "LLMNLPBackend requires ANTHROPIC_API_KEY or OPENAI_API_KEY."
            )

        # Lazy-import per available key to keep startup light.
        if anthropic_key:
            from anthropic import AsyncAnthropic  # type: ignore

            self._anthropic = AsyncAnthropic(api_key=anthropic_key)
            self._provider = "anthropic"
        else:
            from openai import AsyncOpenAI  # type: ignore

            self._openai = AsyncOpenAI(api_key=openai_key)
            self._provider = "openai"

    # ---- Protocol impl -------------------------------------------------------

    async def segment(
        self, article: str, target_seconds_per_seg: int = 8
    ) -> list[Segment]:
        prompt = _segment_prompt(article, target_seconds_per_seg)
        raw = await self._chat_json(prompt)
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
            'for stock-image search. Return JSON: {"keywords": ["..."]}\n\n'
            f"Text:\n```\n{segment.text}\n```"
        )
        raw = await self._chat_json(prompt)
        return list(raw.get("keywords", []))[:top_k]

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if source_lang == target_lang or not target_lang:
            return text
        prompt = (
            f"Translate from {source_lang} to {target_lang}. Preserve tone.\n"
            'Return JSON: {"translation": "..."}\n\n'
            f"Source:\n```\n{text}\n```"
        )
        raw = await self._chat_json(prompt)
        return str(raw.get("translation", text)).strip()

    async def detect_mood(self, article: str) -> str:
        prompt = (
            "Classify the dominant mood of this article into exactly one of: "
            "calm, energetic, sad, positive.\n"
            'Return JSON: {"mood": "<one>"}\n\n'
            f"Article:\n```\n{article}\n```"
        )
        raw = await self._chat_json(prompt)
        mood = str(raw.get("mood", "calm")).lower()
        return mood if mood in {"calm", "energetic", "sad", "positive"} else "calm"

    # ---- Provider dispatch ---------------------------------------------------

    async def _chat_json(self, prompt: str) -> dict[str, Any]:
        if self._provider == "anthropic":
            return await self._anthropic_chat(prompt)
        return await self._openai_chat(prompt)

    async def _anthropic_chat(self, prompt: str) -> dict[str, Any]:
        message = await self._anthropic.messages.create(
            model=self._model,
            max_tokens=4096,
            system=(
                "You are a precise content-to-video assistant. "
                "Always respond with valid JSON matching the requested schema. "
                "No prose outside the JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text  # type: ignore[attr-defined]
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        return _safe_json(text)

    async def _openai_chat(self, prompt: str) -> dict[str, Any]:
        completion = await self._openai.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return _safe_json(completion.choices[0].message.content or "{}")


def _safe_json(text: str) -> dict[str, Any]:
    """Parse JSON, stripping common wrappers if model added them."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences.
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON: %s", exc)
        return {}
