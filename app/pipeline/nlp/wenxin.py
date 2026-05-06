"""百度文心一言 ERNIE-backed NLP impl.

Uses Baidu Wenxin AI API (ERNIE Bot).
Docs: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flzn5wntt
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import httpx

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


class WenxinNLPBackend:
    """NLPBackend impl backed by Baidu Wenxin (文心一言 ERNIE)."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        model: str = "ernie-4.0-8k-latest",
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._model = model
        self._access_token: str | None = None
        self._token_expires_at: float = 0

    async def _get_access_token(self) -> str:
        """Get access token using API key and secret key."""
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self._api_key,
            "client_secret": self._secret_key,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 30 * 24 * 3600)
            self._token_expires_at = time.time() + expires_in - 300

        return self._access_token

    async def _chat_json(self, prompt: str, temperature: float = 0) -> dict[str, Any]:
        """Call Wenxin ERNIE API with JSON output."""
        access_token = await self._get_access_token()

        url = (
            f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"
            f"completions?access_token={access_token}"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                json={
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            if "error_code" in data:
                raise RuntimeError(
                    f"Wenxin API error {data.get('error_code')}: "
                    f"{data.get('error_msg')}"
                )

            return _safe_json(data.get("result", "{}"))

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
            'Return JSON: {"segments": [{"text": "...", "keywords": ["...", "..."]}]}\n\n'
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
            'for stock-image search. Return JSON: {"keywords": ["...", "..."]}\n\n'
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
            f"Translate from {source_lang} to {target_lang}. Preserve tone.\n"
            'Return JSON: {"translation": "..."}\n\n'
            f"Source:\n```\n{text}\n```"
        )
        response = await self._chat_json(prompt, temperature=0.2)
        return str(response.get("translation", text)).strip()

    async def detect_mood(self, article: str) -> str:
        prompt = (
            "Classify the dominant mood of this article into exactly one of: "
            "calm, energetic, sad, positive.\n"
            'Return JSON: {"mood": "one_of_four"}\n\n'
            f"Article:\n```\n{article}\n```"
        )
        response = await self._chat_json(prompt)
        mood = str(response.get("mood", "calm")).lower()
        return mood if mood in {"calm", "energetic", "sad", "positive"} else "calm"


def _safe_json(text: str) -> dict[str, Any]:
    """Parse JSON, stripping common wrappers."""
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
