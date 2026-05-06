"""讯飞星火 Spark-backed NLP impl.

Uses iFlytek Spark API (WebSocket-based).
Docs: https://www.xfyun.cn/doc/spark/
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

import websockets

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


class SparkNLPBackend:
    """NLPBackend impl backed by iFlytek Spark (讯飞星火)."""

    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        model: str = "generalv3.5",
    ) -> None:
        self._app_id = app_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._model = model

    def _generate_auth_url(self) -> str:
        """Generate authentication URL with signature."""
        import datetime

        now = datetime.datetime.now()
        date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

        signature_origin = f"host: spark-api.xf-yun.com\ndate: {date}\nGET /v3.5/chat HTTP/1.1"
        signature_sha = hashlib.sha256(signature_origin.encode()).digest()
        signature_sha_b64 = signature_sha.hex()

        authorization_origin = (
            f'api_key="{self._api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha_b64}"'
        )
        authorization = hashlib.b64encode(authorization_origin.encode()).decode()

        return (
            f"wss://spark-api.xf-yun.com/v3.5/chat?"
            f"authorization={authorization}&date={date}&host=spark-api.xf-yun.com"
        )

    async def _chat_raw(
        self, prompt: str, temperature: float = 0
    ) -> str:
        """Send chat request and return response text."""
        url = self._generate_auth_url()

        payload = {
            "header": {
                "app_id": self._app_id,
                "uid": "user_id",
            },
            "parameter": {
                "chat": {
                    "domain": self._model,
                    "temperature": temperature,
                    "max_tokens": 4096,
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": prompt},
                    ]
                }
            },
        }

        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(payload))

            full_response = ""
            while True:
                response = await ws.recv()
                data = json.loads(response)
                code = data.get("header", {}).get("code", 0)

                if code != 0:
                    raise RuntimeError(
                        f"Spark API error: {data.get('header', {}).get('message', 'Unknown error')}"
                    )

                choices = data.get("payload", {}).get("choices", {})
                text = choices.get("text", [])
                for item in text:
                    full_response += item.get("content", "")

                status = choices.get("status", 0)
                if status == 2:
                    break

        return full_response

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
            "Return JSON format: {\"segments\": [{\"text\": \"...\", \"keywords\": [\"...\", \"...\"]}]}\n\n"
            f"Article:\n```\n{article}\n```"
        )
        response = await self._chat_raw(prompt)
        raw = _safe_json(response)
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
            "for stock-image search. Prefer concrete nouns, scenes, objects.\n"
            'Return JSON: {"keywords": ["...", "..."]}\n\n'
            f"Text:\n```\n{segment.text}\n```"
        )
        response = await self._chat_raw(prompt)
        raw = _safe_json(response)
        return list(raw.get("keywords", []))[:top_k]

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if source_lang == target_lang or not target_lang:
            return text
        prompt = (
            f"Translate from {source_lang} to {target_lang}. Preserve tone and meaning. "
            "Return only the translation.\n"
            'Return JSON: {"translation": "..."}\n\n'
            f"Source:\n```\n{text}\n```"
        )
        response = await self._chat_raw(prompt, temperature=0.2)
        raw = _safe_json(response)
        return str(raw.get("translation", text)).strip()

    async def detect_mood(self, article: str) -> str:
        prompt = (
            "Classify the dominant mood of this article into exactly one of: "
            "calm, energetic, sad, positive.\n"
            'Return JSON: {"mood": "one_of_four"}\n\n'
            f"Article:\n```\n{article}\n```"
        )
        response = await self._chat_raw(prompt)
        raw = _safe_json(response)
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
