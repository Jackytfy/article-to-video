"""Pexels stock-media provider.

Free API: https://www.pexels.com/api/. Free tier: 200 req/hour, 20k req/month.
Auth: Authorization header with API key (no Bearer prefix).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.pipeline.models import MediaAsset, MediaType, Orientation

logger = logging.getLogger(__name__)

_BASE = "https://api.pexels.com"
_IMG_SEARCH = f"{_BASE}/v1/search"
_VID_SEARCH = f"{_BASE}/videos/search"
_PEXELS_ORIENTATION = {
    "landscape": "landscape",
    "portrait": "portrait",
    "square": "square",
}


class PexelsProvider:
    """Pexels client returning unified MediaAsset list."""

    name = "pexels"

    def __init__(self, api_key: str, timeout_s: float = 15.0) -> None:
        if not api_key:
            raise ValueError("Pexels API key required")
        self._api_key = api_key
        self._timeout_s = timeout_s

    async def search(
        self,
        keywords: list[str],
        orientation: Orientation,
        media_type: MediaType,
        limit: int = 10,
    ) -> list[MediaAsset]:
        if not keywords:
            return []
        query = " ".join(keywords[:3])  # Pexels prefers concise queries

        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            headers={"Authorization": self._api_key},
        ) as client:
            if media_type == "video":
                return await self._search_videos(client, query, orientation, limit)
            return await self._search_photos(client, query, orientation, limit)

    async def search_both(
        self,
        keywords: list[str],
        orientation: Orientation,
        limit_each: int = 8,
    ) -> list[MediaAsset]:
        """Convenience: fetch images and videos in parallel."""
        if not keywords:
            return []
        videos, photos = await asyncio.gather(
            self.search(keywords, orientation, "video", limit_each),
            self.search(keywords, orientation, "image", limit_each),
            return_exceptions=True,
        )
        out: list[MediaAsset] = []
        for batch in (videos, photos):
            if isinstance(batch, Exception):
                logger.warning("Pexels partial failure: %s", batch)
                continue
            out.extend(batch)
        return out

    # ---- Internal -----------------------------------------------------------

    async def _search_photos(
        self,
        client: httpx.AsyncClient,
        query: str,
        orientation: Orientation,
        limit: int,
    ) -> list[MediaAsset]:
        params = {
            "query": query,
            "orientation": _PEXELS_ORIENTATION[orientation],
            "per_page": min(limit, 80),
        }
        resp = await client.get(_IMG_SEARCH, params=params)
        resp.raise_for_status()
        return [self._photo_to_asset(p) for p in resp.json().get("photos", [])]

    async def _search_videos(
        self,
        client: httpx.AsyncClient,
        query: str,
        orientation: Orientation,
        limit: int,
    ) -> list[MediaAsset]:
        params = {
            "query": query,
            "orientation": _PEXELS_ORIENTATION[orientation],
            "per_page": min(limit, 80),
        }
        resp = await client.get(_VID_SEARCH, params=params)
        resp.raise_for_status()
        return [self._video_to_asset(v) for v in resp.json().get("videos", [])]

    def _photo_to_asset(self, photo: dict[str, Any]) -> MediaAsset:
        src = photo.get("src", {})
        url = src.get("large2x") or src.get("large") or src.get("original") or ""
        return MediaAsset(
            provider=self.name,
            media_type="image",
            url=url,
            width=int(photo.get("width", 0)),
            height=int(photo.get("height", 0)),
            license="pexels",
            tags=tuple(self._tags_from_photo(photo)),
        )

    def _video_to_asset(self, video: dict[str, Any]) -> MediaAsset:
        # Pick the best mp4 file: prefer hd/sd, ≥720p, mp4 type.
        files: list[dict[str, Any]] = video.get("video_files", [])
        mp4s = [f for f in files if f.get("file_type") == "video/mp4"]
        mp4s.sort(
            key=lambda f: (f.get("height") or 0) if f.get("height") else 0,
            reverse=True,
        )
        best = mp4s[0] if mp4s else (files[0] if files else {})
        url = best.get("link", "")
        return MediaAsset(
            provider=self.name,
            media_type="video",
            url=url,
            width=int(best.get("width") or video.get("width", 0)),
            height=int(best.get("height") or video.get("height", 0)),
            duration_s=float(video.get("duration", 0)) or None,
            license="pexels",
            tags=tuple(self._tags_from_video(video)),
        )

    @staticmethod
    def _tags_from_photo(photo: dict[str, Any]) -> list[str]:
        # Pexels photos don't expose tags; derive from alt text + photographer.
        alt = (photo.get("alt") or "").lower()
        return [w for w in alt.split() if len(w) > 2]

    @staticmethod
    def _tags_from_video(video: dict[str, Any]) -> list[str]:
        tags = video.get("tags") or []
        if isinstance(tags, list):
            return [str(t).lower() for t in tags]
        return []
