"""Pixabay stock media provider.

Pixabay Image API: https://pixabay.com/api/
Pixabay Video API: https://pixabay.com/api/videos/

Free tier: 100 req/min, attribution not required (Pixabay license).
Auth: API key as `key` query parameter.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.pipeline.models import MediaAsset, MediaType, Orientation

logger = logging.getLogger(__name__)

_IMG_BASE = "https://pixabay.com/api/"
_VID_BASE = "https://pixabay.com/api/videos/"
_PIXABAY_ORIENTATION = {
    "landscape": "horizontal",
    "portrait": "vertical",
    "square": "all",  # Pixabay has no square; "all" returns all.
}


class PixabayProvider:
    """Pixabay client returning unified MediaAsset list."""

    name = "pixabay"

    def __init__(self, api_key: str, timeout_s: float = 15.0) -> None:
        if not api_key:
            raise ValueError("Pixabay API key required")
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

        query = "+".join(keywords[:3])  # Pixabay uses + for AND
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            if media_type == "video":
                return await self._search_videos(client, query, orientation, limit)
            return await self._search_images(client, query, orientation, limit)

    async def search_both(
        self,
        keywords: list[str],
        orientation: Orientation,
        limit_each: int = 8,
    ) -> list[MediaAsset]:
        """Convenience: fetch images and videos in parallel."""
        if not keywords:
            return []
        videos, images = await asyncio.gather(
            self.search(keywords, orientation, "video", limit_each),
            self.search(keywords, orientation, "image", limit_each),
            return_exceptions=True,
        )
        out: list[MediaAsset] = []
        for batch in (videos, images):
            if isinstance(batch, Exception):
                logger.warning("Pixabay partial failure: %s", batch)
                continue
            out.extend(batch)
        return out

    # ---- Internal -----------------------------------------------------------

    async def _search_images(
        self,
        client: httpx.AsyncClient,
        query: str,
        orientation: Orientation,
        limit: int,
    ) -> list[MediaAsset]:
        params = {
            "key": self._api_key,
            "q": query,
            "orientation": _PIXABAY_ORIENTATION[orientation],
            "image_type": "photo",
            "per_page": max(3, min(limit, 200)),
            "safesearch": "true",
        }
        resp = await client.get(_IMG_BASE, params=params)
        resp.raise_for_status()
        return [self._image_to_asset(item) for item in resp.json().get("hits", [])]

    async def _search_videos(
        self,
        client: httpx.AsyncClient,
        query: str,
        orientation: Orientation,
        limit: int,
    ) -> list[MediaAsset]:
        params = {
            "key": self._api_key,
            "q": query,
            "video_type": "all",
            "per_page": max(3, min(limit, 200)),
            "safesearch": "true",
        }
        # Pixabay video API does not support orientation. We post-filter.
        resp = await client.get(_VID_BASE, params=params)
        resp.raise_for_status()
        items = resp.json().get("hits", [])
        assets = [self._video_to_asset(it) for it in items]
        return [a for a in assets if a.url and self._matches_orientation(a, orientation)]

    @staticmethod
    def _matches_orientation(asset: MediaAsset, orientation: Orientation) -> bool:
        return orientation == "square" or asset.orientation in (orientation, "square")

    def _image_to_asset(self, hit: dict[str, Any]) -> MediaAsset:
        url = hit.get("largeImageURL") or hit.get("webformatURL") or ""
        return MediaAsset(
            provider=self.name,
            media_type="image",
            url=url,
            width=int(hit.get("imageWidth", 0)),
            height=int(hit.get("imageHeight", 0)),
            license="pixabay",
            tags=tuple(self._split_tags(hit.get("tags", ""))),
        )

    def _video_to_asset(self, hit: dict[str, Any]) -> MediaAsset:
        # Pixabay returns videos in multiple sizes: large, medium, small, tiny.
        videos: dict[str, Any] = hit.get("videos", {})
        chosen: dict[str, Any] = {}
        for size in ("large", "medium", "small", "tiny"):
            block = videos.get(size, {})
            if block.get("url"):
                chosen = block
                break
        url = chosen.get("url", "")
        return MediaAsset(
            provider=self.name,
            media_type="video",
            url=url,
            width=int(chosen.get("width") or hit.get("imageWidth", 0)),
            height=int(chosen.get("height") or hit.get("imageHeight", 0)),
            duration_s=float(hit.get("duration", 0)) or None,
            license="pixabay",
            tags=tuple(self._split_tags(hit.get("tags", ""))),
        )

    @staticmethod
    def _split_tags(tags: str) -> list[str]:
        return [t.strip().lower() for t in tags.split(",") if t.strip()]
