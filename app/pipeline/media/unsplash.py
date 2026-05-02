"""Unsplash stock-image provider.

Unsplash API: https://unsplash.com/documentation
Auth: `Authorization: Client-ID <access_key>` header.
Free tier: 50 req/hour (demo apps), 5000/hour (approved production apps).
Unsplash has no video — `media_type="video"` returns []. Image-only.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.pipeline.models import MediaAsset, MediaType, Orientation

logger = logging.getLogger(__name__)

_BASE = "https://api.unsplash.com"
_SEARCH = f"{_BASE}/search/photos"
_UNSPLASH_ORIENTATION = {
    "landscape": "landscape",
    "portrait": "portrait",
    "square": "squarish",
}


class UnsplashProvider:
    """Unsplash client returning unified MediaAsset list. Images only."""

    name = "unsplash"

    def __init__(self, access_key: str, timeout_s: float = 15.0) -> None:
        if not access_key:
            raise ValueError("Unsplash access key required")
        self._access_key = access_key
        self._timeout_s = timeout_s

    async def search(
        self,
        keywords: list[str],
        orientation: Orientation,
        media_type: MediaType,
        limit: int = 10,
    ) -> list[MediaAsset]:
        if not keywords or media_type == "video":
            return []

        query = " ".join(keywords[:3])
        params = {
            "query": query,
            "orientation": _UNSPLASH_ORIENTATION[orientation],
            "per_page": min(limit, 30),
            "content_filter": "high",
        }
        headers = {"Authorization": f"Client-ID {self._access_key}"}

        async with httpx.AsyncClient(
            timeout=self._timeout_s, headers=headers
        ) as client:
            resp = await client.get(_SEARCH, params=params)
            resp.raise_for_status()
            return [self._to_asset(p) for p in resp.json().get("results", [])]

    async def search_both(
        self,
        keywords: list[str],
        orientation: Orientation,
        limit_each: int = 8,
    ) -> list[MediaAsset]:
        """Image-only provider; videos return []. Same shape as other providers."""
        return await self.search(keywords, orientation, "image", limit_each)

    # ---- Internal -----------------------------------------------------------

    def _to_asset(self, photo: dict[str, Any]) -> MediaAsset:
        urls = photo.get("urls", {})
        url = urls.get("full") or urls.get("regular") or urls.get("raw") or ""
        tags_block = photo.get("tags") or []
        tag_titles = [
            (t.get("title") or "").lower() for t in tags_block if isinstance(t, dict)
        ]
        # Unsplash also offers alt_description.
        alt = (photo.get("alt_description") or "").lower()
        derived = [w for w in alt.split() if len(w) > 2]
        tags = tag_titles + derived

        return MediaAsset(
            provider=self.name,
            media_type="image",
            url=url,
            width=int(photo.get("width", 0)),
            height=int(photo.get("height", 0)),
            license="unsplash",
            tags=tuple(t for t in tags if t),
        )
