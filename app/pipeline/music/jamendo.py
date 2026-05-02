"""Jamendo music provider.

Jamendo API: https://developer.jamendo.com/v3.0
Free dev tier: requires `client_id`. Tracks are CC-licensed; check per-track
license metadata before commercial use.

Notes on Jamendo quirks:
- `audiodlformat` accepts mp31 | mp32 | ogg | flac (NOT bare "mp3"). mp32 = ~192kbps MP3.
- The CDN at `prod-N.storage.jamendo.com` is flaky and frequently returns 500 or
  redirects to an HTML error page. We try the streaming `audio` URL first and fall
  back to `audiodownload`. After download we verify the content-type is audio/* —
  otherwise we drop the file and try the next track.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.pipeline.media.cache import MediaCache
from app.pipeline.music.base import MusicProvider, MusicTrack
from app.pipeline.music.mood import normalize_mood, tags_for

logger = logging.getLogger(__name__)

_BASE = "https://api.jamendo.com/v3.0"
_TRACKS = f"{_BASE}/tracks/"

# A real audio file should be at least this large; smaller = error-page HTML.
_MIN_AUDIO_BYTES = 32 * 1024  # 32 KB
_AUDIO_CT_PREFIXES = ("audio/", "application/ogg")


class JamendoProvider(MusicProvider):
    """Jamendo music client downloading and caching the chosen track."""

    name = "jamendo"

    def __init__(
        self,
        client_id: str,
        cache: MediaCache,
        timeout_s: float = 20.0,
    ) -> None:
        if not client_id:
            raise ValueError("Jamendo client_id required")
        self._client_id = client_id
        self._cache = cache
        self._timeout_s = timeout_s

    async def find(
        self,
        mood: str,
        min_duration_s: float | None = None,
    ) -> MusicTrack | None:
        canonical = normalize_mood(mood)

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            for tag in tags_for(canonical):
                track = await self._search_tag(client, canonical, tag, min_duration_s)
                if track is not None:
                    return track
        return None

    # ---- Internal -----------------------------------------------------------

    async def _search_tag(
        self,
        client: httpx.AsyncClient,
        canonical_mood: str,
        tag: str,
        min_duration_s: float | None,
    ) -> MusicTrack | None:
        params = {
            "client_id": self._client_id,
            "format": "json",
            "limit": 10,
            "tags": tag,
            "audiodlformat": "mp32",
            "include": "musicinfo licenses",
            "order": "popularity_total",
        }
        if min_duration_s is not None:
            params["durationbetween"] = f"{int(min_duration_s)}_3600"

        try:
            resp = await client.get(_TRACKS, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Jamendo search failed (tag=%s): %s", tag, exc)
            return None

        body = resp.json()
        if body.get("headers", {}).get("status") != "success":
            logger.warning(
                "Jamendo API non-success (tag=%s): %s",
                tag,
                body.get("headers", {}).get("error_message"),
            )
            return None

        results: list[dict[str, Any]] = body.get("results", []) or []
        for hit in results:
            track = await self._try_download(client, hit, canonical_mood)
            if track is not None:
                return track
        return None

    async def _try_download(
        self,
        client: httpx.AsyncClient,
        hit: dict[str, Any],
        canonical_mood: str,
    ) -> MusicTrack | None:
        # Order: streaming `audio` URL first (more reliable), then dl URL.
        candidates = [
            hit.get("audio"),
            hit.get("audiodownload"),
        ]
        for url in candidates:
            if not url:
                continue
            local_path = await self._download_audio(client, url)
            if local_path is None:
                continue
            return MusicTrack(
                local_path=local_path,
                duration_s=float(hit.get("duration", 0)) or None,
                mood=canonical_mood,
                title=str(hit.get("name") or local_path.stem),
                license=str(hit.get("license_ccurl") or "jamendo-cc"),
            )
        return None

    async def _download_audio(
        self, client: httpx.AsyncClient, url: str
    ) -> "Any | None":
        """Download + validate it's actually audio. Returns Path or None on bad data."""
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Jamendo audio fetch failed (%s): %s", url[:80], exc)
            return None

        content_type = (resp.headers.get("content-type") or "").lower()
        if not any(content_type.startswith(p) for p in _AUDIO_CT_PREFIXES):
            logger.warning(
                "Jamendo response is not audio (content-type=%s, url=%s)",
                content_type,
                url[:80],
            )
            return None
        if len(resp.content) < _MIN_AUDIO_BYTES:
            logger.warning(
                "Jamendo audio too small (%d bytes); likely error page",
                len(resp.content),
            )
            return None

        # Build filename directly: cache.path_for can pick up junk suffixes
        # like ".com" from query-string URLs. Force a sane audio extension
        # based on content-type.
        import hashlib

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        ext = ".mp3"
        if "ogg" in content_type:
            ext = ".ogg"
        elif "flac" in content_type:
            ext = ".flac"
        elif "wav" in content_type:
            ext = ".wav"

        target = self._cache._root / f"{digest}{ext}"  # noqa: SLF001
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
        return target
