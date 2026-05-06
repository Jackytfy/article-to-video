"""Media stage: stock providers + factory + cache + ranker."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.pipeline.media.base import MediaProvider


def make_providers(settings: Settings | None = None) -> list[MediaProvider]:
    """Return all configured providers (those with API keys set).

    Order matters for ranking ties: earlier = preferred.
    Web-recorder is appended last so stock providers take priority,
    but it's always available (no API key needed).
    """
    settings = settings or get_settings()
    providers: list[MediaProvider] = []

    if settings.pexels_api_key:
        from app.pipeline.media.pexels import PexelsProvider

        providers.append(PexelsProvider(api_key=settings.pexels_api_key))

    if settings.pixabay_api_key:
        from app.pipeline.media.pixabay import PixabayProvider

        providers.append(PixabayProvider(api_key=settings.pixabay_api_key))

    if settings.unsplash_access_key:
        from app.pipeline.media.unsplash import UnsplashProvider

        providers.append(UnsplashProvider(access_key=settings.unsplash_access_key))

    if settings.web_recorder_enabled:
        from app.pipeline.media.web_recorder import WebRecorderProvider

        providers.append(WebRecorderProvider(
            max_duration_s=settings.web_recorder_max_duration,
            mode=settings.web_recorder_mode,
        ))

    return providers


__all__ = ["make_providers"]
