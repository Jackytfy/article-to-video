"""Music stage: providers + factory + mood mapping."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.pipeline.music.base import MusicProvider


def make_providers(settings: Settings | None = None) -> list[MusicProvider]:
    """Return all configured BGM providers.

    Order: local library first (offline + zero-latency), then Jamendo if a
    client_id is set.
    """
    settings = settings or get_settings()
    providers: list[MusicProvider] = []

    from app.pipeline.music.library import LocalMusicLibrary

    bgm_root = settings.cache_dir.parent / "bgm"
    if bgm_root.exists():
        providers.append(LocalMusicLibrary(bgm_root))

    if settings.jamendo_client_id:
        from app.pipeline.media.cache import MediaCache
        from app.pipeline.music.jamendo import JamendoProvider

        cache = MediaCache(settings.cache_dir / "music")
        providers.append(
            JamendoProvider(
                client_id=settings.jamendo_client_id,
                cache=cache,
            )
        )

    return providers


__all__ = ["make_providers"]
