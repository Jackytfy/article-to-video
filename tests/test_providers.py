"""Tests for new providers (Phase 7): Pixabay, Unsplash, Jamendo."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

# ============================================================================
# Pixabay
# ============================================================================


@pytest.mark.asyncio
async def test_pixabay_image_search_parses_response() -> None:
    from app.pipeline.media.pixabay import PixabayProvider

    payload = {
        "hits": [
            {
                "imageWidth": 1920,
                "imageHeight": 1080,
                "tags": "sunset, mountain, landscape",
                "largeImageURL": "https://cdn.pixabay.com/photo/1.jpg",
            }
        ]
    }
    with respx.mock(assert_all_called=True) as router:
        router.get("https://pixabay.com/api/").mock(
            return_value=httpx.Response(200, json=payload)
        )
        provider = PixabayProvider(api_key="test-key")
        assets = await provider.search(
            ["sunset", "mountain"], "landscape", "image", limit=5
        )

    assert len(assets) == 1
    asset = assets[0]
    assert asset.url == "https://cdn.pixabay.com/photo/1.jpg"
    assert asset.media_type == "image"
    assert "sunset" in asset.tags
    assert asset.width == 1920


@pytest.mark.asyncio
async def test_pixabay_video_search_picks_best_size() -> None:
    from app.pipeline.media.pixabay import PixabayProvider

    payload = {
        "hits": [
            {
                "duration": 12,
                "tags": "city, traffic",
                "imageWidth": 1920,
                "imageHeight": 1080,
                "videos": {
                    "large": {
                        "url": "https://cdn.pixabay.com/large.mp4",
                        "width": 1920,
                        "height": 1080,
                    },
                    "medium": {
                        "url": "https://cdn.pixabay.com/medium.mp4",
                        "width": 1280,
                        "height": 720,
                    },
                },
            }
        ]
    }
    with respx.mock(assert_all_called=True) as router:
        router.get("https://pixabay.com/api/videos/").mock(
            return_value=httpx.Response(200, json=payload)
        )
        provider = PixabayProvider(api_key="test-key")
        assets = await provider.search(
            ["city"], "landscape", "video", limit=5
        )

    assert len(assets) == 1
    assert assets[0].url == "https://cdn.pixabay.com/large.mp4"
    assert assets[0].duration_s == 12
    assert assets[0].media_type == "video"


@pytest.mark.asyncio
async def test_pixabay_empty_keywords_returns_empty() -> None:
    from app.pipeline.media.pixabay import PixabayProvider

    provider = PixabayProvider(api_key="test-key")
    assert await provider.search([], "landscape", "image") == []


def test_pixabay_requires_api_key() -> None:
    from app.pipeline.media.pixabay import PixabayProvider

    with pytest.raises(ValueError):
        PixabayProvider(api_key="")


# ============================================================================
# Unsplash
# ============================================================================


@pytest.mark.asyncio
async def test_unsplash_search_parses_response() -> None:
    from app.pipeline.media.unsplash import UnsplashProvider

    payload = {
        "results": [
            {
                "width": 4000,
                "height": 2667,
                "alt_description": "sunset over mountains",
                "urls": {
                    "full": "https://images.unsplash.com/photo-1.jpg",
                    "regular": "https://images.unsplash.com/photo-1-small.jpg",
                },
                "tags": [
                    {"title": "Sunset"},
                    {"title": "Mountain"},
                ],
            }
        ]
    }
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.unsplash.com/search/photos").mock(
            return_value=httpx.Response(200, json=payload)
        )

        provider = UnsplashProvider(access_key="abcd")
        assets = await provider.search(
            ["sunset"], "landscape", "image", limit=5
        )

    assert len(assets) == 1
    asset = assets[0]
    assert asset.url == "https://images.unsplash.com/photo-1.jpg"
    assert asset.media_type == "image"
    assert "sunset" in asset.tags
    assert asset.license == "unsplash"


@pytest.mark.asyncio
async def test_unsplash_video_returns_empty() -> None:
    """Unsplash has no video; provider must short-circuit cleanly."""
    from app.pipeline.media.unsplash import UnsplashProvider

    provider = UnsplashProvider(access_key="abcd")
    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://api.unsplash.com/search/photos")
        result = await provider.search(["sunset"], "landscape", "video")
        assert result == []
        assert route.call_count == 0


def test_unsplash_requires_access_key() -> None:
    from app.pipeline.media.unsplash import UnsplashProvider

    with pytest.raises(ValueError):
        UnsplashProvider(access_key="")


# ============================================================================
# Jamendo
# ============================================================================


@pytest.mark.asyncio
async def test_jamendo_finds_track_for_mood(tmp_path: Path) -> None:
    from app.pipeline.media.cache import MediaCache
    from app.pipeline.music.jamendo import JamendoProvider

    audio_url = "https://prod.jamendo.com/?trackid=1&format=mp3"
    payload = {
        "headers": {"status": "success"},
        "results": [
            {
                "id": 1,
                "name": "Calm Piano",
                "duration": 120,
                "audio": audio_url,
                "audiodownload": audio_url,
                "license_ccurl": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.jamendo.com/v3.0/tracks/").mock(
            return_value=httpx.Response(200, json=payload)
        )
        # 64 KB of fake MP3 bytes — over the _MIN_AUDIO_BYTES threshold.
        router.get(audio_url).mock(
            return_value=httpx.Response(
                200,
                content=b"\xff\xfb" + b"\x00" * (64 * 1024),
                headers={"content-type": "audio/mpeg"},
            )
        )

        cache = MediaCache(tmp_path)
        provider = JamendoProvider(client_id="test-id", cache=cache)
        track = await provider.find(mood="calm", min_duration_s=60)

    assert track is not None
    assert track.title == "Calm Piano"
    assert track.duration_s == 120
    assert track.local_path.exists()
    assert track.local_path.suffix == ".mp3"
    assert track.mood == "calm"


@pytest.mark.asyncio
async def test_jamendo_returns_none_when_no_results(tmp_path: Path) -> None:
    from app.pipeline.media.cache import MediaCache
    from app.pipeline.music.jamendo import JamendoProvider

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.jamendo.com/v3.0/tracks/").mock(
            return_value=httpx.Response(
                200, json={"headers": {"status": "success"}, "results": []}
            )
        )
        provider = JamendoProvider(
            client_id="test-id", cache=MediaCache(tmp_path)
        )
        track = await provider.find(mood="calm")

    assert track is None


@pytest.mark.asyncio
async def test_jamendo_skips_track_without_audio_url(tmp_path: Path) -> None:
    from app.pipeline.media.cache import MediaCache
    from app.pipeline.music.jamendo import JamendoProvider

    payload = {
        "headers": {"status": "success"},
        "results": [
            {"id": 1, "name": "Ghost Track"},  # no audio/audiodownload
        ],
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.jamendo.com/v3.0/tracks/").mock(
            return_value=httpx.Response(200, json=payload)
        )
        provider = JamendoProvider(
            client_id="test-id", cache=MediaCache(tmp_path)
        )
        track = await provider.find(mood="calm")

    assert track is None


def test_jamendo_requires_client_id(tmp_path: Path) -> None:
    from app.pipeline.media.cache import MediaCache
    from app.pipeline.music.jamendo import JamendoProvider

    with pytest.raises(ValueError):
        JamendoProvider(client_id="", cache=MediaCache(tmp_path))


# ============================================================================
# Factories
# ============================================================================


def test_media_factory_includes_all_when_keys_set() -> None:
    from app.config import Settings
    from app.pipeline.media import make_providers

    settings = Settings(
        pexels_api_key="a",
        pixabay_api_key="b",
        unsplash_access_key="c",
    )
    providers = make_providers(settings=settings)
    names = [p.name for p in providers]
    assert names == ["pexels", "pixabay", "unsplash"]


def test_media_factory_skips_missing_keys() -> None:
    from app.config import Settings
    from app.pipeline.media import make_providers

    settings = Settings(
        pexels_api_key=None,
        pixabay_api_key="b",
        unsplash_access_key=None,
    )
    providers = make_providers(settings=settings)
    names = [p.name for p in providers]
    assert names == ["pixabay"]


@pytest.mark.real_music_factory
def test_music_factory_includes_jamendo_when_id_set(tmp_path: Path) -> None:
    from app.config import Settings
    from app.pipeline.music import make_providers

    settings = Settings(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        jamendo_client_id="abc",
    )
    # No bgm dir on disk -> only Jamendo should appear.
    providers = make_providers(settings=settings)
    names = [p.name for p in providers]
    assert names == ["jamendo"]
