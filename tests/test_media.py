"""Tests for media stage: ranker, cache, Pexels client (mocked), orchestrator."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from app.pipeline.models import MediaAsset, Segment

# ============================================================================
# Ranker
# ============================================================================


def test_aspect_to_orientation_maps_correctly() -> None:
    from app.pipeline.media.ranker import aspect_to_orientation

    assert aspect_to_orientation("16:9") == "landscape"
    assert aspect_to_orientation("9:16") == "portrait"
    assert aspect_to_orientation("1:1") == "square"


def test_aspect_to_orientation_rejects_unknown() -> None:
    from app.pipeline.media.ranker import aspect_to_orientation

    with pytest.raises(ValueError):
        aspect_to_orientation("4:3")


def test_score_asset_prefers_keyword_overlap() -> None:
    from app.pipeline.media.ranker import score_asset

    matching = MediaAsset(
        provider="pexels",
        media_type="image",
        url="x",
        width=1920,
        height=1080,
        tags=("sunset", "mountain"),
    )
    irrelevant = MediaAsset(
        provider="pexels",
        media_type="image",
        url="y",
        width=1920,
        height=1080,
        tags=("car", "city"),
    )
    keywords = ["sunset", "mountain"]
    assert score_asset(matching, keywords, "landscape") > score_asset(
        irrelevant, keywords, "landscape"
    )


def test_score_asset_prefers_orientation_match() -> None:
    from app.pipeline.media.ranker import score_asset

    landscape = MediaAsset(
        provider="pexels",
        media_type="image",
        url="x",
        width=1920,
        height=1080,
    )
    portrait = MediaAsset(
        provider="pexels",
        media_type="image",
        url="y",
        width=1080,
        height=1920,
    )
    assert score_asset(landscape, [], "landscape") > score_asset(
        portrait, [], "landscape"
    )


def test_rank_assets_returns_best_first() -> None:
    from app.pipeline.media.ranker import rank_assets

    perfect = MediaAsset(
        provider="pexels",
        media_type="video",
        url="best",
        width=1920,
        height=1080,
        tags=("sunset", "mountain"),
    )
    okay = MediaAsset(
        provider="pexels",
        media_type="image",
        url="ok",
        width=640,
        height=360,
        tags=("sunset",),
    )
    bad = MediaAsset(
        provider="pexels",
        media_type="image",
        url="bad",
        width=400,
        height=300,
        tags=("car",),
    )
    ranked = rank_assets([bad, okay, perfect], ["sunset", "mountain"], "landscape")
    assert ranked[0].url == "best"
    assert ranked[-1].url == "bad"


# ============================================================================
# Cache
# ============================================================================


@pytest.mark.asyncio
async def test_cache_downloads_and_dedupes(tmp_path: Path) -> None:
    from app.pipeline.media.cache import MediaCache

    url = "https://images.example.com/photo.jpg"
    payload = b"fake-image-bytes"

    cache = MediaCache(tmp_path)
    with respx.mock(assert_all_called=True) as router:
        route = router.get(url).mock(
            return_value=httpx.Response(200, content=payload, headers={"content-type": "image/jpeg"})
        )

        path1 = await cache.fetch(url)
        assert path1.exists()
        assert path1.read_bytes() == payload
        assert route.call_count == 1

    # Second call should NOT hit the network.
    with respx.mock(assert_all_called=False) as router:
        path2 = await cache.fetch(url)
        assert path2 == path1


@pytest.mark.asyncio
async def test_cache_filename_is_deterministic(tmp_path: Path) -> None:
    from app.pipeline.media.cache import MediaCache

    cache = MediaCache(tmp_path)
    a = cache.path_for("https://x.com/y.jpg")
    b = cache.path_for("https://x.com/y.jpg")
    c = cache.path_for("https://x.com/z.jpg")
    assert a == b
    assert a != c
    assert a.suffix == ".jpg"


# ============================================================================
# Pexels client (mocked HTTP)
# ============================================================================


@pytest.mark.asyncio
async def test_pexels_search_photos_parses_response() -> None:
    from app.pipeline.media.pexels import PexelsProvider

    payload = {
        "photos": [
            {
                "id": 1,
                "width": 1920,
                "height": 1080,
                "alt": "sunset over mountains",
                "src": {"large2x": "https://cdn.pexels.com/photos/1.jpg"},
            },
            {
                "id": 2,
                "width": 1080,
                "height": 1920,
                "alt": "city skyline",
                "src": {"large2x": "https://cdn.pexels.com/photos/2.jpg"},
            },
        ]
    }
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.pexels.com/v1/search").mock(
            return_value=httpx.Response(200, json=payload)
        )

        provider = PexelsProvider(api_key="test-key")
        assets = await provider.search(
            keywords=["sunset", "mountain"],
            orientation="landscape",
            media_type="image",
            limit=5,
        )

    assert len(assets) == 2
    assert assets[0].url.endswith("/1.jpg")
    assert assets[0].width == 1920
    assert "sunset" in assets[0].tags or "mountains" in assets[0].tags


@pytest.mark.asyncio
async def test_pexels_search_videos_picks_highest_quality_mp4() -> None:
    from app.pipeline.media.pexels import PexelsProvider

    payload = {
        "videos": [
            {
                "id": 1,
                "width": 1920,
                "height": 1080,
                "duration": 15,
                "tags": ["sunset", "mountain"],
                "video_files": [
                    {
                        "file_type": "video/mp4",
                        "width": 640,
                        "height": 360,
                        "link": "https://cdn.pexels.com/videos/1-360.mp4",
                    },
                    {
                        "file_type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "link": "https://cdn.pexels.com/videos/1-1080.mp4",
                    },
                ],
            }
        ]
    }
    with respx.mock(assert_all_called=True) as router:
        router.get("https://api.pexels.com/videos/search").mock(
            return_value=httpx.Response(200, json=payload)
        )

        provider = PexelsProvider(api_key="test-key")
        assets = await provider.search(
            keywords=["sunset"],
            orientation="landscape",
            media_type="video",
            limit=5,
        )

    assert len(assets) == 1
    asset = assets[0]
    assert asset.media_type == "video"
    assert asset.url.endswith("1-1080.mp4")
    assert asset.duration_s == 15
    assert "sunset" in asset.tags


@pytest.mark.asyncio
async def test_pexels_empty_keywords_returns_empty() -> None:
    from app.pipeline.media.pexels import PexelsProvider

    provider = PexelsProvider(api_key="test-key")
    assets = await provider.search([], "landscape", "image")
    assert assets == []


def test_pexels_requires_api_key() -> None:
    from app.pipeline.media.pexels import PexelsProvider

    with pytest.raises(ValueError):
        PexelsProvider(api_key="")


# ============================================================================
# Orchestrator media stage (with mocked Pexels)
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_runs_full_nlp_plus_media(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    """End-to-end: local NLP + mocked Pexels + cache. Confirm assets attached."""
    from app.pipeline.media.cache import MediaCache
    from app.pipeline.media.pexels import PexelsProvider
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.nlp.local import LocalNLPBackend
    from app.pipeline.orchestrator import PipelineOrchestrator

    article = (
        "The sunrise glowed over the mountains. "
        "Gentle rivers flowed through the valley."
    )

    photo_payload = {
        "photos": [
            {
                "id": 1,
                "width": 1920,
                "height": 1080,
                "alt": "sunrise mountains",
                "src": {"large2x": "https://cdn.pexels.com/photos/sunrise.jpg"},
            }
        ]
    }
    video_payload = {"videos": []}

    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.pexels.com/v1/search").mock(
            return_value=httpx.Response(200, json=photo_payload)
        )
        router.get("https://api.pexels.com/videos/search").mock(
            return_value=httpx.Response(200, json=video_payload)
        )
        router.get("https://cdn.pexels.com/photos/sunrise.jpg").mock(
            return_value=httpx.Response(
                200, content=b"\x89PNG fake image", headers={"content-type": "image/jpeg"}
            )
        )

        job = Job(article=article, nlp_backend="local")
        orchestrator = PipelineOrchestrator(
            nlp=LocalNLPBackend(),
            media_providers=[PexelsProvider(api_key="test-key")],
            cache=MediaCache(tmp_path),
            tts=stub_tts,
            work_dir=tmp_path,
            compose_fn=stub_compose,
            render_fn=stub_render,
        )
        result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert len(result.segments) >= 1
    assert len(orchestrator.segment_assets) >= 1
    asset = next(iter(orchestrator.segment_assets.values()))
    assert asset.local_path is not None
    assert asset.local_path.exists()


@pytest.mark.asyncio
async def test_orchestrator_skips_media_when_no_providers(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    """Without API keys, media stage logs warning and continues without failing."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.nlp.local import LocalNLPBackend
    from app.pipeline.orchestrator import PipelineOrchestrator

    job = Job(article="Hello world from quiet hills.", nlp_backend="local")
    orchestrator = PipelineOrchestrator(
        nlp=LocalNLPBackend(),
        media_providers=[],
        tts=stub_tts,
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    )
    result = await orchestrator.run(job)
    assert result.status is JobStatus.DONE
    assert orchestrator.segment_assets == {}


@pytest.mark.asyncio
async def test_orchestrator_segment_without_keywords_skips_search(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    """A segment with no keywords yields no asset - no provider call."""
    from app.pipeline.media.pexels import PexelsProvider
    from app.pipeline.models import Job, JobStatus, Segment
    from app.pipeline.orchestrator import PipelineOrchestrator

    class StubNLP:
        async def segment(self, article: str, target_seconds_per_seg: int = 8):
            return [Segment(index=0, text=article, keywords=())]

        async def keywords(self, segment: Segment, top_k: int = 5):
            return []

        async def translate(self, text: str, source_lang: str, target_lang: str):
            return text

        async def detect_mood(self, article: str):
            return "calm"

    with respx.mock(assert_all_called=False) as router:
        photo = router.get("https://api.pexels.com/v1/search")
        video = router.get("https://api.pexels.com/videos/search")

        orchestrator = PipelineOrchestrator(
            nlp=StubNLP(),
            media_providers=[PexelsProvider(api_key="test-key")],
            tts=stub_tts,
            work_dir=tmp_path,
            compose_fn=stub_compose,
            render_fn=stub_render,
        )
        result = await orchestrator.run(Job(article="silent text"))

        assert result.status is JobStatus.DONE
        assert photo.call_count == 0
        assert video.call_count == 0
