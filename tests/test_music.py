"""Tests for music stage: mood mapping, local library, orchestrator wiring."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from app.pipeline.models import Segment

# ============================================================================
# Mood mapping
# ============================================================================


def test_normalize_mood_known() -> None:
    from app.pipeline.music.mood import normalize_mood

    assert normalize_mood("calm") == "calm"
    assert normalize_mood("ENERGETIC") == "energetic"
    assert normalize_mood("Sad") == "sad"


def test_normalize_mood_synonyms() -> None:
    from app.pipeline.music.mood import normalize_mood

    assert normalize_mood("ambient") == "calm"
    assert normalize_mood("upbeat") == "energetic"
    assert normalize_mood("melancholic") == "sad"
    assert normalize_mood("happy") == "positive"


def test_normalize_mood_unknown_falls_back_calm() -> None:
    from app.pipeline.music.mood import normalize_mood

    assert normalize_mood("") == "calm"
    assert normalize_mood(None) == "calm"
    assert normalize_mood("xyz") == "calm"


def test_tags_for_returns_primary_then_fallbacks_dedup() -> None:
    from app.pipeline.music.mood import tags_for

    out = tags_for("calm")
    assert out[0] == "calm"
    # No duplicates.
    assert len(out) == len(set(out))
    # Includes synonyms across fallbacks (e.g. uplifting from energetic/positive).
    assert "uplifting" in out


# ============================================================================
# Local BGM library
# ============================================================================


def _make_track(root: Path, mood_dir: str, name: str) -> Path:
    target = root / mood_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"FAKE-MP3-BYTES")
    return target


@pytest.mark.asyncio
async def test_library_finds_track_for_canonical_mood(tmp_path: Path) -> None:
    from app.pipeline.music.library import LocalMusicLibrary

    track_path = _make_track(tmp_path, "calm", "ambient_piano.mp3")

    lib = LocalMusicLibrary(tmp_path, rng=random.Random(0))
    track = await lib.find(mood="calm")

    assert track is not None
    assert track.local_path == track_path
    assert track.mood == "calm"
    assert track.title == "ambient_piano"


@pytest.mark.asyncio
async def test_library_finds_via_synonym_directory(tmp_path: Path) -> None:
    from app.pipeline.music.library import LocalMusicLibrary

    # User stored under "ambient" instead of "calm" — should still match calm.
    _make_track(tmp_path, "ambient", "rain.mp3")

    lib = LocalMusicLibrary(tmp_path)
    track = await lib.find(mood="calm")

    assert track is not None
    assert track.local_path.name == "rain.mp3"
    assert track.mood == "calm"


@pytest.mark.asyncio
async def test_library_falls_back_to_other_mood(tmp_path: Path) -> None:
    """If no calm track exists, library should try fallback chain (e.g. positive)."""
    from app.pipeline.music.library import LocalMusicLibrary

    _make_track(tmp_path, "positive", "happy.mp3")

    lib = LocalMusicLibrary(tmp_path)
    track = await lib.find(mood="calm")

    assert track is not None
    assert track.local_path.name == "happy.mp3"


@pytest.mark.asyncio
async def test_library_returns_none_when_empty(tmp_path: Path) -> None:
    from app.pipeline.music.library import LocalMusicLibrary

    lib = LocalMusicLibrary(tmp_path)
    track = await lib.find(mood="calm")
    assert track is None


@pytest.mark.asyncio
async def test_library_returns_none_when_root_missing(tmp_path: Path) -> None:
    from app.pipeline.music.library import LocalMusicLibrary

    missing = tmp_path / "does_not_exist"
    lib = LocalMusicLibrary(missing)
    track = await lib.find(mood="calm")
    assert track is None


@pytest.mark.asyncio
async def test_library_skips_non_audio_files(tmp_path: Path) -> None:
    from app.pipeline.music.library import LocalMusicLibrary

    (tmp_path / "calm").mkdir(parents=True)
    (tmp_path / "calm" / "notes.txt").write_text("not audio", encoding="utf-8")
    (tmp_path / "calm" / "track.mp3").write_bytes(b"FAKE")

    lib = LocalMusicLibrary(tmp_path)
    track = await lib.find(mood="calm")

    assert track is not None
    assert track.local_path.suffix == ".mp3"


# ============================================================================
# Orchestrator music stage
# ============================================================================


class _StubNLP:
    def __init__(self, mood: str = "calm") -> None:
        self._mood = mood

    async def segment(self, article: str, target_seconds_per_seg: int = 8):
        return [Segment(index=0, text=article, keywords=("scene",))]

    async def keywords(self, segment: Segment, top_k: int = 5):
        return list(segment.keywords)

    async def translate(self, text: str, source_lang: str, target_lang: str):
        return text

    async def detect_mood(self, article: str):
        return self._mood


@pytest.mark.asyncio
async def test_orchestrator_picks_bgm_matching_mood(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    """Music stage selects a track matching detected mood and passes path to compose."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.music.library import LocalMusicLibrary
    from app.pipeline.orchestrator import PipelineOrchestrator

    bgm_root = tmp_path / "bgm"
    track_path = _make_track(bgm_root, "energetic", "drums.mp3")

    job = Job(article="Action article.", nlp_backend="local")
    orchestrator = PipelineOrchestrator(
        nlp=_StubNLP(mood="energetic"),
        media_providers=[],
        tts=stub_tts,
        music_providers=[LocalMusicLibrary(bgm_root, rng=random.Random(0))],
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    )
    result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert orchestrator.bgm_track is not None
    assert orchestrator.bgm_track.local_path == track_path

    # compose_fn called with bgm_path arg (positional index 7).
    args, _ = stub_compose.call_args
    assert args[7] == track_path
    # bgm_gain (index 8) defaults to 0.10.
    assert args[8] == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_orchestrator_skips_bgm_when_disabled(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    """If job.bgm_enabled=False, music stage doesn't touch providers."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.orchestrator import PipelineOrchestrator

    class FailingProvider:
        name = "failing"

        async def find(self, mood: str, min_duration_s=None):
            raise AssertionError("should not be called when bgm disabled")

    job = Job(article="Quiet text.", nlp_backend="local", bgm_enabled=False)
    orchestrator = PipelineOrchestrator(
        nlp=_StubNLP(),
        media_providers=[],
        tts=stub_tts,
        music_providers=[FailingProvider()],
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    )
    result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert orchestrator.bgm_track is None
    args, _ = stub_compose.call_args
    assert args[7] is None  # bgm_path


@pytest.mark.asyncio
async def test_orchestrator_no_providers_continues_without_bgm(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.orchestrator import PipelineOrchestrator

    job = Job(article="No music available.", nlp_backend="local")
    orchestrator = PipelineOrchestrator(
        nlp=_StubNLP(),
        media_providers=[],
        tts=stub_tts,
        music_providers=[],
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    )
    result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert orchestrator.bgm_track is None


@pytest.mark.asyncio
async def test_orchestrator_provider_error_falls_through(
    tmp_path: Path, stub_tts, stub_compose, stub_render
) -> None:
    """Provider raises -> orchestrator logs and tries next provider."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.music.library import LocalMusicLibrary
    from app.pipeline.orchestrator import PipelineOrchestrator

    bgm_root = tmp_path / "bgm"
    _make_track(bgm_root, "calm", "fallback.mp3")

    class BoomProvider:
        name = "boom"

        async def find(self, mood: str, min_duration_s=None):
            raise RuntimeError("network down")

    orchestrator = PipelineOrchestrator(
        nlp=_StubNLP(mood="calm"),
        media_providers=[],
        tts=stub_tts,
        music_providers=[BoomProvider(), LocalMusicLibrary(bgm_root)],
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    )
    result = await orchestrator.run(Job(article="Fall through.", nlp_backend="local"))

    assert result.status is JobStatus.DONE
    assert orchestrator.bgm_track is not None
    assert orchestrator.bgm_track.local_path.name == "fallback.mp3"
