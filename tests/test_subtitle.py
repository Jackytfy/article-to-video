"""Tests for SRT subtitle generator + orchestrator subtitle stage."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.pipeline.models import Segment
from app.pipeline.subtitle.srt import (
    SubtitleCue,
    build_cues,
    group_words_for_segment,
    render_srt,
    write_srt,
)
from app.pipeline.tts.edge_tts import TTSResult, WordTiming

# ============================================================================
# SRT formatting
# ============================================================================


def test_render_srt_format() -> None:
    cues = [
        SubtitleCue(index=1, start_ms=0, end_ms=1500, text="Hello"),
        SubtitleCue(index=2, start_ms=1500, end_ms=3200, text="world"),
    ]
    out = render_srt(cues)
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello\n" in out
    assert "2\n00:00:01,500 --> 00:00:03,200\nworld\n" in out


def test_render_srt_handles_hour_boundary() -> None:
    cue = SubtitleCue(index=1, start_ms=3_661_000, end_ms=3_662_500, text="late")
    out = render_srt([cue])
    assert "01:01:01,000 --> 01:01:02,500" in out


def test_write_srt_creates_file(tmp_path: Path) -> None:
    cues = [SubtitleCue(index=1, start_ms=0, end_ms=1000, text="hi")]
    path = write_srt(cues, tmp_path / "captions.srt")
    assert path.exists()
    assert "00:00:00,000 --> 00:00:01,000" in path.read_text(encoding="utf-8")


# ============================================================================
# Word grouping
# ============================================================================


def test_group_words_english_one_per_cue() -> None:
    words = [
        WordTiming(text="Hello", start_ms=0, end_ms=400),
        WordTiming(text="world", start_ms=400, end_ms=900),
    ]
    chunks = group_words_for_segment(
        words, segment_start_ms=0, segment_end_ms=900, fallback_text=""
    )
    assert len(chunks) == 2
    assert chunks[0] == (0, 400, "Hello")
    assert chunks[1] == (400, 900, "world")


def test_group_words_chinese_chunks_into_groups_of_5() -> None:
    words = [
        WordTiming(text=ch, start_ms=i * 200, end_ms=(i + 1) * 200)
        for i, ch in enumerate("今天天气很好阳光明媚")  # 10 chars
    ]
    chunks = group_words_for_segment(
        words, segment_start_ms=1000, segment_end_ms=3000, fallback_text=""
    )
    # Should yield 2 chunks of 5 chars each.
    assert len(chunks) == 2
    assert chunks[0][2] == "今天天气很"
    assert chunks[1][2] == "好阳光明媚"
    # Offsets shifted by segment_start_ms.
    assert chunks[0][0] == 1000


def test_group_words_no_timing_returns_fallback() -> None:
    chunks = group_words_for_segment(
        [], segment_start_ms=2000, segment_end_ms=5000, fallback_text="full segment"
    )
    assert chunks == [(2000, 5000, "full segment")]


# ============================================================================
# build_cues across segments
# ============================================================================


def test_build_cues_advances_cursor_across_segments() -> None:
    seg0 = Segment(index=0, text="Hello world.")
    seg1 = Segment(index=1, text="Second segment.")

    tts_results = {
        0: TTSResult(
            audio_path=Path("/tmp/0.mp3"),
            duration_ms=1200,
            words=[
                WordTiming(text="Hello", start_ms=0, end_ms=500),
                WordTiming(text="world", start_ms=500, end_ms=1200),
            ],
        ),
        1: TTSResult(
            audio_path=Path("/tmp/1.mp3"),
            duration_ms=1500,
            words=[
                WordTiming(text="Second", start_ms=0, end_ms=700),
                WordTiming(text="segment", start_ms=700, end_ms=1500),
            ],
        ),
    }

    cues = build_cues([seg0, seg1], tts_results)
    assert [c.index for c in cues] == [1, 2, 3, 4]
    # Second segment cues offset by first segment duration (1200ms).
    assert cues[2].start_ms == 1200
    assert cues[2].end_ms == 1900
    assert cues[3].end_ms == 2700


# ============================================================================
# Orchestrator subtitle stage (mocked TTS)
# ============================================================================


class _StubNLP:
    async def segment(self, article: str, target_seconds_per_seg: int = 8):
        return [Segment(index=0, text=article, keywords=("scene",))]

    async def keywords(self, segment: Segment, top_k: int = 5):
        return list(segment.keywords)

    async def translate(self, text: str, source_lang: str, target_lang: str):
        return text

    async def detect_mood(self, article: str):
        return "calm"


class _FakeCommunicate:
    def __init__(self, text: str, voice: str) -> None:
        self.text = text
        self.voice = voice

    async def stream(self):
        yield {"type": "audio", "data": b"FAKE"}
        yield {
            "type": "WordBoundary",
            "offset": 0,
            "duration": 10_000_000,
            "text": self.text,
        }


@pytest.mark.asyncio
async def test_orchestrator_runs_tts_and_subtitles(
    tmp_path: Path, stub_compose, stub_render
) -> None:
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.orchestrator import PipelineOrchestrator
    from app.pipeline.tts import edge_tts as edge_mod
    from app.pipeline.tts.edge_tts import EdgeTTS

    with patch.object(edge_mod, "edge_tts") as mock_edge:
        mock_edge.Communicate.side_effect = lambda text, voice: _FakeCommunicate(text, voice)

        job = Job(article="Some short narration.", nlp_backend="local")
        orchestrator = PipelineOrchestrator(
            nlp=_StubNLP(),
            media_providers=[],  # skip media
            tts=EdgeTTS(tmp_path),
            work_dir=tmp_path,
            compose_fn=stub_compose,
            render_fn=stub_render,
        )
        result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert orchestrator.tts_results
    assert orchestrator.srt_path is not None
    assert orchestrator.srt_path.exists()
    srt_text = orchestrator.srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000" in srt_text
    # Segment 0 had a single 1000ms word (10_000_000 ticks = 1000ms).
    assert "00:00:01,000" in srt_text

    assert result.status is JobStatus.DONE
    assert orchestrator.tts_results
    assert orchestrator.srt_path is not None
    assert orchestrator.srt_path.exists()
    srt_text = orchestrator.srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000" in srt_text
    # Segment 0 had a single 1000ms word (10_000_000 ticks = 1000ms).
    assert "00:00:01,000" in srt_text
