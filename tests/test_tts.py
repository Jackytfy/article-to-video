"""Tests for TTS stage (Edge-TTS wrapper) with mocked Communicate."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# Voice resolver
# ============================================================================


def test_resolve_voice_passthrough_full_id() -> None:
    from app.pipeline.tts.voices import resolve_voice

    assert resolve_voice("zh-CN-XiaoxiaoNeural") == "zh-CN-XiaoxiaoNeural"
    assert resolve_voice("en-US-AriaNeural") == "en-US-AriaNeural"


def test_resolve_voice_short_codes() -> None:
    from app.pipeline.tts.voices import resolve_voice

    assert resolve_voice("zh") == "zh-CN-XiaoxiaoNeural"
    assert resolve_voice("en") == "en-US-AriaNeural"
    assert resolve_voice("ja") == "ja-JP-NanamiNeural"


def test_resolve_voice_unknown_falls_back() -> None:
    from app.pipeline.tts.voices import resolve_voice

    assert resolve_voice(None) == "en-US-AriaNeural"
    assert resolve_voice("xx-YY") == "en-US-AriaNeural"


# ============================================================================
# Edge-TTS wrapper (Communicate mocked)
# ============================================================================


class _FakeCommunicate:
    """Stand-in for edge_tts.Communicate that yields fake audio + WordBoundary."""

    def __init__(self, text: str, voice: str, *, scenario: str = "with_words") -> None:
        self.text = text
        self.voice = voice
        self.scenario = scenario

    async def stream(self):
        if self.scenario == "with_words":
            yield {"type": "audio", "data": b"FAKE-AUDIO-CHUNK-1"}
            yield {
                "type": "WordBoundary",
                "offset": 0,
                "duration": 5_000_000,  # 500ms
                "text": "Hello",
            }
            yield {"type": "audio", "data": b"FAKE-AUDIO-CHUNK-2"}
            yield {
                "type": "WordBoundary",
                "offset": 5_000_000,
                "duration": 7_000_000,  # 700ms
                "text": "world",
            }
        elif self.scenario == "no_words":
            yield {"type": "audio", "data": b"FAKE"}


@pytest.mark.asyncio
async def test_edge_tts_synthesize_writes_audio(tmp_path: Path) -> None:
    from app.pipeline.tts import edge_tts as mod

    with patch.object(mod, "edge_tts") as mock_edge:
        mock_edge.Communicate.side_effect = lambda text, voice: _FakeCommunicate(text, voice)

        synth = mod.EdgeTTS(tmp_path)
        result = await synth.synthesize("Hello world", "en-US-AriaNeural", "seg-0001")

    assert result.audio_path.exists()
    assert result.audio_path.read_bytes().startswith(b"FAKE-AUDIO")
    assert len(result.words) == 2
    assert result.words[0].text == "Hello"
    assert result.words[0].start_ms == 0
    assert result.words[0].end_ms == 500
    assert result.words[1].start_ms == 500
    assert result.words[1].end_ms == 1200
    assert result.duration_ms == 1200


@pytest.mark.asyncio
async def test_edge_tts_handles_no_word_boundary(tmp_path: Path) -> None:
    from app.pipeline.tts import edge_tts as mod

    with patch.object(mod, "edge_tts") as mock_edge:
        mock_edge.Communicate.side_effect = lambda text, voice: _FakeCommunicate(
            text, voice, scenario="no_words"
        )

        synth = mod.EdgeTTS(tmp_path)
        result = await synth.synthesize(
            "Some narration text.", "en-US-AriaNeural", "seg-0002"
        )

    assert result.words == []
    assert result.duration_ms > 0  # estimated fallback


@pytest.mark.asyncio
async def test_edge_tts_synthesize_segments_parallel(tmp_path: Path) -> None:
    from app.pipeline.tts import edge_tts as mod

    with patch.object(mod, "edge_tts") as mock_edge:
        mock_edge.Communicate.side_effect = lambda text, voice: _FakeCommunicate(text, voice)

        synth = mod.EdgeTTS(tmp_path)
        items = [
            (0, "First chunk.", "en-US-AriaNeural"),
            (1, "Second chunk.", "en-US-AriaNeural"),
            (2, "Third chunk.", "en-US-AriaNeural"),
        ]
        results = await synth.synthesize_segments(items, max_concurrency=2)

    assert set(results.keys()) == {0, 1, 2}
    for r in results.values():
        assert r.audio_path.exists()


@pytest.mark.asyncio
async def test_edge_tts_rejects_empty_text(tmp_path: Path) -> None:
    from app.pipeline.tts.edge_tts import EdgeTTS

    synth = EdgeTTS(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        await synth.synthesize("   ", "en-US-AriaNeural", "x")
