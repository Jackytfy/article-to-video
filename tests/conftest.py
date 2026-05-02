"""Shared test fixtures and stubs."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.pipeline.tts.edge_tts import TTSResult, WordTiming


class StubTTS:
    """Test double for EdgeTTS — writes empty MP3s, returns synthetic timing."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str, voice: str, filename_stem: str) -> TTSResult:
        audio = self._output_dir / f"{filename_stem}.mp3"
        audio.write_bytes(b"FAKE-MP3")
        # 100 ms per character, capped 200 chars.
        duration_ms = min(len(text), 200) * 100
        words = [
            WordTiming(text=text, start_ms=0, end_ms=duration_ms),
        ]
        return TTSResult(audio_path=audio, duration_ms=duration_ms, words=words)

    async def synthesize_segments(
        self,
        items: Iterable[tuple[int, str, str]],
        max_concurrency: int = 4,
    ) -> dict[int, TTSResult]:
        out: dict[int, TTSResult] = {}
        for idx, text, voice in items:
            out[idx] = await self.synthesize(text, voice, f"seg-{idx:04d}")
        return out


@pytest.fixture
def stub_tts(tmp_path: Path) -> StubTTS:
    """Returns a StubTTS instance writing into tmp_path."""
    return StubTTS(tmp_path)


@pytest.fixture
def stub_compose():
    """Returns a fake compose_fn that yields a MagicMock clip."""
    fake_clip = MagicMock(name="composed_clip")
    fake_clip.close = MagicMock()
    return MagicMock(return_value=fake_clip, name="compose_fn")


@pytest.fixture
def stub_render():
    """Returns a fake render_fn that writes a placeholder MP4."""

    def _render(clip, output_path, *, fps=30, use_gpu=False, threads=4):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"FAKE-MP4")
        return Path(output_path)

    return _render


@pytest.fixture(autouse=True)
def _no_real_music_providers(request, monkeypatch):
    """Prevent any test from accidentally hitting the real Jamendo API.

    Tests that focus on the music stage explicitly inject `music_providers=...`
    on the orchestrator, which bypasses this default. Other tests fall through
    to `make_music_providers()` — without this fixture they'd hit live Jamendo
    when JAMENDO_CLIENT_ID is set in the dev `.env`.

    Tests that need to verify the factory itself can opt out with
    `@pytest.mark.real_music_factory`.
    """
    if request.node.get_closest_marker("real_music_factory"):
        return

    import app.pipeline.music as music_pkg
    import app.pipeline.orchestrator as orch_pkg

    monkeypatch.setattr(music_pkg, "make_providers", lambda settings=None: [])
    monkeypatch.setattr(orch_pkg, "make_music_providers", lambda settings=None: [])
