"""Edge-TTS wrapper.

Synthesizes per-segment narration to MP3 + captures word-level timing for
subtitle alignment via Edge-TTS WordBoundary events.

Edge-TTS is free (uses Microsoft's public Edge browser TTS endpoint), no API
key needed. WordBoundary offsets are in 100-nanosecond units; convert to ms.

Reference: https://github.com/rany2/edge-tts
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import edge_tts

from app.pipeline.tts.voices import resolve_voice

logger = logging.getLogger(__name__)

# Edge-TTS reports offsets in 100ns ticks.
_TICKS_PER_MS = 10_000


@dataclass(frozen=True)
class WordTiming:
    """One word/character span aligned to TTS audio."""

    text: str
    start_ms: int
    end_ms: int


@dataclass
class TTSResult:
    """Output of synthesizing one segment."""

    audio_path: Path
    duration_ms: int
    words: list[WordTiming]


class EdgeTTS:
    """Async Edge-TTS synthesizer."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(
        self,
        text: str,
        voice: str,
        filename_stem: str,
    ) -> TTSResult:
        """Synthesize `text` to MP3, return path + word timings."""
        if not text.strip():
            raise ValueError("TTS text must be non-empty")

        voice_id = resolve_voice(voice)
        audio_path = self._output_dir / f"{filename_stem}.mp3"

        communicate = edge_tts.Communicate(text, voice_id)
        words: list[WordTiming] = []
        audio_chunks: list[bytes] = []

        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio":
                audio_chunks.append(chunk["data"])
            elif ctype == "WordBoundary":
                words.append(_word_from_chunk(chunk))

        audio_path.write_bytes(b"".join(audio_chunks))

        duration_ms = words[-1].end_ms if words else 0
        if duration_ms == 0:
            duration_ms = _estimate_duration_ms(text)

        return TTSResult(audio_path=audio_path, duration_ms=duration_ms, words=words)

    async def synthesize_segments(
        self,
        items: list[tuple[int, str, str]],
        max_concurrency: int = 4,
    ) -> dict[int, TTSResult]:
        """Synthesize many segments in parallel.

        Args:
            items: list of (segment_index, text, voice).
            max_concurrency: bound on simultaneous Edge-TTS streams.
        """
        sem = asyncio.Semaphore(max_concurrency)

        async def _bounded(idx: int, text: str, voice: str) -> tuple[int, TTSResult]:
            async with sem:
                logger.info("TTS synth segment %d (voice=%s)", idx, voice)
                result = await self.synthesize(text, voice, filename_stem=f"seg-{idx:04d}")
                return idx, result

        tasks = [_bounded(idx, text, voice) for idx, text, voice in items]
        completed = await asyncio.gather(*tasks)
        return dict(completed)


# ---- Helpers ----------------------------------------------------------------


def _word_from_chunk(chunk: dict[str, Any]) -> WordTiming:
    offset_ticks = int(chunk.get("offset", 0))
    duration_ticks = int(chunk.get("duration", 0))
    start_ms = offset_ticks // _TICKS_PER_MS
    end_ms = (offset_ticks + duration_ticks) // _TICKS_PER_MS
    text = str(chunk.get("text", ""))
    return WordTiming(text=text, start_ms=start_ms, end_ms=end_ms)


def _estimate_duration_ms(text: str) -> int:
    """Very rough fallback if Edge-TTS gives no WordBoundary events."""
    han = sum(1 for c in text if "一" <= c <= "鿿")
    if han > len(text) * 0.3:
        return int((han / 4.5) * 1000)
    return int((len(text.split()) / 2.5) * 1000)
