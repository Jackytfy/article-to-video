"""SRT subtitle generator.

Builds SRT cues from per-segment TTS results. Cues are placed at the segment's
start in the global timeline; for Chinese, characters are grouped into 4-6
char chunks for readability.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.pipeline.models import Segment
from app.pipeline.tts.edge_tts import TTSResult, WordTiming

# Roughly: 4-6 zh chars per cue, or one Edge "word" for latin scripts.
_ZH_GROUP_SIZE = 5
_MAX_CUE_CHARS = 35


@dataclass(frozen=True)
class SubtitleCue:
    """One SRT cue."""

    index: int
    start_ms: int
    end_ms: int
    text: str


def _is_chinese_word(w: str) -> bool:
    return any("一" <= c <= "鿿" for c in w)


def _format_ts(ms: int) -> str:
    """SRT timestamp: HH:MM:SS,mmm."""
    if ms < 0:
        ms = 0
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def group_words_for_segment(
    words: list[WordTiming],
    segment_start_ms: int,
    segment_end_ms: int,
    fallback_text: str,
) -> list[tuple[int, int, str]]:
    """Group word timings into subtitle-sized chunks.

    Returns list of (start_ms, end_ms, text) tuples, all offset by
    `segment_start_ms`.
    """
    if not words:
        # No word timing — show the full segment text spanning its duration.
        return [(segment_start_ms, segment_end_ms, fallback_text.strip())]

    chunks: list[tuple[int, int, str]] = []
    buf_words: list[WordTiming] = []
    buf_chars = 0

    for w in words:
        text = w.text
        if not text.strip():
            continue

        is_zh = _is_chinese_word(text)
        group_cap = _ZH_GROUP_SIZE if is_zh else 1
        char_cap = _MAX_CUE_CHARS

        will_overflow = (
            len(buf_words) >= group_cap
            or buf_chars + len(text) > char_cap
        )
        if buf_words and will_overflow:
            chunks.append(_emit_chunk(buf_words, segment_start_ms))
            buf_words, buf_chars = [], 0

        buf_words.append(w)
        buf_chars += len(text)

    if buf_words:
        chunks.append(_emit_chunk(buf_words, segment_start_ms))

    return chunks


def _emit_chunk(
    buf_words: list[WordTiming], segment_start_ms: int
) -> tuple[int, int, str]:
    start = segment_start_ms + buf_words[0].start_ms
    end = segment_start_ms + buf_words[-1].end_ms
    sep = "" if _is_chinese_word(buf_words[0].text) else " "
    text = sep.join(w.text for w in buf_words).strip()
    return (start, end, text)


def build_cues(
    segments: list[Segment],
    tts_results: dict[int, TTSResult],
) -> list[SubtitleCue]:
    """Build cues across all segments using accumulated audio start offsets."""
    cues: list[SubtitleCue] = []
    cursor_ms = 0

    for seg in segments:
        result = tts_results.get(seg.index)
        if result is None:
            continue

        seg_chunks = group_words_for_segment(
            result.words,
            segment_start_ms=cursor_ms,
            segment_end_ms=cursor_ms + result.duration_ms,
            fallback_text=seg.text,
        )
        for start_ms, end_ms, text in seg_chunks:
            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                )
            )
        cursor_ms += result.duration_ms

    return cues


def render_srt(cues: Iterable[SubtitleCue]) -> str:
    """Render SRT plain-text from cues."""
    parts: list[str] = []
    for cue in cues:
        parts.append(
            f"{cue.index}\n"
            f"{_format_ts(cue.start_ms)} --> {_format_ts(cue.end_ms)}\n"
            f"{cue.text}\n"
        )
    return "\n".join(parts)


def write_srt(cues: Iterable[SubtitleCue], path: Path) -> Path:
    """Write SRT file. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(cues), encoding="utf-8")
    return path


__all__ = [
    "SubtitleCue",
    "build_cues",
    "render_srt",
    "write_srt",
    "group_words_for_segment",
]
