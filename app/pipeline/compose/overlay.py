"""Subtitle overlay builder.

Converts SubtitleCues into MoviePy TextClip overlays positioned at the
bottom-third of the frame, with stroke for legibility.

MoviePy is imported lazily so unit tests can mock it via the module attribute.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.pipeline.subtitle.srt import SubtitleCue

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Sensible defaults for Windows; override via app config.
_DEFAULT_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei (zh + en)
    "C:/Windows/Fonts/simhei.ttf",     # SimHei (zh)
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def resolve_font_path(preferred: str | None = None) -> str | None:
    """Return first existing font path. Returns None if none found."""
    candidates = [preferred] if preferred else []
    candidates += _DEFAULT_FONT_PATHS
    for path in candidates:
        if path and Path(path).exists():
            return path
    logger.warning(
        "No font file located. Subtitles may fail; set font path in config."
    )
    return None


def build_subtitle_overlays(
    cues: list[SubtitleCue],
    target_w: int,
    target_h: int,
    font_path: str | None = None,
    font_size: int | None = None,
) -> list[Any]:
    """Build MoviePy TextClips for each cue. Returns list of clips.

    Each clip is timed (start/duration) and positioned in the bottom third.
    """
    if not cues:
        return []

    # Lazy import so the module loads in test envs without MoviePy.
    from moviepy import TextClip

    font = resolve_font_path(font_path)
    if font is None:
        logger.warning("Skipping subtitles: no usable font found")
        return []

    fs = font_size or max(28, target_h // 24)

    overlays: list[Any] = []
    for cue in cues:
        duration_s = max((cue.end_ms - cue.start_ms) / 1000.0, 0.05)
        start_s = cue.start_ms / 1000.0

        clip = TextClip(
            font=font,
            text=cue.text,
            font_size=fs,
            color="white",
            stroke_color="black",
            stroke_width=3,
            method="caption",
            size=(int(target_w * 0.85), None),
            text_align="center",
        )
        # 76% from top -> bottom third placement.
        clip = clip.with_start(start_s).with_duration(duration_s).with_position(
            ("center", 0.78), relative=True
        )
        overlays.append(clip)

    return overlays


__all__ = ["build_subtitle_overlays", "resolve_font_path"]
