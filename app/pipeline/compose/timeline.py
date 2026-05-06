"""Timeline composer (MoviePy 2.x).

Builds a CompositeVideoClip from per-segment media + narration audio + subtitles.
- Image segments: ImageClip with subtle Ken Burns zoom.
- Video segments: VideoFileClip resized + center-cropped, looped if shorter
  than narration duration.
- Audio: per-segment narration MP3 concatenated and laid as the master track.
- Subtitles: TextClip overlays from build_subtitle_overlays.

MoviePy is imported lazily inside `compose_video` so unit tests can patch it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.pipeline.compose.aspect import (
    AspectRatio,
    CropPlan,
    plan_center_crop,
    target_dimensions,
)
from app.pipeline.compose.overlay import build_subtitle_overlays
from app.pipeline.models import MediaAsset, Segment
from app.pipeline.subtitle.srt import SubtitleCue
from app.pipeline.tts.edge_tts import TTSResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def compose_video(
    segments: list[Segment],
    segment_assets: dict[int, MediaAsset],
    tts_results: dict[int, TTSResult],
    cues: list[SubtitleCue],
    aspect: AspectRatio,
    burn_subtitles: bool,
    font_path: str | None = None,
    bgm_path: Path | None = None,
    bgm_gain: float = 0.10,
    gap_between_segments: float = 0.15,
) -> Any:
    """Build a CompositeVideoClip ready for write_videofile().

    Args:
        bgm_path: optional BGM file. Mixed under narration at `bgm_gain`
            (0.10 ≈ -20dB). Looped or trimmed to total video duration.
        bgm_gain: linear gain applied to BGM (1.0 = unchanged).

    Caller is responsible for closing/disposing the clip.
    """
    from moviepy import (  # noqa: F401
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )

    target_w, target_h = target_dimensions(aspect)

    visual_clips: list[Any] = []
    audio_clips: list[Any] = []
    cursor_s = 0.0
    gap = gap_between_segments

    # Add brief silence at start for natural feel
    if gap > 0:
        cursor_s = gap

    for seg in segments:
        result = tts_results.get(seg.index)
        if result is None:
            logger.warning("Segment %d has no TTS audio; skipping.", seg.index)
            continue

        duration_s = max(result.duration_ms / 1000.0, 0.5)
        asset = segment_assets.get(seg.index)

        visual = _build_segment_visual(
            asset=asset,
            duration_s=duration_s + gap,  # Extend visual slightly for gap
            target_w=target_w,
            target_h=target_h,
            aspect=aspect,
        )
        visual_clips.append(visual)

        audio_clip = AudioFileClip(str(result.audio_path))
        audio_clips.append(audio_clip.with_start(cursor_s))
        cursor_s += duration_s + gap

    if not visual_clips:
        raise RuntimeError("compose_video: no segments produced visual clips")

    base_video = concatenate_videoclips(visual_clips, method="compose")
    total_duration_s = cursor_s

    audio_layers: list[Any] = list(audio_clips)
    if bgm_path is not None and total_duration_s > 0:
        bgm_layer = _build_bgm_layer(bgm_path, total_duration_s, bgm_gain)
        if bgm_layer is not None:
            audio_layers.append(bgm_layer)

    narration = CompositeAudioClip(audio_layers) if audio_layers else None

    layers: list[Any] = [base_video]
    if burn_subtitles and cues:
        layers.extend(
            build_subtitle_overlays(
                cues=cues,
                target_w=target_w,
                target_h=target_h,
                font_path=font_path,
            )
        )

    final = CompositeVideoClip(layers, size=(target_w, target_h))
    if narration is not None:
        final = final.with_audio(narration)

    return final


def _build_bgm_layer(
    bgm_path: Path, total_duration_s: float, gain: float
) -> Any | None:
    """Load BGM, loop or trim to total duration, apply gain. Returns clip or None."""
    from moviepy import AudioFileClip, concatenate_audioclips

    try:
        bgm = AudioFileClip(str(bgm_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("BGM load failed (%s): %s", bgm_path, exc)
        return None

    bgm_duration = float(bgm.duration) if isinstance(bgm.duration, (int, float)) else 0
    if bgm_duration <= 0:
        bgm.close()
        return None

    if bgm_duration < total_duration_s:
        loops = int(total_duration_s / bgm_duration) + 1
        bgm = concatenate_audioclips([bgm] * loops)

    bgm = bgm.subclipped(0, total_duration_s)
    bgm = bgm.with_volume_scaled(gain)
    return bgm.with_start(0)


def _build_segment_visual(
    asset: MediaAsset | None,
    duration_s: float,
    target_w: int,
    target_h: int,
    aspect: AspectRatio,
) -> Any:
    """Build a per-segment visual clip scaled+cropped to target dims."""
    from moviepy import ColorClip, ImageClip, VideoFileClip

    if asset is None or asset.local_path is None:
        # Fallback: solid dark background.
        return ColorClip((target_w, target_h), color=(20, 20, 30)).with_duration(
            duration_s
        )

    src_path = asset.local_path
    if asset.media_type == "video":
        clip = VideoFileClip(str(src_path), audio=False)
        clip_duration = float(clip.duration) if isinstance(clip.duration, (int, float)) else 0.0
        if clip_duration < duration_s:
            # Loop short clips by self-concatenation.
            from moviepy import concatenate_videoclips

            loops = int(duration_s / clip_duration) + 1
            clip = concatenate_videoclips([clip] * loops).subclipped(0, duration_s)
        else:
            clip = clip.subclipped(0, duration_s)
    else:
        clip = ImageClip(str(src_path)).with_duration(duration_s)

    plan = plan_center_crop(asset.width, asset.height, aspect)
    return _apply_crop_plan(clip, plan)


def _apply_crop_plan(clip: Any, plan: CropPlan) -> Any:
    """Resize then center-crop a clip per CropPlan."""
    from moviepy.video.fx import Crop

    scaled_w = int(round(clip.w * plan.scale))
    scaled_h = int(round(clip.h * plan.scale))
    clip = clip.resized((scaled_w, scaled_h))

    return clip.with_effects(
        [
            Crop(
                x1=plan.crop_x,
                y1=plan.crop_y,
                x2=plan.crop_x + plan.target_w,
                y2=plan.crop_y + plan.target_h,
            )
        ]
    )


__all__ = ["compose_video"]
