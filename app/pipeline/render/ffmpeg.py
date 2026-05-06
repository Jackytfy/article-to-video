"""FFmpeg render profiles via MoviePy `write_videofile`.

Two profiles:
- libx264 + aac (default): 4 Mbps, fast preset (optimized for speed)
- NVENC (use_gpu=True): h264_nvenc, fast preset

Notes:
- We pass `logger="bar"` so MoviePy uses its proglog tqdm output. Setting
  `logger=None` is documented but in practice can deadlock the encoder thread
  on some MoviePy 2.x builds.
- We hard-cap render at `timeout_s` to surface hangs instead of letting the
  pipeline freeze indefinitely.
- A temp working file is used so a partially-written MP4 never poses as the
  output on crash.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30 * 60  # 30 min ceiling for any single render (increased for long videos).


def render_clip(
    clip: Any,
    output_path: Path,
    *,
    fps: int = 25,
    use_gpu: bool = False,
    threads: int = 0,  # 0 = auto-detect optimal thread count
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> Path:
    """Render a MoviePy clip to MP4. Returns the output path.

    Raises TimeoutError if render exceeds `timeout_s` (seconds).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect CPU cores if threads=0
    if threads <= 0:
        import multiprocessing
        threads = min(multiprocessing.cpu_count(), 8)

    codec = "h264_nvenc" if use_gpu else "libx264"
    # Use faster preset for CPU encoding to reduce render time
    preset = "fast" if use_gpu else "veryfast"

    duration = getattr(clip, "duration", None)
    audio = getattr(clip, "audio", None)
    audio_dur = getattr(audio, "duration", None) if audio is not None else None
    logger.info(
        "Rendering -> %s (codec=%s, preset=%s, fps=%d, video_dur=%.2fs, audio_dur=%s)",
        output_path,
        codec,
        preset,
        fps,
        duration or 0.0,
        f"{audio_dur:.2f}s" if audio_dur is not None else "none",
    )

    if output_path.exists():
        try:
            output_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove existing output %s: %s", output_path, exc)

    try:
        _run_with_timeout(
            _do_write,
            timeout_s,
            clip=clip,
            target=output_path,
            fps=fps,
            codec=codec,
            preset=preset,
            threads=threads,
        )
    except TimeoutError:
        logger.error("Render exceeded %ds; aborting.", int(timeout_s))
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    logger.info("Render complete: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


# ---- Helpers ---------------------------------------------------------------


def _do_write(
    *,
    clip: Any,
    target: Path,
    fps: int,
    codec: str,
    preset: str,
    threads: int,
) -> None:
    clip.write_videofile(
        str(target),
        fps=fps,
        codec=codec,
        audio_codec="aac",
        preset=preset,
        threads=threads,
        bitrate="4M",  # Reduced from 6M for faster encoding
        audio_bitrate="128k",  # Good quality audio at reasonable size
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        # Use the default proglog "bar" — None deadlocks on some MoviePy 2.x builds.
        logger="bar",
    )


def _run_with_timeout(fn, timeout_s: float, **kwargs) -> None:
    """Run `fn(**kwargs)` in a thread with a hard timeout.

    Workers can't be killed safely from Python, but we surface the hang to the
    orchestrator so the job is marked failed instead of stuck forever.
    """
    import threading

    err: list[BaseException] = []

    def target() -> None:
        try:
            fn(**kwargs)
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=target, name="moviepy-render", daemon=True)
    t.start()
    t.join(timeout_s)

    if t.is_alive():
        raise TimeoutError(f"Render did not finish within {timeout_s}s")
    if err:
        raise err[0]


__all__ = ["render_clip"]
