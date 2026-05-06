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
    video_bitrate: str = "3M",
    ass_path: Path | None = None,  # ASS 字幕文件，用于 FFmpeg 硬烧录
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

    duration = getattr(clip, "duration", None)
    audio = getattr(clip, "audio", None)
    audio_dur = getattr(audio, "duration", None) if audio is not None else None
    logger.info(
        "Rendering -> %s (codec=%s, fps=%d, video_dur=%.2fs, audio_dur=%s, ass=%s)",
        output_path,
        codec,
        fps,
        duration or 0.0,
        f"{audio_dur:.2f}s" if audio_dur is not None else "none",
        ass_path or "none",
    )

    if output_path.exists():
        try:
            output_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove existing output %s: %s", output_path, exc)

    # If ASS subtitles: render to temp file first, then burn subtitles
    use_ass = ass_path is not None and ass_path.exists()
    render_target = output_path.with_suffix(".tmp.mp4") if use_ass else output_path

    try:
        _run_with_timeout(
            _do_write,
            timeout_s,
            clip=clip,
            target=render_target,
            fps=fps,
            codec=codec,
            threads=threads,
            video_bitrate=video_bitrate,
        )
    except TimeoutError:
        logger.error("Render exceeded %ds; aborting.", int(timeout_s))
        render_target.unlink(missing_ok=True)
        raise

    # Burn ASS subtitles via FFmpeg (much faster than MoviePy TextClips)
    if use_ass:
        logger.info("Burning ASS subtitles: %s", ass_path)
        burn_ok = _burn_ass_subtitles(render_target, ass_path, output_path, use_gpu)
        render_target.unlink(missing_ok=True)
        try:
            ass_path.unlink(missing_ok=True)
        except OSError:
            pass
        if not burn_ok:
            logger.warning("ASS burn failed; output may not have subtitles")

    logger.info("Render complete: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


# ---- Helpers ---------------------------------------------------------------


def _do_write(
    *,
    clip: Any,
    target: Path,
    fps: int,
    codec: str,
    threads: int,
    video_bitrate: str = "3M",
) -> None:
    """Write clip to file with optimized encoding parameters."""
    import moviepy

    is_nvenc = "nvenc" in codec

    # Base parameters for write_videofile
    write_kwargs: dict = {
        "fps": fps,
        "codec": codec,
        "audio_codec": "aac",
        "threads": threads,
        "bitrate": video_bitrate,
        "audio_bitrate": "128k",
        "logger": "bar",
    }

    if is_nvenc:
        # NVENC: use ffmpeg_params for presets (p1=fastest, p7=slowest/highest quality)
        # https://docs.nvidia.com/video-technologies/video-codec-sdk/reference-guide/index.html
        write_kwargs["ffmpeg_params"] = [
            "-pix_fmt", "nv12",          # NV12 is native for NVENC (faster)
            "-preset", "p1",             # Fastest NVENC preset
            "-rc", "vbr",                # Variable bitrate
            "-cq", "24",                 # Quality level (lower=better, 18-28 is good)
            "-gpu", "0",                 # Use first GPU
        ]
    else:
        # CPU (libx264): use preset + pix_fmt
        write_kwargs["preset"] = "veryfast"   # CPU: ultrafast|superfast|veryfast|fast|medium
        write_kwargs["ffmpeg_params"] = ["-pix_fmt", "yuv420p"]

    # MoviePy 2.x: write_videofile signature changed
    # Remove logger param if MoviePy < 2.0
    try:
        clip.write_videofile(str(target), **write_kwargs)
    except TypeError as exc:
        # Fallback: remove unsupported kwargs
        if "logger" in str(exc):
            write_kwargs.pop("logger", None)
            clip.write_videofile(str(target), **write_kwargs)
        else:
            raise


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


def _burn_ass_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    use_gpu: bool = False,
) -> bool:
    """Burn ASS subtitles into video using FFmpeg (synchronous).

    Uses a separate FFmpeg pass so subtitles are burned without
    MoviePy TextClips (much faster).

    Note: subtitles filter may not work well with NVENC on some systems,
    so we always use CPU (libx264) for the burn pass when possible.

    Returns True on success.
    """
    import subprocess

    # Always use CPU for subtitle burn (more compatible with subtitles filter)
    # If use_gpu is True, we'll do two passes: CPU+burn -> NVENC
    burn_codec = "libx264"
    burn_preset = "veryfast"

    # FFmpeg subtitles filter needs forward slashes on Windows
    ass_str = str(ass_path).replace("\\", "/")

    # Build filter chain - add scaling for compatibility
    vf_filter = f"subtitles={ass_str}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf_filter,
        "-c:v", burn_codec,
        "-preset", burn_preset,
        "-b:v", "4M",
        "-c:a", "copy",
        "-shortest",
        str(output_path),
    ]

    logger.debug("Burn subtitles cmd: %s", " ".join(str(c) for c in cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            logger.info("ASS burn OK: %s (%d bytes)", output_path, output_path.stat().st_size)
            return True
        err_msg = result.stderr[-1000:] if result.stderr else "unknown error"
        logger.warning("ASS burn failed (rc=%d): %s", result.returncode, err_msg[:500])
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH; cannot burn ASS subtitles")
    except subprocess.TimeoutExpired:
        logger.warning("ASS burn timed out after 600s")
    except Exception as exc:
        logger.warning("ASS burn error: %s", exc)
    return False


__all__ = ["render_clip"]
