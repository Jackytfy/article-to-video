"""Aspect-ratio helpers: target dims + smart center-crop math.

Pure functions. No MoviePy dependency so unit tests are fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AspectRatio = Literal["16:9", "9:16", "1:1"]

# 1080p target dims per aspect.
_TARGET_DIMS: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


def target_dimensions(aspect: AspectRatio) -> tuple[int, int]:
    """Return (width, height) for the target aspect at 1080 base."""
    if aspect not in _TARGET_DIMS:
        raise ValueError(f"Unsupported aspect ratio: {aspect!r}")
    return _TARGET_DIMS[aspect]


@dataclass(frozen=True)
class CropPlan:
    """Plan for fitting a source frame into the target box.

    Strategy: scale source by `scale`, then center-crop the over-sized axis.
    Width/height of the output equal target_w/target_h.
    """

    scale: float
    crop_x: int  # left crop in scaled-source pixels
    crop_y: int
    target_w: int
    target_h: int


def plan_center_crop(
    src_w: int,
    src_h: int,
    aspect: AspectRatio,
) -> CropPlan:
    """Center-crop fitting src into target aspect.

    The source is scaled so the SHORTER target axis fills, then we crop the
    longer axis. This keeps the most relevant center of the frame.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source dims: {src_w}x{src_h}")

    target_w, target_h = target_dimensions(aspect)

    # Scale so source fills both axes (cover, not contain).
    scale = max(target_w / src_w, target_h / src_h)
    scaled_w = int(round(src_w * scale))
    scaled_h = int(round(src_h * scale))

    crop_x = max((scaled_w - target_w) // 2, 0)
    crop_y = max((scaled_h - target_h) // 2, 0)

    return CropPlan(
        scale=scale,
        crop_x=crop_x,
        crop_y=crop_y,
        target_w=target_w,
        target_h=target_h,
    )


__all__ = [
    "AspectRatio",
    "CropPlan",
    "target_dimensions",
    "plan_center_crop",
]
