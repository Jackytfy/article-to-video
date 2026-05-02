"""Asset ranking and orientation helpers."""
from __future__ import annotations

from app.pipeline.models import MediaAsset, Orientation


def aspect_to_orientation(aspect_ratio: str) -> Orientation:
    """Map aspect ratio string to preferred orientation."""
    if aspect_ratio == "16:9":
        return "landscape"
    if aspect_ratio == "9:16":
        return "portrait"
    if aspect_ratio == "1:1":
        return "square"
    raise ValueError(f"Unsupported aspect ratio: {aspect_ratio!r}")


def score_asset(
    asset: MediaAsset,
    keywords: list[str],
    target_orientation: Orientation,
) -> float:
    """Higher = better. Combines keyword overlap, orientation, resolution.

    Score components:
    - Keyword overlap with provider tags (0..3 typical)
    - Orientation exact match (+2) or square fallback (+0.5)
    - Resolution bonus (capped)
    - Video preferred slightly over image (+0.3)
    """
    score = 0.0

    if keywords and asset.tags:
        kw_lower = {k.lower() for k in keywords}
        tag_lower = {t.lower() for t in asset.tags}
        overlap = len(kw_lower & tag_lower)
        score += overlap

    if asset.orientation == target_orientation:
        score += 2.0
    elif asset.orientation == "square":
        score += 0.5

    # Resolution: prefer ≥1080 on min dimension; cap bonus at +1.
    min_dim = min(asset.width, asset.height)
    score += min(min_dim / 1080.0, 1.0)

    if asset.media_type == "video":
        score += 0.3

    return score


def rank_assets(
    assets: list[MediaAsset],
    keywords: list[str],
    target_orientation: Orientation,
) -> list[MediaAsset]:
    """Return assets sorted best-first."""
    return sorted(
        assets,
        key=lambda a: score_asset(a, keywords, target_orientation),
        reverse=True,
    )
