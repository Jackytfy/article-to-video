"""Media provider Protocol — implemented per stock service."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.pipeline.models import MediaAsset, MediaType, Orientation


@runtime_checkable
class MediaProvider(Protocol):
    """Stock media search interface."""

    name: str

    async def search(
        self,
        keywords: list[str],
        orientation: Orientation,
        media_type: MediaType,
        limit: int = 10,
    ) -> list[MediaAsset]:
        """Return ranked candidates. Caller picks + downloads."""
        ...
