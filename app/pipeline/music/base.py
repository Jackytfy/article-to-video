"""Music provider Protocol — plug in different BGM sources behind one interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MusicTrack:
    """One BGM candidate ready to mix."""

    local_path: Path
    duration_s: float | None
    mood: str
    title: str
    license: str = "free"


@runtime_checkable
class MusicProvider(Protocol):
    """Pluggable BGM source: local lib, Jamendo, FMA, etc."""

    name: str

    async def find(
        self,
        mood: str,
        min_duration_s: float | None = None,
    ) -> MusicTrack | None:
        """Return one track matching `mood`, or None if no match."""
        ...
