"""Local filesystem BGM library.

Layout convention: `<root>/<mood>/*.mp3` where mood is one of the canonical
labels from `mood.py`. Synonyms also accepted (e.g., `ambient/` resolves to
`calm`).

Track durations probed lazily via mutagen if installed; otherwise None.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from app.pipeline.music.base import MusicProvider, MusicTrack
from app.pipeline.music.mood import MOOD_TAGS, normalize_mood, tags_for

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _probe_duration(path: Path) -> float | None:
    try:
        from mutagen import File  # type: ignore

        meta = File(str(path))
        if meta is not None and meta.info is not None:
            length = getattr(meta.info, "length", None)
            return float(length) if length else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Duration probe failed for %s: %s", path, exc)
    return None


class LocalMusicLibrary(MusicProvider):
    """Reads BGM tracks from a directory tree on disk."""

    name = "local"

    def __init__(self, root: Path, *, rng: random.Random | None = None) -> None:
        self._root = root
        self._rng = rng or random.Random()

    async def find(
        self,
        mood: str,
        min_duration_s: float | None = None,
    ) -> MusicTrack | None:
        if not self._root.exists():
            logger.info("BGM library missing: %s", self._root)
            return None

        canonical = normalize_mood(mood)
        for tag in tags_for(canonical):
            tracks = self._list_tracks(tag)
            if not tracks:
                continue
            picks = [t for t in tracks if self._satisfies_min_duration(t, min_duration_s)]
            if not picks:
                picks = tracks  # Take what we have rather than nothing.
            chosen = self._rng.choice(picks)
            logger.info(
                "BGM picked: %s (mood=%s, tag=%s)", chosen.local_path.name, canonical, tag
            )
            return chosen

        logger.info("No BGM track matched mood=%s in %s", canonical, self._root)
        return None

    # ---- Helpers ------------------------------------------------------------

    def _list_tracks(self, tag: str) -> list[MusicTrack]:
        directory = self._root / tag
        if not directory.is_dir():
            return []

        canonical = self._canonical_for_dir(tag)

        return [
            MusicTrack(
                local_path=p,
                duration_s=_probe_duration(p),
                mood=canonical,
                title=p.stem,
            )
            for p in sorted(directory.iterdir())
            if p.suffix.lower() in _AUDIO_EXTS and p.is_file()
        ]

    @staticmethod
    def _canonical_for_dir(dir_name: str) -> str:
        for canonical, tags in MOOD_TAGS.items():
            if dir_name in tags:
                return canonical
        return dir_name

    @staticmethod
    def _satisfies_min_duration(
        track: MusicTrack, min_duration_s: float | None
    ) -> bool:
        if min_duration_s is None:
            return True
        if track.duration_s is None:
            return True  # Unknown duration: assume OK rather than reject.
        return track.duration_s + 0.5 >= min_duration_s
