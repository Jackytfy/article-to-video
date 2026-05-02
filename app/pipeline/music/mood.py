"""Mood → BGM tag/category mapping.

Mood labels come from NLPBackend.detect_mood (calm | energetic | sad | positive).
Tag map is kept generic so any provider (local lib, Jamendo, etc.) can use it.
"""
from __future__ import annotations

from typing import Final

# Canonical moods returned by NLP backends.
KNOWN_MOODS: Final[tuple[str, ...]] = ("calm", "energetic", "sad", "positive")

# Synonyms used as filesystem dir names + provider tags. Order = preference.
MOOD_TAGS: Final[dict[str, tuple[str, ...]]] = {
    "calm": ("calm", "ambient", "peaceful", "relaxing"),
    "energetic": ("energetic", "upbeat", "action", "uplifting"),
    "sad": ("sad", "melancholic", "emotional", "dramatic"),
    "positive": ("positive", "happy", "cheerful", "uplifting"),
}

# Fallback chain: if no track matches the primary mood, try these in order.
FALLBACK_CHAIN: Final[dict[str, tuple[str, ...]]] = {
    "calm": ("positive", "energetic"),
    "energetic": ("positive", "calm"),
    "sad": ("calm", "positive"),
    "positive": ("calm", "energetic"),
}


def normalize_mood(mood: str | None) -> str:
    """Coerce arbitrary input to one of KNOWN_MOODS."""
    if not mood:
        return "calm"
    m = mood.strip().lower()
    if m in KNOWN_MOODS:
        return m
    # Loose match against synonyms.
    for canonical, tags in MOOD_TAGS.items():
        if m in tags:
            return canonical
    return "calm"


def tags_for(mood: str) -> tuple[str, ...]:
    """Return search tags for `mood` (and its fallbacks)."""
    canonical = normalize_mood(mood)
    primary = MOOD_TAGS[canonical]
    fallbacks = tuple(
        tag
        for fb in FALLBACK_CHAIN[canonical]
        for tag in MOOD_TAGS[fb]
    )
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for tag in primary + fallbacks:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return tuple(out)


__all__ = [
    "KNOWN_MOODS",
    "MOOD_TAGS",
    "FALLBACK_CHAIN",
    "normalize_mood",
    "tags_for",
]
