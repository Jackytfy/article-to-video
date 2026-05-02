"""Shared dataclasses used across pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class Segment:
    """One narrated chunk of the article (~5-10 seconds of speech)."""

    index: int
    text: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    translation: str | None = None
    duration_s: float | None = None  # filled after TTS


MediaType = Literal["image", "video"]
Orientation = Literal["landscape", "portrait", "square"]


@dataclass(frozen=True)
class MediaAsset:
    """Stock media candidate or downloaded file."""

    provider: str
    media_type: MediaType
    url: str
    width: int
    height: int
    duration_s: float | None = None
    license: str = "free"
    tags: tuple[str, ...] = field(default_factory=tuple)
    local_path: Path | None = None  # set after download

    @property
    def orientation(self) -> Orientation:
        if self.width > self.height:
            return "landscape"
        if self.height > self.width:
            return "portrait"
        return "square"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStage(StrEnum):
    PENDING = "pending"
    NLP = "nlp"
    MEDIA_FETCH = "media_fetch"
    TTS = "tts"
    SUBTITLE = "subtitle"
    MUSIC = "music"
    COMPOSE = "compose"
    RENDER = "render"
    COMPLETE = "complete"


@dataclass
class Job:
    """A pipeline run from article -> rendered video."""

    article: str
    aspect_ratio: str = "16:9"
    source_lang: str = "zh"
    translate_to: str | None = None
    nlp_backend: str = "ollama"
    voice_primary: str = "zh-CN-XiaoxiaoNeural"
    voice_secondary: str | None = "en-US-AriaNeural"
    bgm_enabled: bool = True
    burn_subtitles: bool = True

    id: str = field(default_factory=_new_id)
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.PENDING
    progress: float = 0.0
    error: str | None = None
    output_path: Path | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    # populated as pipeline runs
    segments: list[Segment] = field(default_factory=list)
    mood: str | None = None

    def touch(self) -> None:
        self.updated_at = _now()
