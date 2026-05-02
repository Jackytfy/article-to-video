"""Pydantic request/response models for the HTTP layer."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AspectRatio = Literal["16:9", "9:16", "1:1"]
NLPBackendName = Literal["ollama", "llm", "local"]


class JobCreate(BaseModel):
    """POST /jobs body."""

    article: str = Field(..., min_length=1, description="Article body to convert.")
    source_lang: str = Field(default="zh")
    translate_to: str | None = Field(default=None)
    aspect_ratio: AspectRatio = "16:9"
    nlp_backend: NLPBackendName = "ollama"
    voice_primary: str = "zh-CN-XiaoxiaoNeural"
    voice_secondary: str | None = "en-US-AriaNeural"
    bgm_enabled: bool = True
    burn_subtitles: bool = True


class JobCreatedResponse(BaseModel):
    """POST /jobs response."""

    job_id: str


class JobStatusResponse(BaseModel):
    """GET /jobs/{id} response."""

    job_id: str
    status: str
    stage: str
    progress: float
    error: str | None = None
    output_url: str | None = None
    mood: str | None = None
    created_at: datetime
    updated_at: datetime


class JobListResponse(BaseModel):
    """GET /jobs response."""

    jobs: list[JobStatusResponse]
