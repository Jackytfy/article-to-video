"""Tests for compose stage: aspect math + orchestrator wiring with mocked compose/render."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.pipeline.compose.aspect import (
    plan_center_crop,
    target_dimensions,
)
from app.pipeline.models import Segment

# ============================================================================
# Aspect helpers
# ============================================================================


def test_target_dimensions_per_aspect() -> None:
    assert target_dimensions("16:9") == (1920, 1080)
    assert target_dimensions("9:16") == (1080, 1920)
    assert target_dimensions("1:1") == (1080, 1080)


def test_target_dimensions_unknown_raises() -> None:
    with pytest.raises(ValueError):
        target_dimensions("4:3")  # type: ignore[arg-type]


def test_plan_center_crop_landscape_to_landscape() -> None:
    plan = plan_center_crop(3840, 2160, "16:9")
    assert plan.target_w == 1920
    assert plan.target_h == 1080
    # 3840 -> 1920 means scale 0.5; height 2160 * 0.5 = 1080, perfect fit, no crop.
    assert plan.scale == pytest.approx(0.5)
    assert plan.crop_x == 0
    assert plan.crop_y == 0


def test_plan_center_crop_landscape_to_portrait() -> None:
    plan = plan_center_crop(1920, 1080, "9:16")
    assert plan.target_w == 1080
    assert plan.target_h == 1920
    # Need to scale up so height fills 1920: scale = 1920/1080 ≈ 1.778
    assert plan.scale > 1.5
    # After scaling, width = 1920*1.778 ≈ 3414; crop to 1080 keeps centered.
    scaled_w = round(1920 * plan.scale)
    assert plan.crop_x == (scaled_w - 1080) // 2


def test_plan_center_crop_portrait_to_landscape() -> None:
    plan = plan_center_crop(1080, 1920, "16:9")
    assert plan.target_w == 1920
    assert plan.target_h == 1080
    # Scale must fill width 1920: scale = 1920/1080 ≈ 1.778
    assert plan.scale > 1.5
    # After scaling, height = 1920*1.778 ≈ 3414; crop center vertically.
    scaled_h = round(1920 * plan.scale)
    assert plan.crop_y == (scaled_h - 1080) // 2
    assert plan.crop_x == 0


def test_plan_center_crop_invalid_dims() -> None:
    with pytest.raises(ValueError):
        plan_center_crop(0, 100, "16:9")


# ============================================================================
# Orchestrator compose+render stage (mocked)
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_compose_and_render_called(
    tmp_path: Path, stub_tts
) -> None:
    """Compose+render stage is invoked with expected args; output_path set."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.orchestrator import PipelineOrchestrator

    fake_clip = MagicMock(name="composed_clip")
    fake_clip.close = MagicMock()
    compose_fn = MagicMock(return_value=fake_clip, name="compose_fn")
    rendered_path: list[Path] = []

    def render_fn(clip, output_path, *, fps, use_gpu, threads):
        rendered_path.append(Path(output_path))
        Path(output_path).write_bytes(b"FAKE-MP4")
        return Path(output_path)

    class StubNLP:
        async def segment(self, article: str, target_seconds_per_seg: int = 8):
            return [
                Segment(index=0, text="Sunrise.", keywords=("sunrise",)),
                Segment(index=1, text="Mountain.", keywords=("mountain",)),
            ]

        async def keywords(self, segment: Segment, top_k: int = 5):
            return list(segment.keywords)

        async def translate(self, text: str, source_lang: str, target_lang: str):
            return text

        async def detect_mood(self, article: str):
            return "calm"

    job = Job(article="Sunrise. Mountain.", aspect_ratio="9:16")
    orchestrator = PipelineOrchestrator(
        nlp=StubNLP(),
        media_providers=[],
        tts=stub_tts,
        work_dir=tmp_path,
        compose_fn=compose_fn,
        render_fn=render_fn,
    )
    result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.output_path.read_bytes() == b"FAKE-MP4"

    # compose_fn invoked with the right shape.
    compose_fn.assert_called_once()
    args, _ = compose_fn.call_args
    assert args[0] == result.segments  # segments
    assert args[4] == "9:16"           # aspect ratio
    assert args[5] is True             # burn_subtitles default

    # clip.close() called for cleanup.
    fake_clip.close.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_render_failure_marks_job_failed(
    tmp_path: Path, stub_tts
) -> None:
    """If render raises, Job status becomes FAILED with error message."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.orchestrator import PipelineOrchestrator

    fake_clip = MagicMock(close=MagicMock())
    compose_fn = MagicMock(return_value=fake_clip)

    def render_fn(clip, output_path, **kwargs):
        raise RuntimeError("ffmpeg exploded")

    class StubNLP:
        async def segment(self, article: str, target_seconds_per_seg: int = 8):
            return [Segment(index=0, text="Hello.", keywords=("hello",))]

        async def keywords(self, segment: Segment, top_k: int = 5):
            return list(segment.keywords)

        async def translate(self, text: str, source_lang: str, target_lang: str):
            return text

        async def detect_mood(self, article: str):
            return "calm"

    orchestrator = PipelineOrchestrator(
        nlp=StubNLP(),
        media_providers=[],
        tts=stub_tts,
        work_dir=tmp_path,
        compose_fn=compose_fn,
        render_fn=render_fn,
    )
    result = await orchestrator.run(Job(article="Hello."))

    assert result.status is JobStatus.FAILED
    assert "ffmpeg exploded" in (result.error or "")
    # Clip still closed even though render raised.
    fake_clip.close.assert_called_once()
