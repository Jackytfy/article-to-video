"""Phase 1 smoke tests: imports + FastAPI app boots + orchestrator stub runs."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_app_imports() -> None:
    """Importing the FastAPI app must not raise."""
    from app.main import app

    assert app.title == "Article to Video"


def test_health_endpoint() -> None:
    """GET /health returns 200 with version + backend info."""
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "nlp_backend" in body


def test_pipeline_models_importable() -> None:
    """Shared dataclasses must construct cleanly."""
    from app.pipeline.models import Job, JobStatus, MediaAsset, Segment

    seg = Segment(index=0, text="hello world")
    assert seg.index == 0

    asset = MediaAsset(
        provider="pexels",
        media_type="image",
        url="https://example.com/x.jpg",
        width=1920,
        height=1080,
    )
    assert asset.orientation == "landscape"

    job = Job(article="lorem ipsum")
    assert job.status is JobStatus.QUEUED
    assert len(job.id) > 0


def test_protocols_importable() -> None:
    """Protocol modules must import without optional deps."""
    from app.pipeline.media.base import MediaProvider
    from app.pipeline.nlp.base import NLPBackend

    assert NLPBackend is not None
    assert MediaProvider is not None


@pytest.mark.asyncio
async def test_orchestrator_runs_with_local_backend(
    tmp_path, stub_tts, stub_compose, stub_render
) -> None:
    """Orchestrator end-to-end (Phase 5: NLP+TTS+subtitle+compose+render mocked)."""
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.nlp.local import LocalNLPBackend
    from app.pipeline.orchestrator import PipelineOrchestrator

    job = Job(
        article="The sun rose over the hills. Birds sang their morning songs.",
        nlp_backend="local",
    )
    result = await PipelineOrchestrator(
        nlp=LocalNLPBackend(),
        media_providers=[],
        tts=stub_tts,
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    ).run(job)
    assert result.status is JobStatus.DONE
    assert result.progress == 1.0
    assert len(result.segments) >= 1
    assert result.output_path is not None
    assert result.output_path.exists()


def test_create_job_endpoint() -> None:
    """POST /jobs accepts an article and returns 202 + job_id."""
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/jobs",
        json={"article": "Phase 1 smoke article."},
    )
    assert response.status_code == 202
    assert "job_id" in response.json()
