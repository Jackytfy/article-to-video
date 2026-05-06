"""Jobs HTTP endpoints: POST /jobs, GET /jobs/{id}, SSE stream, list, download."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

from app.api.schemas import (
    JobCreate,
    JobCreatedResponse,
    JobListResponse,
    JobStatusResponse,
)
from app.jobs.store import get_store
from app.pipeline.models import Job, JobStatus
from app.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


def _job_to_status(job: Job) -> JobStatusResponse:
    output_url = (
        f"/jobs/{job.id}/download" if job.output_path is not None else None
    )
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        stage=job.stage.value,
        progress=job.progress,
        error=job.error,
        output_url=output_url,
        mood=job.mood,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(
    body: JobCreate, background: BackgroundTasks
) -> JobCreatedResponse:
    """Queue a new article-to-video job. Runs in background task."""
    job = Job(
        article=body.article,
        source_lang=body.source_lang,
        translate_to=body.translate_to,
        aspect_ratio=body.aspect_ratio,
        nlp_backend=body.nlp_backend,
        voice_primary=body.voice_primary,
        voice_secondary=body.voice_secondary,
        bgm_enabled=body.bgm_enabled,
        burn_subtitles=body.burn_subtitles,
    )
    store = get_store()
    await store.add(job)

    async def on_progress(j: Job) -> None:
        await store.notify(j.id)

    orchestrator = PipelineOrchestrator(on_progress=on_progress)
    background.add_task(_run_safely, orchestrator, job)

    return JobCreatedResponse(job_id=job.id)


@router.get("", response_model=JobListResponse)
async def list_jobs() -> JobListResponse:
    """List all jobs ordered by creation time (newest first)."""
    jobs = await get_store().list()
    return JobListResponse(jobs=[_job_to_status(j) for j in jobs])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    """Poll job status."""
    job = await get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_to_status(job)


@router.get("/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of job state changes until terminal."""
    store = get_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_source() -> AsyncIterator[bytes]:
        async for j in store.watch(job_id, poll_interval_s=1.0):
            payload = _job_to_status(j).model_dump(mode="json")
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n".encode()
            if j.status in (JobStatus.DONE, JobStatus.FAILED):
                yield b"event: done\ndata: {}\n\n"
                return

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/{job_id}/download")
async def download_video(job_id: str) -> FileResponse:
    """Download rendered MP4."""
    job = await get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(
            status_code=409, detail="video not ready for this job"
        )
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=f"{job.id}.mp4",
    )


@router.get("/{job_id}/srt")
async def download_srt(job_id: str) -> FileResponse:
    """Download SRT subtitle file (when available)."""
    job = await get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    # SRT lives next to the MP4 in the job's work dir.
    if job.output_path is None:
        raise HTTPException(status_code=409, detail="job not yet rendered")
    srt_path = job.output_path.parent / "captions.srt"
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="no subtitles for this job")
    return FileResponse(
        path=srt_path,
        media_type="application/x-subrip",
        filename=f"{job.id}.srt",
    )


async def _run_safely(orchestrator: PipelineOrchestrator, job: Job) -> None:
    try:
        await orchestrator.run(job)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Background job %s crashed", job.id)
    finally:
        await get_store().notify(job.id)


# Echoed for tests that need to stamp `now()` deterministically.
def _utcnow() -> datetime:
    return datetime.now(UTC)
