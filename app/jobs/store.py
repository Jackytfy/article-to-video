"""Async-safe in-memory job store + per-job event bus for SSE."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.pipeline.models import Job, JobStatus


class JobStore:
    """Async-safe map of jobs + per-job change notifier for SSE streaming."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._notifiers: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def add(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.id] = job
            self._notifiers.setdefault(job.id, asyncio.Event())
        return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list(self) -> list[Job]:
        async with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True,
            )

    async def notify(self, job_id: str) -> None:
        """Signal subscribers that the job has changed state."""
        async with self._lock:
            event = self._notifiers.get(job_id)
        if event is None:
            return
        event.set()
        # Re-arm so the next change wakes subscribers again.
        async with self._lock:
            self._notifiers[job_id] = asyncio.Event()

    async def watch(
        self, job_id: str, *, poll_interval_s: float = 1.0
    ) -> AsyncIterator[Job]:
        """Yield the job each time it changes; ends when status terminal.

        First yield is the current state; subsequent yields fire when notify()
        is called or every `poll_interval_s` seconds (heartbeat).
        """
        while True:
            job = await self.get(job_id)
            if job is None:
                return
            yield job
            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                return
            async with self._lock:
                event = self._notifiers.setdefault(job_id, asyncio.Event())
            try:
                await asyncio.wait_for(event.wait(), timeout=poll_interval_s)
            except TimeoutError:
                pass


_store: JobStore | None = None


def get_store() -> JobStore:
    """Singleton accessor for the in-memory store."""
    global _store
    if _store is None:
        _store = JobStore()
    return _store


def reset_store() -> None:
    """Test helper: drops the global store."""
    global _store
    _store = None
