"""API integration tests: full job lifecycle, list, status, SSE, download."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.jobs.store import reset_store


@pytest.fixture(autouse=True)
def _isolated_store():
    """Each test starts with a fresh in-memory job store."""
    reset_store()
    yield
    reset_store()


def test_index_route_serves_html() -> None:
    """GET / returns the static UI."""
    from app.main import app

    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Article" in resp.text


def test_static_assets_served() -> None:
    """GET /static/style.css and /static/app.js return 200."""
    from app.main import app

    client = TestClient(app)
    css = client.get("/static/style.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "/jobs" in js.text  # sanity: js does talk to API


def test_create_job_returns_202_and_id() -> None:
    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/jobs",
        json={"article": "Sunrise over the hills.", "nlp_backend": "local"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert isinstance(job_id, str) and len(job_id) > 0


def test_get_job_404_when_missing() -> None:
    from app.main import app

    client = TestClient(app)
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_list_jobs_returns_created_jobs() -> None:
    from app.main import app

    client = TestClient(app)
    a = client.post("/jobs", json={"article": "Job A.", "nlp_backend": "local"}).json()
    b = client.post("/jobs", json={"article": "Job B.", "nlp_backend": "local"}).json()

    listing = client.get("/jobs").json()
    ids = [j["job_id"] for j in listing["jobs"]]
    assert a["job_id"] in ids
    assert b["job_id"] in ids


def test_download_state_after_creation() -> None:
    """Download endpoint returns 200 (when render done) or 409 (still running).
    404 reserved for unknown job ids — covered separately.
    """
    from app.main import app

    client = TestClient(app)
    job_id = client.post(
        "/jobs", json={"article": "x", "nlp_backend": "local"}
    ).json()["job_id"]

    resp = client.get(f"/jobs/{job_id}/download")
    assert resp.status_code in (200, 409)


def test_download_unknown_job_returns_404() -> None:
    from app.main import app

    client = TestClient(app)
    resp = client.get("/jobs/no-such-id/download")
    assert resp.status_code == 404


def test_status_response_shape() -> None:
    from app.main import app

    client = TestClient(app)
    job_id = client.post(
        "/jobs", json={"article": "Hello.", "nlp_backend": "local"}
    ).json()["job_id"]

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert "status" in body
    assert "stage" in body
    assert "progress" in body
    assert "created_at" in body
    assert "updated_at" in body


# ============================================================================
# JobStore + watcher unit tests (the layer SSE relies on)
# ============================================================================


@pytest.mark.asyncio
async def test_jobstore_watch_emits_until_terminal() -> None:
    from app.jobs.store import JobStore
    from app.pipeline.models import Job, JobStatus

    store = JobStore()
    job = Job(article="x")
    await store.add(job)

    received: list[str] = []

    async def consume() -> None:
        async for j in store.watch(job.id, poll_interval_s=0.01):
            received.append(j.status.value)

    import asyncio

    consumer = asyncio.create_task(consume())

    # Initial yield comes immediately; let the watcher run once.
    await asyncio.sleep(0.05)
    job.status = JobStatus.RUNNING
    await store.notify(job.id)
    await asyncio.sleep(0.05)
    job.status = JobStatus.DONE
    await store.notify(job.id)

    await asyncio.wait_for(consumer, timeout=2.0)

    # We saw queued (initial), running, done.
    assert "done" in received
    assert received[-1] == "done"


@pytest.mark.asyncio
async def test_jobstore_watch_unknown_id_returns_immediately() -> None:
    from app.jobs.store import JobStore

    store = JobStore()
    seen: list = []
    async for j in store.watch("nope", poll_interval_s=0.01):
        seen.append(j)
    assert seen == []
