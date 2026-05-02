"""FastAPI application entrypoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api.jobs import router as jobs_router
from app.config import get_settings

settings = get_settings()

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"
_TEMPLATES_DIR = _BASE_DIR / "templates"

app = FastAPI(
    title="Article to Video",
    version=__version__,
    description="Turn article text into rendered video via stock media + TTS + BGM.",
)

app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

templates = Jinja2Templates(directory=_TEMPLATES_DIR) if _TEMPLATES_DIR.exists() else None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    """Serve the static UI. Falls back to a tiny placeholder if templates dir missing."""
    if templates is None:
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8>"
            "<title>Article to Video</title>"
            "<p>UI templates not found. See <code>app/templates/</code>.</p>"
        )
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": __version__,
        "nlp_backend": settings.nlp_backend,
    }
