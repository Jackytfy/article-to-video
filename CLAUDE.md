# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Package manager is **uv** (not pip directly). Python `>=3.11`.

```bash
# First-time setup
uv venv .venv
uv pip install fastapi "uvicorn[standard]" "pydantic>=2.9" "pydantic-settings>=2.6" \
               moviepy ffmpeg-python imageio-ffmpeg pillow \
               edge-tts httpx mutagen python-dotenv ollama jieba
uv pip install pytest pytest-asyncio respx ruff mypy   # dev tools

# Run API (auto-reload)
uv run uvicorn app.main:app --reload

# CLI entry
uv run python -m scripts.run --article tests/fixtures/sample_zh.txt --aspect 9:16 --backend local

# Tests
uv run pytest tests/ --tb=short
uv run pytest tests/test_nlp.py::test_specific_thing   # single test
uv run pytest -m real_music_factory                    # opt into live Jamendo factory tests

# Lint / format / types
uv run ruff check app scripts tests
uv run ruff format app scripts tests
uv run mypy app                                        # strict mode

# Docker stack (API + Ollama)
docker compose up --build
docker exec -it a2v-ollama ollama pull qwen2.5:7b      # one-time model pull
```

## Architecture

Stage pipeline `Article → NLP → Media → TTS → Subtitle → Music → Compose → Render → MP4+SRT`. Full diagram in `docs/ARCHITECTURE.md`.

### Pluggable backends (Protocol pattern)

Every external integration is behind a Protocol:

| Protocol | Implementations | Selected by |
|----------|-----------------|-------------|
| `NLPBackend` (`app/pipeline/nlp/base.py`) | `ollama.py`, `llm.py` (Anthropic→OpenAI), `local.py` (jieba+regex) | `NLP_BACKEND` env |
| `MediaProvider` (`app/pipeline/media/base.py`) | `pexels.py`, `pixabay.py`, `unsplash.py` | API key env presence |
| `MusicProvider` (`app/pipeline/music/base.py`) | `library.py` (filesystem), `jamendo.py` | always-on local + key-gated jamendo |

Each subpackage's `__init__.py` exposes a `make_providers(settings)` / `make_backend(settings)` factory. **To add a backend: implement Protocol + register in factory** — no orchestrator changes.

### Orchestrator (`app/pipeline/orchestrator.py`)

`PipelineOrchestrator.run(job)` runs stages sequentially, mutates `Job` state, calls `on_progress` callback after each stage. Per-segment state (assets, TTS results) kept on instance, keyed by `segment.index`. Compose + render run inside `asyncio.to_thread` to avoid blocking event loop.

Stage→progress mapping (orchestrator constants): NLP 0.05–0.15, media 0.20–0.40, TTS 0.45–0.60, subtitle 0.70, music 0.78, compose 0.82, render 0.92, complete 1.00.

### Job state + SSE bus (`app/jobs/store.py`)

In-memory async map. `JobStore.notify(id)` flips an `asyncio.Event` per job; SSE endpoint subscribes via `JobStore.watch(id)` async generator, emits heartbeats on poll timeout to keep proxies alive. **No DB** — jobs lost on restart.

### Lazy heavy imports

`moviepy` and `ollama` are imported **inside functions**, not at module top. This keeps tests offline-fast and prod startup snappy. Preserve this — do not hoist these to top of file.

### Config (`app/config.py`)

`Settings` (pydantic-settings) loaded from `.env`. Cached via `lru_cache` — **changes to env require process restart**. `cache_dir` / `output_dir` auto-`mkdir` in validator. `translate_to=""` coerces to `None`.

## Testing

`tests/conftest.py` provides three stubs to keep tests fast/offline:

| Fixture | Replaces | Output |
|---------|----------|--------|
| `stub_tts` | `EdgeTTS.synthesize*` | empty MP3 + synthetic 100ms/char timing |
| `stub_compose` | MoviePy compose | `MagicMock` clip |
| `stub_render` | FFmpeg render | placeholder MP4 bytes |

**Auto-applied fixture `_no_real_music_providers`** monkey-patches `make_music_providers` to `[]` so tests never hit live Jamendo (even when `JAMENDO_CLIENT_ID` is set in dev `.env`). Tests that verify the real factory must opt out with `@pytest.mark.real_music_factory`. Tests that test the music stage inject `music_providers=` directly on the orchestrator.

`test_api.py` is the slow lane (~60s) — hits real Edge-TTS endpoint + ffmpeg. All others are fast/mocked.

## Config invariants

- At least one of `PEXELS_API_KEY` / `PIXABAY_API_KEY` / `UNSPLASH_ACCESS_KEY` required for media stage.
- `NLP_BACKEND=ollama` (default) requires Ollama running at `OLLAMA_HOST` with `OLLAMA_MODEL` pulled. Use `local` backend for offline/CI runs.
- CJK subtitle rendering: Linux needs `fonts-noto-cjk` (in Dockerfile); Windows needs Microsoft YaHei. Font resolver lives in `app/pipeline/compose/overlay.py`.
- BGM library uses mood-tagged subdirs under `assets/bgm/` — synonyms (e.g. `ambient/` → `calm`) handled in `app/pipeline/music/mood.py`.

## Style

- Ruff: line-length 100, `select = [E,F,I,N,W,UP,B,ASYNC]`, `E501` ignored.
- Mypy strict on `app/`. Keep type annotations on public functions.
- Async-first: all I/O stages are coroutines. CPU-heavy ops (compose/render) wrapped in `asyncio.to_thread`.
