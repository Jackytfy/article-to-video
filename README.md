# Article → Video

Turn article text into shareable video: auto-fetched stock media + bilingual TTS narration + mood-matched BGM, rendered via FFmpeg.

[![CI](https://github.com/your-org/article-to-video/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/article-to-video/actions/workflows/ci.yml)

## Features

- **Pluggable NLP**: Ollama (default, free local), Claude/OpenAI, or pure-Python (jieba + regex)
- **3 media providers**: Pexels, Pixabay, Unsplash — searched in parallel, ranked by relevance + orientation + resolution
- **Free TTS** via Edge-TTS (Microsoft) with word-level timing for tight subtitle sync
- **Multilingual**: zh, en, ja, ko, es, fr, de, ru, pt, it, ar (and more) with bilingual mode
- **Mood-aware BGM**: local library or Jamendo API, ducked under narration
- **Multi-aspect**: 16:9 / 9:16 / 1:1 with smart center-crop
- **MoviePy 2.x + FFmpeg** rendering with optional NVENC GPU encoding
- **HTTP API**: FastAPI + SSE progress + MP4/SRT download
- **Docker**: single command stack (API + Ollama)

## Quickstart

```bash
git clone https://github.com/your-org/article-to-video.git
cd article-to-video

# Install uv: https://docs.astral.sh/uv/
uv venv .venv
uv pip install fastapi "uvicorn[standard]" "pydantic>=2.9" "pydantic-settings>=2.6" \
               moviepy ffmpeg-python imageio-ffmpeg pillow \
               edge-tts httpx mutagen python-dotenv \
               ollama jieba

cp .env.example .env
# Edit .env to add PEXELS_API_KEY (free at https://www.pexels.com/api/)

# CLI
uv run python -m scripts.run --article tests/fixtures/sample_zh.txt --aspect 9:16 --backend local

# API
uv run uvicorn app.main:app --reload
```

## CLI

```bash
python -m scripts.run \
  --article path/to/article.txt \
  --aspect 16:9 \                  # 16:9 | 9:16 | 1:1
  --backend ollama \               # ollama | llm | local | deepseek | qwen | zhipu | spark | wenxin
  --source-lang zh \
  --translate-to en \
  --no-bgm                         # disable music
```

## HTTP API

```
POST   /jobs                 → 202 + {"job_id": "..."}
GET    /jobs                 → list all jobs
GET    /jobs/{id}            → status + progress + mood
GET    /jobs/{id}/events     → SSE stream of state changes
GET    /jobs/{id}/download   → rendered MP4
GET    /jobs/{id}/srt        → SRT subtitles
GET    /health               → liveness probe
```

### Create job

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "article": "今天天气很好。",
    "aspect_ratio": "9:16",
    "nlp_backend": "ollama",
    "voice_primary": "zh-CN-XiaoxiaoNeural",
    "bgm_enabled": true
  }'
```

### Watch progress (SSE)

```bash
curl -N http://127.0.0.1:8000/jobs/<job-id>/events
```

```
event: progress
data: {"job_id":"...","status":"running","stage":"media_fetch","progress":0.34,...}

event: progress
data: {"job_id":"...","status":"running","stage":"tts","progress":0.55,...}

event: done
data: {}
```

### Download result

```bash
curl -o video.mp4 http://127.0.0.1:8000/jobs/<job-id>/download
```

## Env config (`.env`)

| Var | Default | Notes |
|-----|---------|-------|
| `NLP_BACKEND` | `ollama` | `ollama` \| `llm` \| `local` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:7b` | any chat-capable model |
| `LLM_MODEL` | `claude-sonnet-4-5` | when `NLP_BACKEND=llm` |
| `ANTHROPIC_API_KEY` | — | Claude API |
| `OPENAI_API_KEY` | — | OpenAI fallback |
| `PEXELS_API_KEY` | — | https://www.pexels.com/api/ |
| `PIXABAY_API_KEY` | — | https://pixabay.com/api/docs/ |
| `UNSPLASH_ACCESS_KEY` | — | https://unsplash.com/developers |
| `JAMENDO_CLIENT_ID` | — | https://developer.jamendo.com/v3.0 |
| `ASPECT_RATIO` | `16:9` | default for new jobs |
| `VOICE_PRIMARY` | `zh-CN-XiaoxiaoNeural` | full Edge-TTS voice id or short lang code |
| `VOICE_SECONDARY` | `en-US-AriaNeural` | for translated narration |
| `BGM_ENABLED` | `true` | global default |
| `BURN_SUBTITLES` | `true` | overlay vs sidecar |
| `USE_GPU` | `false` | NVENC if true |
| `CACHE_DIR` | `./assets/cache` | downloaded media |
| `OUTPUT_DIR` | `./assets/output` | rendered MP4s |

## BGM library

Drop royalty-free tracks into mood-tagged subdirs:

```
assets/bgm/
├── calm/
│   └── ambient_piano.mp3
├── energetic/
│   └── upbeat_drums.mp3
├── sad/
│   └── melancholic_strings.mp3
└── positive/
    └── happy_acoustic.mp3
```

Synonym dirs work too: `ambient/` → `calm`, `upbeat/` → `energetic`. Free
sources: Pixabay Music site, Free Music Archive, Incompetech, Bensound.

## Docker

```bash
docker compose up --build
# pull NLP model in ollama container:
docker exec -it a2v-ollama ollama pull qwen2.5:7b
```

API at `http://127.0.0.1:8000`. Outputs persisted in `./assets/output/`.

## Development

```bash
# Install dev tools
uv pip install pytest pytest-asyncio respx ruff mypy

# Run tests (skips real-network E2E in test_api.py is not auto-skipped — set PYTEST_ADDOPTS to opt out)
uv run pytest tests/ --tb=short

# Lint + format
uv run ruff check app scripts tests
uv run ruff format app scripts tests
```

### Test layers

| File | Speed | Network |
|------|-------|---------|
| `test_compose.py` | fast | none |
| `test_media.py` | fast | mocked (respx) |
| `test_music.py` | fast | none |
| `test_nlp.py` | fast | none (mocked Ollama) |
| `test_providers.py` | fast | mocked |
| `test_smoke.py` | fast | none |
| `test_subtitle.py` | fast | none |
| `test_tts.py` | fast | mocked Edge-TTS |
| `test_api.py` | slow (~60s) | hits real Edge-TTS + ffmpeg |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline diagram,
module map, and extension points.

## Troubleshooting

- **`No font file located`**: install Microsoft YaHei (Windows) or `fonts-noto-cjk` (Linux).
- **`No media providers configured`**: set at least `PEXELS_API_KEY` in `.env`.
- **`ollama._types.ResponseError`**: ensure Ollama running and model pulled (`ollama pull qwen2.5:7b`).
- **Subtitles burned in wrong place**: tweak `font_size` in `app/pipeline/compose/overlay.py` (default scales with `target_h // 24`).
- **BGM too loud**: reduce `bgm_gain` in orchestrator `_run_compose_and_render` (default `0.10` ≈ −20 dB).
- **Render slow**: enable `USE_GPU=true` if NVIDIA GPU + recent ffmpeg with NVENC.

## License

MIT (this project). All third-party licenses respected:

- Pexels: [Pexels License](https://www.pexels.com/license/)
- Pixabay: [Pixabay Content License](https://pixabay.com/service/license/)
- Unsplash: [Unsplash License](https://unsplash.com/license)
- Jamendo: per-track Creative Commons; check `track.license` field
- Edge-TTS: Microsoft Edge browser endpoint (free for non-commercial; check terms)
