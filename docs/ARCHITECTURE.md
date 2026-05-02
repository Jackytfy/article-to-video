# Architecture

## Pipeline overview

```
Article text
     │
     ▼
┌────────────────────┐
│  NLP backend       │  ← pluggable: ollama | llm | local
│  segment + keywords│
│  + mood + translate│
└────────┬───────────┘
         │ Segment[]
         ▼
┌────────────────────┐
│  Media providers   │  ← pluggable: pexels | pixabay | unsplash
│  search per seg    │     run in parallel via asyncio.gather
│  rank + cache      │
└────────┬───────────┘
         │ MediaAsset per Segment
         ▼
┌────────────────────┐
│  Edge-TTS          │  ← Microsoft free TTS
│  per-seg synth     │     parallel synthesize, semaphore-bounded
│  + WordBoundary    │
└────────┬───────────┘
         │ TTSResult per Segment (mp3 + word timing)
         ▼
┌────────────────────┐
│  SRT generator     │  pure-Python, zh char grouping
└────────┬───────────┘
         │ SubtitleCue[]
         ▼
┌────────────────────┐
│  Music providers   │  ← pluggable: local lib | jamendo
│  mood -> tag       │     fallback chain
└────────┬───────────┘
         │ MusicTrack (mp3)
         ▼
┌────────────────────┐
│  MoviePy compose   │  ImageClip / VideoFileClip per seg
│  + aspect crop     │  Ken Burns ready, BGM duck mix
│  + subtitle layer  │  TextClip with stroke
└────────┬───────────┘
         │ CompositeVideoClip
         ▼
┌────────────────────┐
│  FFmpeg render     │  libx264 / NVENC, AAC audio, yuv420p
└────────┬───────────┘
         ▼
       MP4 + SRT
```

## Module map

```
app/
├── api/                  # FastAPI HTTP layer
│   ├── jobs.py           # POST/GET/SSE/download endpoints
│   └── schemas.py        # Pydantic v2 request/response
├── jobs/
│   └── store.py          # Async job map + event bus
├── pipeline/
│   ├── orchestrator.py   # PipelineOrchestrator, stage runner
│   ├── models.py         # Segment / MediaAsset / Job / JobStage / JobStatus
│   ├── nlp/
│   │   ├── base.py       # NLPBackend Protocol
│   │   ├── ollama.py     # Default; structured JSON output
│   │   ├── llm.py        # Anthropic-pref, OpenAI fallback
│   │   └── local.py      # jieba + regex sentence split + lexicon mood
│   ├── media/
│   │   ├── base.py       # MediaProvider Protocol
│   │   ├── pexels.py     # image + video, header auth
│   │   ├── pixabay.py    # image + video, query-key auth
│   │   ├── unsplash.py   # image only, Client-ID auth
│   │   ├── cache.py      # SHA256-keyed disk cache
│   │   └── ranker.py     # score = overlap + orientation + resolution + video bonus
│   ├── tts/
│   │   ├── voices.py     # lang -> voice map (zh/en/ja/ko/...)
│   │   └── edge_tts.py   # Communicate stream + WordBoundary capture
│   ├── subtitle/
│   │   └── srt.py        # build_cues + render_srt + write_srt
│   ├── music/
│   │   ├── base.py       # MusicProvider Protocol
│   │   ├── mood.py       # KNOWN_MOODS + tags_for + FALLBACK_CHAIN
│   │   ├── library.py    # LocalMusicLibrary (filesystem)
│   │   └── jamendo.py    # Jamendo API client
│   ├── compose/
│   │   ├── aspect.py     # target dims + plan_center_crop (cover-fit)
│   │   ├── overlay.py    # subtitle TextClip builder + font resolver
│   │   └── timeline.py   # compose_video assembly
│   └── render/
│       └── ffmpeg.py     # x264 / NVENC encoding profiles
├── config.py             # pydantic-settings (env-driven)
└── main.py               # FastAPI app factory
```

## Key design decisions

### Pluggable backends via Protocol

Every external integration is behind a Protocol (`NLPBackend`, `MediaProvider`,
`MusicProvider`). Factory functions in each subpackage's `__init__.py` build a
list based on env config. Adding a new provider = one file + factory entry.

### Lazy heavy imports

MoviePy and ollama are imported inside functions, not at module top. Tests run
without these libs. Prod startup stays fast.

### Async-first

All stages are coroutines. The orchestrator runs MoviePy compose + FFmpeg
render under `asyncio.to_thread` to avoid blocking the event loop. Background
job execution uses FastAPI `BackgroundTasks`.

### Job event bus

`JobStore.notify(id)` flips an `asyncio.Event` per job. SSE endpoint subscribes
via `JobStore.watch(id)` async generator. Heartbeat on poll timeout keeps
proxies alive.

### Test stubs

`tests/conftest.py` exports `stub_tts`, `stub_compose`, `stub_render`. Tests
that don't care about real synthesis/rendering inject these to keep runs fast
(<10s) and offline.

## Extension points

| To add | Implement | Register |
|--------|-----------|----------|
| New NLP backend | NLPBackend Protocol | `app/pipeline/nlp/__init__.py` factory |
| New media provider | MediaProvider Protocol + optional `search_both` | `app/pipeline/media/__init__.py` factory |
| New music provider | MusicProvider Protocol | `app/pipeline/music/__init__.py` factory |
| New aspect ratio | `_TARGET_DIMS` + provider orientation map | `app/pipeline/compose/aspect.py` |
| New render profile | `render_clip` codec branch | `app/pipeline/render/ffmpeg.py` |

## Stage progress mapping

| Progress | Stage |
|----------|-------|
| 0.05 | NLP starting |
| 0.15 | NLP done |
| 0.20 | Media fetch starting |
| 0.40 | Media fetch done |
| 0.45 | TTS starting |
| 0.60 | TTS done |
| 0.70 | Subtitle starting |
| 0.78 | Music starting |
| 0.82 | Compose starting |
| 0.92 | Render starting |
| 1.00 | Complete |
