"""Tests for NLP backends.

Strategy:
- Local backend: real run (no external deps required, jieba optional).
- Ollama backend: mocked AsyncClient.chat.
- LLM backend: mocked Anthropic/OpenAI clients.
- Factory: ensures correct dispatch on backend name.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.pipeline.models import Segment

# ============================================================================
# Local backend (no mocks: pure-Python impl)
# ============================================================================


@pytest.mark.asyncio
async def test_local_backend_segments_chinese() -> None:
    from app.pipeline.nlp.local import LocalNLPBackend

    article = (
        "今天天气很好，阳光明媚。"
        "我们去公园散步，看见小鸟在唱歌。"
        "孩子们在草地上奔跑，笑声不断。"
        "傍晚时分，夕阳染红了天空。"
    )
    backend = LocalNLPBackend()
    segments = await backend.segment(article, target_seconds_per_seg=4)

    assert len(segments) >= 2
    assert all(isinstance(s, Segment) for s in segments)
    assert all(s.text for s in segments)
    # Indices should be 0..N-1.
    assert [s.index for s in segments] == list(range(len(segments)))


@pytest.mark.asyncio
async def test_local_backend_segments_english() -> None:
    from app.pipeline.nlp.local import LocalNLPBackend

    article = (
        "The sun rose over the mountains. Birds began their morning songs. "
        "A gentle breeze whispered through the trees. Children laughed in the distance."
    )
    backend = LocalNLPBackend()
    segments = await backend.segment(article, target_seconds_per_seg=5)

    assert len(segments) >= 1
    assert all(s.text for s in segments)


@pytest.mark.asyncio
async def test_local_backend_mood_detection() -> None:
    from app.pipeline.nlp.local import LocalNLPBackend

    backend = LocalNLPBackend()
    happy = await backend.detect_mood("This is amazing! Wonderful joy and love.")
    sad = await backend.detect_mood("Sad memories. The loss was tragic.")
    assert happy in {"positive", "energetic"}
    assert sad == "sad"


@pytest.mark.asyncio
async def test_local_backend_translate_is_passthrough() -> None:
    from app.pipeline.nlp.local import LocalNLPBackend

    backend = LocalNLPBackend()
    out = await backend.translate("hello", "en", "zh")
    assert out == "hello"  # local backend has no translation


# ============================================================================
# Ollama backend (mocked client)
# ============================================================================


@pytest.mark.asyncio
async def test_ollama_backend_segment_parses_json() -> None:
    from app.pipeline.nlp.ollama import OllamaNLPBackend

    fake_response = {
        "message": {
            "content": json.dumps(
                {
                    "segments": [
                        {"text": "First chunk.", "keywords": ["sunset", "mountain"]},
                        {"text": "Second chunk.", "keywords": ["river", "valley"]},
                    ]
                }
            )
        }
    }

    with patch(
        "app.pipeline.nlp.ollama.AsyncClient", autospec=True
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat = AsyncMock(return_value=fake_response)

        backend = OllamaNLPBackend(host="http://x", model="qwen2.5:7b")
        segments = await backend.segment("Article text.")

    assert len(segments) == 2
    assert segments[0].text == "First chunk."
    assert segments[0].keywords == ("sunset", "mountain")
    assert segments[1].index == 1


@pytest.mark.asyncio
async def test_ollama_backend_keywords_returns_cached() -> None:
    """If a Segment already has keywords, no LLM call needed."""
    from app.pipeline.nlp.ollama import OllamaNLPBackend

    with patch(
        "app.pipeline.nlp.ollama.AsyncClient", autospec=True
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat = AsyncMock()

        backend = OllamaNLPBackend(host="http://x", model="m")
        seg = Segment(index=0, text="text", keywords=("a", "b", "c"))
        kws = await backend.keywords(seg, top_k=2)

    assert kws == ["a", "b"]
    mock_client.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_ollama_backend_translate_skips_when_same_lang() -> None:
    from app.pipeline.nlp.ollama import OllamaNLPBackend

    with patch(
        "app.pipeline.nlp.ollama.AsyncClient", autospec=True
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat = AsyncMock()

        backend = OllamaNLPBackend(host="http://x", model="m")
        out = await backend.translate("hello", "en", "en")

    assert out == "hello"
    mock_client.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_ollama_backend_detect_mood_parses_enum() -> None:
    from app.pipeline.nlp.ollama import OllamaNLPBackend

    fake_response = {
        "message": {"content": json.dumps({"mood": "energetic"})}
    }
    with patch(
        "app.pipeline.nlp.ollama.AsyncClient", autospec=True
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat = AsyncMock(return_value=fake_response)

        backend = OllamaNLPBackend(host="http://x", model="m")
        mood = await backend.detect_mood("article body")

    assert mood == "energetic"


# ============================================================================
# Factory
# ============================================================================


def test_factory_unknown_backend_raises() -> None:
    from app.pipeline.nlp import make_backend

    with pytest.raises(ValueError, match="Unknown NLP backend"):
        make_backend(backend="bogus")


def test_factory_local_does_not_require_keys() -> None:
    from app.pipeline.nlp import make_backend
    from app.pipeline.nlp.local import LocalNLPBackend

    backend = make_backend(backend="local")
    assert isinstance(backend, LocalNLPBackend)


def test_factory_llm_raises_without_keys() -> None:
    from app.config import Settings
    from app.pipeline.nlp import make_backend

    settings = Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        nlp_backend="llm",
    )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY or OPENAI_API_KEY"):
        make_backend(backend="llm", settings=settings)


# ============================================================================
# Orchestrator with NLP wired (uses LocalBackend - no external deps)
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_runs_nlp_stage_with_local_backend(
    tmp_path, stub_tts, stub_compose, stub_render
) -> None:
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.nlp.local import LocalNLPBackend
    from app.pipeline.orchestrator import PipelineOrchestrator

    article = (
        "The morning sun warmed the valley. Children played in fields of flowers. "
        "Their laughter echoed through the hills."
    )
    job = Job(article=article, nlp_backend="local")
    orchestrator = PipelineOrchestrator(
        nlp=LocalNLPBackend(),
        media_providers=[],
        tts=stub_tts,
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    )
    result = await orchestrator.run(job)

    assert result.status is JobStatus.DONE
    assert len(result.segments) >= 1
    assert result.mood is not None
    assert result.progress == 1.0


@pytest.mark.asyncio
async def test_orchestrator_translates_when_target_lang_given(
    tmp_path, stub_tts, stub_compose, stub_render
) -> None:
    from app.pipeline.models import Job, JobStatus
    from app.pipeline.nlp.base import NLPBackend  # noqa: F401
    from app.pipeline.orchestrator import PipelineOrchestrator

    class StubNLP:
        async def segment(
            self, article: str, target_seconds_per_seg: int = 8
        ) -> list[Segment]:
            return [Segment(index=0, text="hello", keywords=("greeting",))]

        async def keywords(self, segment: Segment, top_k: int = 5) -> list[str]:
            return list(segment.keywords)

        async def translate(
            self, text: str, source_lang: str, target_lang: str
        ) -> str:
            return f"[{target_lang}]{text}"

        async def detect_mood(self, article: str) -> str:
            return "calm"

    job = Job(article="hello", source_lang="en", translate_to="zh")
    result = await PipelineOrchestrator(
        nlp=StubNLP(),
        media_providers=[],
        tts=stub_tts,
        work_dir=tmp_path,
        compose_fn=stub_compose,
        render_fn=stub_render,
    ).run(job)

    assert result.status is JobStatus.DONE
    assert result.segments[0].translation == "[zh]hello"
