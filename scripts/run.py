"""CLI driver: `python -m scripts.run --article path.txt --aspect 16:9`.

Phase 1: parses args, builds Job, runs orchestrator stub, prints status.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.pipeline.models import Job
from app.pipeline.orchestrator import PipelineOrchestrator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="a2v",
        description="Article to Video — turn text into a rendered MP4.",
    )
    parser.add_argument(
        "--article",
        type=Path,
        required=True,
        help="Path to article text file (UTF-8).",
    )
    parser.add_argument(
        "--aspect",
        choices=["16:9", "9:16", "1:1"],
        default=None,
        help="Output aspect ratio. Defaults from .env.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "llm", "local", "deepseek", "qwen", "zhipu", "spark", "wenxin"],
        default=None,
        help="NLP backend. Defaults from .env.",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help="Source language ISO code (e.g. zh, en).",
    )
    parser.add_argument(
        "--translate-to",
        default=None,
        help="Target language for bilingual narration. Empty disables.",
    )
    parser.add_argument(
        "--no-bgm",
        action="store_true",
        help="Disable background music.",
    )
    return parser.parse_args(argv)


async def _run(job: Job) -> Job:
    orchestrator = PipelineOrchestrator()
    return await orchestrator.run(job)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.article.exists():
        print(f"error: article file not found: {args.article}", file=sys.stderr)
        return 2

    settings = get_settings()
    article_text = args.article.read_text(encoding="utf-8").strip()
    if not article_text:
        print("error: article file is empty", file=sys.stderr)
        return 2

    job = Job(
        article=article_text,
        aspect_ratio=args.aspect or settings.aspect_ratio,
        source_lang=args.source_lang or settings.source_lang,
        translate_to=args.translate_to or settings.translate_to,
        nlp_backend=args.backend or settings.nlp_backend,
        voice_primary=settings.voice_primary,
        voice_secondary=settings.voice_secondary,
        bgm_enabled=not args.no_bgm and settings.bgm_enabled,
        burn_subtitles=settings.burn_subtitles,
    )

    result = asyncio.run(_run(job))

    print(
        json.dumps(
            {
                "job_id": result.id,
                "status": result.status.value,
                "stage": result.stage.value,
                "progress": result.progress,
                "error": result.error,
                "output": str(result.output_path) if result.output_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.status.value == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
