"""Pipeline orchestrator: runs stages in order, updates Job state.

Phase 6: NLP + media + TTS + subtitle + music + compose + render wired.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from app.config import get_settings
from app.pipeline.compose.aspect import AspectRatio
from app.pipeline.media import make_providers
from app.pipeline.media.base import MediaProvider
from app.pipeline.media.cache import MediaCache
from app.pipeline.media.ranker import aspect_to_orientation, rank_assets
from app.pipeline.media.semantic import SemanticSearchEnhancer
from app.pipeline.models import Job, JobStage, JobStatus, MediaAsset, Orientation, Segment
from app.pipeline.music import make_providers as make_music_providers
from app.pipeline.music.base import MusicProvider, MusicTrack
from app.pipeline.nlp import make_backend
from app.pipeline.nlp.base import NLPBackend
from app.pipeline.subtitle.srt import SubtitleCue, build_cues, write_srt
from app.pipeline.tts.edge_tts import EdgeTTS, TTSResult

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Runs an article -> video Job through all stages."""

    def __init__(
        self,
        nlp: NLPBackend | None = None,
        media_providers: list[MediaProvider] | None = None,
        cache: MediaCache | None = None,
        tts: EdgeTTS | None = None,
        music_providers: list[MusicProvider] | None = None,
        work_dir: Path | None = None,
        compose_fn: Any = None,
        render_fn: Any = None,
        on_progress: Any = None,
    ) -> None:
        self._nlp = nlp
        self._media_providers = media_providers
        self._cache = cache
        self._tts = tts
        self._music_providers = music_providers
        self._work_dir = work_dir
        self._compose_fn = compose_fn
        self._render_fn = render_fn
        self._on_progress = on_progress
        # Per-segment state, keyed by segment.index.
        self._segment_assets: dict[int, MediaAsset] = {}
        self._tts_results: dict[int, TTSResult] = {}
        self._srt_path: Path | None = None
        self._cues: list[SubtitleCue] = []
        self._bgm_track: MusicTrack | None = None
        # Semantic search enhancer for better video matching
        self._semantic = SemanticSearchEnhancer()

    async def run(self, job: Job) -> Job:
        job.status = JobStatus.RUNNING
        await self._touch(job)
        logger.info("Job %s started", job.id)

        try:
            await self._run_nlp(job)
            await self._run_media(job)
            await self._run_tts(job)
            await self._run_subtitles(job)
            await self._run_music(job)
            await self._run_compose_and_render(job)

            job.stage = JobStage.COMPLETE
            job.progress = 1.0
            job.status = JobStatus.DONE
            logger.info(
                "Job %s done: %d segments, %d assets, mood=%s, bgm=%s, output=%s",
                job.id,
                len(job.segments),
                len(self._segment_assets),
                job.mood,
                self._bgm_track.title if self._bgm_track else "none",
                job.output_path,
            )
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.exception("Job %s failed", job.id)
        finally:
            # Cleanup web-recorder temp files
            self._cleanup_web_recorders()
            await self._touch(job)

        return job

    async def _touch(self, job: Job) -> None:
        """Update timestamp + notify subscribers."""
        job.touch()
        if self._on_progress is not None:
            try:
                await self._on_progress(job)
            except Exception:  # noqa: BLE001
                logger.exception("on_progress hook raised")

    @property
    def segment_assets(self) -> dict[int, MediaAsset]:
        return dict(self._segment_assets)

    @property
    def tts_results(self) -> dict[int, TTSResult]:
        return dict(self._tts_results)

    @property
    def srt_path(self) -> Path | None:
        return self._srt_path

    @property
    def cues(self) -> list[SubtitleCue]:
        return list(self._cues)

    @property
    def bgm_track(self) -> MusicTrack | None:
        return self._bgm_track

    # ---- Stage: NLP ----------------------------------------------------------

    async def _run_nlp(self, job: Job) -> None:
        job.stage = JobStage.NLP
        job.progress = 0.05
        await self._touch(job)

        nlp = self._nlp or make_backend(backend=job.nlp_backend)

        segments = await nlp.segment(job.article)
        if not segments:
            raise RuntimeError("NLP segmentation produced zero segments")

        if job.translate_to and job.translate_to != job.source_lang:
            translated_segments = []
            for seg in segments:
                translation = await nlp.translate(
                    seg.text, job.source_lang, job.translate_to
                )
                translated_segments.append(replace(seg, translation=translation))
            segments = translated_segments

        job.segments = segments
        job.mood = await nlp.detect_mood(job.article)
        job.progress = 0.15
        await self._touch(job)
        logger.info(
            "NLP done: %d segments, mood=%s, backend=%s",
            len(segments),
            job.mood,
            job.nlp_backend,
        )

    # ---- Stage: Media fetch --------------------------------------------------

    async def _run_media(self, job: Job) -> None:
        job.stage = JobStage.MEDIA_FETCH
        job.progress = 0.2
        await self._touch(job)

        providers = (
            self._media_providers
            if self._media_providers is not None
            else make_providers()
        )
        if not providers:
            logger.warning(
                "No media providers configured; skipping media stage. "
                "Set PEXELS_API_KEY in .env to enable."
            )
            job.progress = 0.35
            return

        cache = self._cache or MediaCache(get_settings().cache_dir)
        orientation: Orientation = aspect_to_orientation(job.aspect_ratio)

        total = len(job.segments)
        for i, segment in enumerate(job.segments):
            asset = await self._pick_segment_asset(segment, providers, orientation)
            if asset is None:
                logger.warning(
                    "No asset found for segment %d (keywords=%s)",
                    segment.index,
                    list(segment.keywords),
                )
                continue

            # Web-recorder assets already have local_path set (downloaded during search)
            if asset.local_path and asset.local_path.exists():
                self._segment_assets[segment.index] = asset
            else:
                local_path = await cache.fetch(asset.url)
                self._segment_assets[segment.index] = replace(
                    asset, local_path=local_path
                )

            job.progress = 0.2 + 0.2 * ((i + 1) / total)
            await self._touch(job)

        logger.info(
            "Media fetch done: %d/%d segments matched",
            len(self._segment_assets),
            total,
        )

    async def _pick_segment_asset(
        self,
        segment: Segment,
        providers: list[MediaProvider],
        orientation: Orientation,
    ) -> MediaAsset | None:
        # Use semantic enhancer to generate better search keywords (async for LLM)
        original_keywords = list(segment.keywords)
        enhanced_keywords = await self._semantic.generate_keywords_async(
            segment.text, segment.index
        )

        # Prefer enhanced keywords but keep some original
        keywords = enhanced_keywords if enhanced_keywords else original_keywords
        if not keywords:
            return None

        # Also generate Chinese keywords for Pixabay's lang=zh mode
        zh_keywords = self._semantic.extract_chinese_keywords(segment.text)

        # Determine preferred media type based on content
        video_type = self._semantic.suggest_video_type(segment.text, keywords)
        logger.info(
            "Segment %d: keywords=%s, zh_keywords=%s (type=%s)",
            segment.index,
            keywords[:3],
            zh_keywords[:3] if zh_keywords else [],
            video_type,
        )

        searches = []
        for p in providers:
            search_both = getattr(p, "search_both", None)
            if callable(search_both):
                # Pixabay supports lang parameter for Chinese search
                if p.name == "pixabay" and zh_keywords:
                    searches.append(search_both(
                        keywords, orientation, lang="zh"
                    ))
                    # Also search with Chinese keywords for better local results
                    searches.append(search_both(
                        zh_keywords, orientation, lang="zh"
                    ))
                else:
                    searches.append(search_both(keywords, orientation))
            else:
                if video_type in ("image", "both"):
                    if p.name == "pixabay" and zh_keywords:
                        searches.append(p.search(
                            keywords, orientation, "image", lang="zh"
                        ))
                    else:
                        searches.append(p.search(keywords, orientation, "image"))
                if video_type in ("video", "both"):
                    if p.name == "pixabay" and zh_keywords:
                        searches.append(p.search(
                            keywords, orientation, "video", lang="zh"
                        ))
                    else:
                        searches.append(p.search(keywords, orientation, "video"))

        results = await asyncio.gather(*searches, return_exceptions=True)
        all_assets: list[MediaAsset] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Provider search error: %s", r)
                continue
            if isinstance(r, list):
                all_assets.extend(r)

        if not all_assets:
            return None

        ranked = rank_assets(all_assets, keywords, orientation)  # type: ignore[arg-type]
        chosen = ranked[0]
        logger.info(
            "Selected %s for segment %d: %s (%dx%d)",
            chosen.media_type,
            segment.index,
            chosen.url[:60],
            chosen.width,
            chosen.height,
        )
        return chosen

    # ---- Stage: TTS ---------------------------------------------------------

    async def _run_tts(self, job: Job) -> None:
        job.stage = JobStage.TTS
        job.progress = 0.45
        await self._touch(job)

        tts = self._tts or EdgeTTS(self._tts_dir(job))

        items = [
            (seg.index, seg.text, job.voice_primary)
            for seg in job.segments
            if seg.text.strip()
        ]
        if not items:
            raise RuntimeError("No text to synthesize")

        results = await tts.synthesize_segments(items)
        self._tts_results.update(results)

        annotated: list[Segment] = []
        for seg in job.segments:
            r = results.get(seg.index)
            duration = (r.duration_ms / 1000.0) if r else seg.duration_s
            annotated.append(replace(seg, duration_s=duration))
        job.segments = annotated

        job.progress = 0.6
        await self._touch(job)
        total_ms = sum(r.duration_ms for r in results.values())
        logger.info(
            "TTS done: %d clips, total duration=%.2fs",
            len(results),
            total_ms / 1000.0,
        )

    # ---- Stage: Subtitles ---------------------------------------------------

    async def _run_subtitles(self, job: Job) -> None:
        job.stage = JobStage.SUBTITLE
        job.progress = 0.7
        await self._touch(job)

        cues = build_cues(job.segments, self._tts_results)
        self._cues = cues

        # Generate SRT for API download endpoint
        srt_path = self._tts_dir(job) / "captions.srt"
        write_srt(cues, srt_path)
        self._srt_path = srt_path

        # Generate ASS for optimized video rendering (10-100x faster than TextClips)
        from app.pipeline.compose.timeline import target_dimensions
        from app.pipeline.subtitle.srt import write_ass
        aspect: AspectRatio = cast(AspectRatio, job.aspect_ratio)
        target_w, target_h = target_dimensions(aspect)
        ass_path = self._tts_dir(job) / f"{job.id}.ass"
        write_ass(cues, ass_path, target_w, target_h)
        self._ass_path = ass_path

        job.progress = 0.75
        await self._touch(job)
        logger.info("Subtitles built: %d cues -> SRT: %s, ASS: %s", len(cues), srt_path, ass_path)

    # ---- Stage: Music --------------------------------------------------------

    async def _run_music(self, job: Job) -> None:
        job.stage = JobStage.MUSIC
        job.progress = 0.78
        await self._touch(job)

        if not job.bgm_enabled:
            logger.info("BGM disabled by job; skipping music stage.")
            return

        providers = self._music_providers
        if providers is None:
            providers = make_music_providers()
        if not providers:
            logger.warning(
                "No music providers configured. Drop tracks into "
                "<output_dir>/../bgm/<mood>/ to enable BGM."
            )
            return

        total_duration_s = sum(
            r.duration_ms for r in self._tts_results.values()
        ) / 1000.0
        mood = job.mood or "calm"

        for provider in providers:
            try:
                track = await provider.find(
                    mood=mood, min_duration_s=total_duration_s
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Music provider %s errored: %s", provider.name, exc)
                continue
            if track is not None:
                self._bgm_track = track
                logger.info(
                    "BGM selected: %s (provider=%s, mood=%s)",
                    track.local_path.name,
                    provider.name,
                    mood,
                )
                return

        logger.info("No BGM track found across %d providers", len(providers))

    # ---- Stage: Compose + Render --------------------------------------------

    async def _run_compose_and_render(self, job: Job) -> None:
        job.stage = JobStage.COMPOSE
        job.progress = 0.82
        await self._touch(job)

        from app.pipeline.compose.timeline import compose_video as default_compose
        from app.pipeline.render.ffmpeg import render_clip as default_render

        compose_fn = self._compose_fn or default_compose
        render_fn = self._render_fn or default_render

        aspect: AspectRatio = cast(AspectRatio, job.aspect_ratio)
        bgm_path = self._bgm_track.local_path if self._bgm_track else None

        # Use pre-generated ASS file (created in _run_subtitles for speed)
        ass_path: Path | None = getattr(self, '_ass_path', None)
        if ass_path and ass_path.exists():
            logger.info("Using pre-generated ASS subtitles: %s", ass_path)
        else:
            logger.warning("ASS subtitles not found, will not burn subtitles via FFmpeg")

        clip = await asyncio.to_thread(
            compose_fn,
            job.segments,
            self._segment_assets,
            self._tts_results,
            self._cues,
            aspect,
            False,       # burn_subtitles: False — will burn via FFmpeg instead
            None,        # font_path
            bgm_path,
            0.10,
            0.15,
        )

        try:
            job.stage = JobStage.RENDER
            job.progress = 0.92
            await self._touch(job)

            output_path = self._tts_dir(job) / f"{job.id}.mp4"
            settings = get_settings()
            await asyncio.to_thread(
                render_fn,
                clip,
                output_path,
                fps=30,
                use_gpu=settings.use_gpu,
                threads=0,  # 0 = auto-detect CPU cores (faster for libx264)
                ass_path=ass_path,  # Pass ASS file for FFmpeg burning
            )
            job.output_path = output_path
            logger.info("Render done: %s", output_path)
        finally:
            close = getattr(clip, "close", None)
            if callable(close):
                close()

    # ---- Helpers ------------------------------------------------------------

    def _cleanup_web_recorders(self) -> None:
        """Clean up temporary files from WebRecorderProvider instances."""
        from app.pipeline.media.web_recorder import WebRecorderProvider

        if self._media_providers:
            for p in self._media_providers:
                if isinstance(p, WebRecorderProvider):
                    p.cleanup()

    def _tts_dir(self, job: Job) -> Path:
        if self._work_dir is not None:
            base = self._work_dir
        else:
            base = get_settings().output_dir / job.id
        base.mkdir(parents=True, exist_ok=True)
        return base
