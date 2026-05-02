"""Local NLP backend.

No external API or LLM required. Uses:
- regex sentence splitting (works for both zh and en)
- jieba for Chinese tokenization + TF-IDF keywords
- simple lexicon mood classifier

Suitable as fallback when no GPU/LLM available. Lower quality than Ollama/LLM.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Final

from app.pipeline.models import Segment

logger = logging.getLogger(__name__)

# Roughly chars-per-second of Mandarin TTS at default rate.
_ZH_CHARS_PER_SECOND: Final[float] = 4.5
# Words per second for English TTS at default rate.
_EN_WORDS_PER_SECOND: Final[float] = 2.5

# Sentence terminators across zh + en.
_SENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[。！？!?\.])\s*|(?<=[；;])\s+"
)

# Mood lexicons — short, intentionally minimal. Can be tuned later.
_MOOD_LEXICON: Final[dict[str, set[str]]] = {
    "energetic": {
        "amazing", "incredible", "fast", "powerful", "win", "rush", "action",
        "惊人", "强大", "迅速", "激动", "热血", "冲", "快",
    },
    "sad": {
        "sad", "loss", "tragic", "grief", "alone", "miss", "regret",
        "悲", "失", "孤独", "遗憾", "哀", "痛", "泪",
    },
    "positive": {
        "joy", "happy", "love", "success", "great", "wonderful", "smile",
        "喜", "爱", "成功", "幸福", "美好", "笑",
    },
    "calm": {
        "peaceful", "quiet", "gentle", "soft", "slow", "still",
        "宁静", "平静", "柔和", "安静", "缓",
    },
}


def _is_chinese(text: str) -> bool:
    han = sum(1 for c in text if "一" <= c <= "鿿")
    return han > len(text) * 0.3


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return parts


def _est_duration_seconds(text: str) -> float:
    if _is_chinese(text):
        chars = sum(1 for c in text if not c.isspace())
        return chars / _ZH_CHARS_PER_SECOND
    words = len(text.split())
    return words / _EN_WORDS_PER_SECOND


def _pack_segments(sentences: list[str], target_seconds: int) -> list[str]:
    """Greedy pack sentences into segments close to target duration."""
    out: list[str] = []
    buf: list[str] = []
    buf_dur = 0.0
    for sent in sentences:
        sent_dur = _est_duration_seconds(sent)
        if buf and buf_dur + sent_dur > target_seconds * 1.4:
            out.append(" ".join(buf) if not _is_chinese(buf[0]) else "".join(buf))
            buf = [sent]
            buf_dur = sent_dur
        else:
            buf.append(sent)
            buf_dur += sent_dur
    if buf:
        out.append(" ".join(buf) if not _is_chinese(buf[0]) else "".join(buf))
    return out


class LocalNLPBackend:
    """Lightweight NLP backend with no LLM calls."""

    def __init__(self) -> None:
        # Lazy import: jieba is only required when this backend is used.
        try:
            import jieba  # type: ignore
            import jieba.analyse  # type: ignore

            self._jieba = jieba
            self._jieba_analyse = jieba.analyse
        except ImportError:
            self._jieba = None
            self._jieba_analyse = None
            logger.warning(
                "jieba not installed; Chinese keyword extraction will degrade. "
                "Install with: uv pip install '.[nlp-local]'"
            )

    async def segment(
        self, article: str, target_seconds_per_seg: int = 8
    ) -> list[Segment]:
        sentences = _split_sentences(article)
        chunks = _pack_segments(sentences, target_seconds_per_seg)
        segments: list[Segment] = []
        for i, chunk in enumerate(chunks):
            kws = await self._extract_keywords(chunk, top_k=5)
            segments.append(
                Segment(index=i, text=chunk, keywords=tuple(kws))
            )
        return segments

    async def keywords(self, segment: Segment, top_k: int = 5) -> list[str]:
        if segment.keywords:
            return list(segment.keywords[:top_k])
        return await self._extract_keywords(segment.text, top_k=top_k)

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if source_lang == target_lang or not target_lang:
            return text
        logger.warning(
            "LocalNLPBackend cannot translate (%s -> %s); returning source unchanged.",
            source_lang,
            target_lang,
        )
        return text

    async def detect_mood(self, article: str) -> str:
        lower = article.lower()
        scores = {
            mood: sum(1 for word in words if word.lower() in lower)
            for mood, words in _MOOD_LEXICON.items()
        }
        if not any(scores.values()):
            return "calm"
        return max(scores.items(), key=lambda kv: kv[1])[0]

    # ---- Internal helpers ----------------------------------------------------

    async def _extract_keywords(self, text: str, top_k: int) -> list[str]:
        if _is_chinese(text) and self._jieba_analyse is not None:
            return list(self._jieba_analyse.extract_tags(text, topK=top_k))
        # Fallback: rank words by frequency, drop stop-ish short tokens.
        tokens = re.findall(r"[A-Za-z一-鿿]{2,}", text)
        counts = Counter(t.lower() for t in tokens)
        return [w for w, _ in counts.most_common(top_k)]
