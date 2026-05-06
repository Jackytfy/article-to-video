"""Semantic search enhancer: LLM-powered topic analysis + scene-aware keyword generation.

Analyses script content to:
1. Identify the topic category (history, tech, cosmos, internet events, etc.)
2. Generate scene-specific English + Chinese search keywords
3. Suggest whether to prefer image or video media

Uses the configured NLP backend (DeepSeek / Qwen / etc.) for LLM calls.
For non-LLM backends (local), falls back to rule-based heuristics.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic categories with scene-mapping knowledge
# ---------------------------------------------------------------------------

class TopicCategory:
    """Known topic categories with search strategy hints."""

    # Category -> (english_search_terms, chinese_search_terms, video_preference)
    # video_preference: "video" | "image" | "both"
    TOPIC_MAP: dict[str, dict[str, Any]] = {
        "history": {
            "en_terms": [
                "ancient", "historical", "medieval", "warrior", "castle",
                "dynasty", "ruins", "archaeology", "battlefield", "monument",
                "emperor", "civilization", "ancient architecture", "historical reenactment",
            ],
            "zh_terms": [
                "历史", "古建筑", "古代", "战场", "王朝", "皇帝",
                "遗址", "文物", "宫殿", "长城", "兵马俑", "古都",
            ],
            "video_pref": "video",
        },
        "cosmos": {
            "en_terms": [
                "space", "galaxy", "nebula", "universe", "stars", "planet",
                "astronomy", "cosmos", "telescope", "astronaut", "milky way",
                "black hole", "solar system", "rocket launch",
            ],
            "zh_terms": [
                "宇宙", "太空", "星云", "银河", "星球", "火箭",
                "天文", "星空", "黑洞", "太阳系",
            ],
            "video_pref": "video",
        },
        "tech": {
            "en_terms": [
                "technology", "digital", "computer", "circuit", "innovation",
                "robot", "AI", "laboratory", "science", "data center",
                "microchip", "server room", "coding", "futuristic",
            ],
            "zh_terms": [
                "科技", "人工智能", "芯片", "机器人", "实验室",
                "数据中心", "编程", "未来科技", "量子计算",
                "技术", "5G", "通讯", "数字化", "智能",
                "互联网", "创新", "算法", "大数据",
            ],
            "video_pref": "both",
        },
        "nature": {
            "en_terms": [
                "nature", "mountain", "ocean", "forest", "river", "sunset",
                "wildlife", "landscape", "waterfall", "jungle", "desert",
                "aurora", "coral reef", "volcano",
            ],
            "zh_terms": [
                "自然", "山脉", "海洋", "森林", "河流", "日落",
                "野生动物", "瀑布", "极光", "火山",
            ],
            "video_pref": "video",
        },
        "internet": {
            "en_terms": [
                "social media", "smartphone", "internet", "viral", "trending",
                "network", "online", "streaming", "digital life", "screen",
                "typing", "blogging", "content creator", "vlog",
            ],
            "zh_terms": [
                "网络", "社交媒体", "直播", "短视频", "网红",
                "手机", "互联网", "热搜", "话题",
            ],
            "video_pref": "both",
        },
        "culture": {
            "en_terms": [
                "culture", "tradition", "festival", "art", "museum",
                "calligraphy", "painting", "music", "dance", "ceremony",
                "heritage", "temple", "tea ceremony",
            ],
            "zh_terms": [
                "文化", "传统", "节日", "艺术", "博物馆",
                "书法", "国画", "茶道", "庙会", "非遗",
            ],
            "video_pref": "video",
        },
        "food": {
            "en_terms": [
                "food", "cooking", "cuisine", "chef", "kitchen",
                "restaurant", "delicious", "meal", "ingredients", "spices",
                "street food", "market", "harvest",
            ],
            "zh_terms": [
                "美食", "烹饪", "厨师", "厨房", "食材",
                "街头小吃", "市场", "火锅", "点心",
            ],
            "video_pref": "video",
        },
        "sports": {
            "en_terms": [
                "sports", "athlete", "competition", "stadium", "fitness",
                "running", "basketball", "soccer", "swimming", "training",
                "victory", "championship", "olympics",
            ],
            "zh_terms": [
                "运动", "体育", "比赛", "健身", "跑步",
                "篮球", "足球", "冠军", "奥运",
            ],
            "video_pref": "video",
        },
        "science": {
            "en_terms": [
                "science", "experiment", "research", "laboratory", "microscope",
                "chemistry", "physics", "biology", "DNA", "molecule",
                "discovery", "innovation", "scientific",
            ],
            "zh_terms": [
                "科学", "实验", "研究", "显微镜", "化学",
                "物理", "生物", "DNA", "分子",
            ],
            "video_pref": "both",
        },
        "military": {
            "en_terms": [
                "military", "soldier", "army", "navy", "aircraft",
                "warship", "tank", "fighter jet", "parade", "defense",
                "border", "patrol", "combat",
            ],
            "zh_terms": [
                "军事", "军队", "士兵", "战舰", "战斗机",
                "坦克", "阅兵", "国防", "边防",
            ],
            "video_pref": "video",
        },
        "education": {
            "en_terms": [
                "education", "school", "university", "student", "teacher",
                "classroom", "library", "study", "graduation", "campus",
                "lecture", "books", "learning",
            ],
            "zh_terms": [
                "教育", "学校", "大学", "学生", "老师",
                "课堂", "图书馆", "毕业", "校园",
            ],
            "video_pref": "both",
        },
        "business": {
            "en_terms": [
                "business", "office", "meeting", "finance", "economy",
                "stock market", "startup", "corporate", "teamwork", "strategy",
                "deal", "growth", "investment",
            ],
            "zh_terms": [
                "商业", "办公", "会议", "金融", "经济",
                "股市", "创业", "团队", "投资",
            ],
            "video_pref": "both",
        },
        "health": {
            "en_terms": [
                "health", "medical", "hospital", "doctor", "medicine",
                "wellness", "mental health", "therapy", "surgery", "care",
                "vaccine", "pharmacy", "nursing",
            ],
            "zh_terms": [
                "健康", "医疗", "医院", "医生", "药品",
                "养生", "心理", "手术", "护理",
            ],
            "video_pref": "both",
        },
        "travel": {
            "en_terms": [
                "travel", "tourism", "adventure", "landmark", "scenic",
                "vacation", "journey", "exploration", "backpacking", "destination",
                "map", "passport", "souvenir",
            ],
            "zh_terms": [
                "旅行", "旅游", "风景", "景点", "冒险",
                "度假", "探索", "背包客", "名胜",
            ],
            "video_pref": "video",
        },
    }

    @classmethod
    def get_categories(cls) -> list[str]:
        return list(cls.TOPIC_MAP.keys())


# ---------------------------------------------------------------------------
# Scene keyword expansions for fine-grained matching
# ---------------------------------------------------------------------------

# Maps specific Chinese entities/events to search-friendly terms
_SCENE_OVERRIDES: dict[str, list[str]] = {
    # Historical figures & events
    "秦始皇": ["qin dynasty", "terracotta warriors", "great wall china"],
    "三国": ["three kingdoms", "ancient china war", "chinese dynasty"],
    "唐朝": ["tang dynasty", "ancient chinese palace", "chinese golden age"],
    "宋朝": ["song dynasty", "ancient chinese culture", "chinese scroll painting"],
    "明朝": ["ming dynasty", "forbidden city", "great wall"],
    "清朝": ["qing dynasty", "imperial palace", "ancient china"],
    "丝绸之路": ["silk road", "ancient trade route", "desert caravan"],
    "长征": ["long march", "chinese revolution", "mountain trek"],
    "抗战": ["world war", "war memorial", "military history"],
    # Cosmic
    "黑洞": ["black hole", "space", "galaxy", "cosmos"],
    "火星": ["mars", "red planet", "space exploration", "mars rover"],
    "引力波": ["gravitational waves", "space observatory", "physics"],
    "暗物质": ["dark matter", "space nebula", "galaxy cluster"],
    # Tech / Internet
    "人工智能": ["artificial intelligence", "AI robot", "neural network", "machine learning"],
    "芯片": ["microchip", "semiconductor", "circuit board", "processor"],
    "区块链": ["blockchain", "digital network", "cryptocurrency"],
    "直播": ["livestream", "streaming", "content creator", "vlog"],
    "短视频": ["short video", "social media", "smartphone recording"],
    # Nature
    "大熊猫": ["giant panda", "wildlife china", "bamboo forest"],
    "长江": ["yangtze river", "chinese river landscape", "river china"],
    "黄河": ["yellow river", "chinese landscape", "river valley"],
}

# ---------------------------------------------------------------------------
# SemanticSearchEnhancer
# ---------------------------------------------------------------------------


class SemanticSearchEnhancer:
    """Analyse script content and produce scene-aware search keywords.

    Strategy:
    1. Try LLM-based topic classification + keyword generation (best quality)
    2. Fall back to rule-based heuristic matching (always available)
    3. Merge LLM keywords with topic-specific scene terms
    """

    def __init__(self) -> None:
        self._llm_checked: bool = False
        self._llm_available: bool = False
        self._client: Any = None  # noqa: reportExplicitAny
        self._model: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_keywords(self, text: str, segment_index: int = 0) -> list[str]:  # noqa: reportUnusedParameter
        """Generate enhanced search keywords for a text segment.

        Returns a list of English keywords suitable for Pexels/Pixabay search,
        plus optional Chinese keywords for Pixabay's lang=zh mode.
        """
        # Rule-based: always compute as baseline (sync version, no LLM)
        rule_kw = self._rule_based_keywords(text)
        llm_kw: list[str] = []

        # Merge: LLM keywords first (higher quality), then rule-based
        merged = self._merge_keywords(llm_kw, rule_kw)

        # Apply scene overrides if text matches known entities
        override_kw = self._apply_scene_overrides(text)
        if override_kw:
            merged = override_kw + [k for k in merged if k not in override_kw]

        return merged[:6]

    async def generate_keywords_async(self, text: str, segment_index: int = 0) -> list[str]:  # noqa: reportUnusedParameter
        """Async version: uses LLM for higher-quality keyword generation."""
        # 1. Rule-based baseline
        rule_kw = self._rule_based_keywords(text)

        # 2. LLM-enhanced keywords
        llm_kw = await self._llm_generate_keywords(text)

        # 3. Merge
        merged = self._merge_keywords(llm_kw, rule_kw)

        # 4. Scene overrides
        override_kw = self._apply_scene_overrides(text)
        if override_kw:
            merged = override_kw + [k for k in merged if k not in override_kw]

        return merged[:6]

    def suggest_video_type(self, text: str, keywords: list[str]) -> str:  # noqa: reportUnusedParameter
        """Suggest media type: 'video', 'image', or 'both'.

        Based on detected topic category — some topics strongly prefer
        video (history, nature, sports), others work well with images.
        """
        topic = self._detect_topic_rule_based(text)
        topic_info = TopicCategory.TOPIC_MAP.get(topic, {})
        return topic_info.get("video_pref", "both")

    def extract_chinese_keywords(self, text: str) -> list[str]:
        """Extract Chinese keywords for Pixabay lang=zh search.

        Returns 2-4 Chinese keywords suitable for searching Chinese
        content on Pixabay.
        """
        # Get topic-specific Chinese terms
        topic = self._detect_topic_rule_based(text)
        topic_info = TopicCategory.TOPIC_MAP.get(topic, {})
        zh_terms = list(topic_info.get("zh_terms", [])[:3])

        # Extract Chinese keywords from text
        text_zh_kw = self._extract_zh_keywords(text)[:3]

        # Check scene overrides
        override_zh: list[str] = []
        for entity in _SCENE_OVERRIDES:
            if entity in text:
                override_zh.append(entity)

        # Merge: overrides first, then extracted, then topic defaults
        merged: list[str] = []
        seen: set[str] = set()
        for kw in override_zh + text_zh_kw + zh_terms:
            if kw not in seen:
                seen.add(kw)
                merged.append(kw)

        return merged[:4]

    async def classify_topic(self, text: str) -> str:
        """Classify the topic of the given text. Returns category name."""
        # Try LLM first
        topic = await self._llm_classify_topic(text)
        if topic:
            return topic
        return self._detect_topic_rule_based(text)

    # ------------------------------------------------------------------
    # Rule-based keyword generation (no LLM required)
    # ------------------------------------------------------------------

    def _rule_based_keywords(self, text: str) -> list[str]:
        """Generate keywords using rule-based heuristics + topic mapping."""
        topic = self._detect_topic_rule_based(text)
        topic_info = TopicCategory.TOPIC_MAP.get(topic, {})

        # Get base terms from topic
        en_terms = list(topic_info.get("en_terms", [])[:4])

        # Extract key nouns from text (simple Chinese keyword extraction)
        zh_terms = self._extract_zh_keywords(text)

        # Translate Chinese keywords to English search terms
        # (using a simple mapping for common terms)
        translated = self._simple_zh_to_en(zh_terms)

        # Combine: translated Chinese terms + topic-specific English terms
        all_kw = translated + en_terms
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for kw in all_kw:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique.append(kw)
        return unique[:6]

    def _detect_topic_rule_based(self, text: str) -> str:
        """Detect topic category using keyword matching."""
        lower = text.lower()
        best_topic = "nature"  # default
        best_score = 0

        for topic, info in TopicCategory.TOPIC_MAP.items():
            score = 0
            for term in info.get("zh_terms", []):
                if term in text:
                    score += 2  # Chinese terms are stronger signal
            for term in info.get("en_terms", []):
                if term.lower() in lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic

    def _extract_zh_keywords(self, text: str) -> list[str]:
        """Extract meaningful Chinese keywords from text.

        Uses jieba for proper word segmentation if available,
        falls back to regex-based extraction.
        """
        # Try jieba first for proper word segmentation
        try:
            import jieba  # type: ignore
            import jieba.analyse  # type: ignore

            # Use TF-IDF extraction for best results
            raw_tags: list[Any] = jieba.analyse.extract_tags(
                text, topK=8, withWeight=False
            )
            # Ensure all tags are strings (jieba may return tuples when withWeight=True)
            tags = [str(t) for t in raw_tags]
            # Filter: only keep terms with 2+ chars, skip stop words
            stop_words = {
                "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                "都", "一", "上", "也", "很", "到", "说", "要", "去", "你",
                "会", "着", "看", "好", "这", "他", "她", "它", "们", "那",
                "个", "为", "中", "来", "从", "对", "与", "但", "被", "把",
                "让", "给", "而", "又", "所", "以", "于", "之", "其",
                "可以", "可能", "应该", "需要", "已经", "正在", "通过",
                "进行", "使用", "包括", "以及", "其中", "同时", "此外",
                "就是", "还是", "不是", "没有", "这样", "那样",
            }
            return [t for t in tags if t not in stop_words and len(t) >= 2][:5]
        except ImportError:
            pass

        # Fallback: regex-based extraction
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
            "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
            "们", "那", "个", "为", "中", "来", "从", "对", "与", "但",
            "被", "把", "让", "给", "而", "又", "所", "以", "于", "之",
            "其", "这个", "那个", "什么", "怎么", "如何", "为什么",
            "可以", "可能", "应该", "需要", "已经", "正在", "通过",
            "进行", "使用", "包括", "以及", "其中", "同时", "此外",
        }

        # Extract Chinese character sequences (2-4 chars for better quality)
        zh_pattern = re.compile(r"[\u4e00-\u9fff]{2,4}")
        candidates = zh_pattern.findall(text)

        # Filter out stop words and too-common terms
        keywords = [c for c in candidates if c not in stop_words and len(c) >= 2]

        # Count frequency and return top unique keywords
        from collections import Counter
        counts = Counter(keywords)
        return [w for w, _ in counts.most_common(5)]

    def _simple_zh_to_en(self, zh_terms: list[str]) -> list[str]:
        """Simple Chinese-to-English translation for search keywords.

        Uses the SCENE_OVERRIDES dict + a basic mapping for common terms.
        """
        result: list[str] = []
        for term in zh_terms:
            # Check scene overrides first
            if term in _SCENE_OVERRIDES:
                result.extend(_SCENE_OVERRIDES[term][:2])
                continue

            # Check if term appears in any topic's zh_terms
            for _topic, info in TopicCategory.TOPIC_MAP.items():
                zh_list = info.get("zh_terms", [])
                if term in zh_list:
                    idx = zh_list.index(term)
                    en_list = info.get("en_terms", [])
                    if idx < len(en_list):
                        result.append(en_list[idx])
                    break

        return result

    # ------------------------------------------------------------------
    # Scene overrides
    # ------------------------------------------------------------------

    def _apply_scene_overrides(self, text: str) -> list[str]:
        """Check if text contains known entities and return specific search terms."""
        overrides: list[str] = []
        for entity, terms in _SCENE_OVERRIDES.items():
            if entity in text:
                overrides.extend(terms[:2])
        return overrides[:4]  # Cap override keywords

    # ------------------------------------------------------------------
    # LLM-based keyword generation
    # ------------------------------------------------------------------

    async def _llm_generate_keywords(self, text: str) -> list[str]:
        """Use LLM to generate scene-aware search keywords."""
        try:
            client, model = self._get_llm_client()
            if client is None:
                return []
        except Exception:
            return []

        prompt = (
            '请只返回JSON格式: {"keywords": ["keyword1", "keyword2"], '
            '"topic": "category"}\n\n'
            "分析以下文本内容，生成适合搜索视频素材的英文关键词。\n"
            "要求：\n"
            "1. 识别文本主题类别（history/cosmos/tech/nature/internet/culture/"
            "food/sports/science/military/education/business/health/travel）\n"
            "2. 生成3-5个视觉化的英文搜索关键词，用于在视频素材网站搜索\n"
            "3. 关键词应该是具体的场景、物体、环境，而非抽象概念\n"
            "4. 优先选择能找到真实视频片段的关键词\n\n"
            f"文本: {text[:500]}"
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content
            raw = _safe_json(content)
            if isinstance(raw, dict):
                return list(raw.get("keywords", []))[:5]
        except Exception as exc:
            logger.warning("LLM keyword generation failed: %s", exc)

        return []

    async def _llm_classify_topic(self, text: str) -> str:
        """Use LLM to classify topic category."""
        try:
            client, model = self._get_llm_client()
            if client is None:
                return ""
        except Exception:
            return ""

        categories = TopicCategory.get_categories()
        prompt = (
            '请只返回JSON格式: {"topic": "category"}\n\n'
            f"将以下文本分类为以下类别之一: {', '.join(categories)}\n\n"
            f"文本: {text[:500]}"
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content
            raw = _safe_json(content)
            if isinstance(raw, dict):
                topic = str(raw.get("topic", "")).lower()
                if topic in categories:
                    return topic
        except Exception as exc:
            logger.warning("LLM topic classification failed: %s", exc)

        return ""

    def _get_llm_client(self) -> tuple[Any, str]:
        """Get or lazily initialize the LLM client."""
        if self._llm_checked:
            if self._llm_available:
                return self._client, self._model
            return None, ""

        self._llm_checked = True
        settings = get_settings()

        # Try DeepSeek first (most likely configured)
        if settings.deepseek_api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=30.0,
            )
            self._model = settings.deepseek_model
            self._llm_available = True
            return self._client, self._model

        # Try Qwen (DashScope)
        if settings.dashscope_api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=30.0,
            )
            self._model = settings.qwen_model
            self._llm_available = True
            return self._client, self._model

        # Try Zhipu
        if settings.zhipu_api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.zhipu_api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4",
                timeout=30.0,
            )
            self._model = settings.zhipu_model
            self._llm_available = True
            return self._client, self._model

        self._llm_available = False
        return None, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_keywords(llm_kw: list[str], rule_kw: list[str]) -> list[str]:
        """Merge LLM and rule-based keywords, LLM first, dedup."""
        seen: set[str] = set()
        result: list[str] = []
        for kw in llm_kw + rule_kw:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                result.append(kw)
        return result


def _safe_json(text: str | None) -> dict[str, Any]:
    """Parse JSON from LLM response, handling common wrappers."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
