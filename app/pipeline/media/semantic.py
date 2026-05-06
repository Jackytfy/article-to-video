"""Semantic video search enhancer.

Uses LLM to generate contextual search queries for finding
relevant video content that matches the article text.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline.models import Segment

logger = logging.getLogger(__name__)


class SemanticSearchEnhancer:
    """Generate optimized search queries from article text using LLM."""

    def __init__(self, llm_client=None) -> None:
        """Initialize enhancer with optional LLM client for advanced matching."""
        self._llm = llm_client

    def generate_keywords(self, text: str, segment_index: int) -> list[str]:
        """Generate effective search keywords from text segment.

        This extracts the most visually descriptive terms that are
        likely to return relevant stock video footage.
        """
        # Keywords that indicate video-worthy content
        video_action_words = {
            "doing", "making", "cooking", "working", "walking", "running",
            "talking", "meeting", "writing", "typing", "reading", "thinking",
            "driving", "flying", "building", "creating", "playing", "singing",
            "dancing", "painting", "drawing", "designing", "coding", "testing",
        }

        # Visual scene indicators
        visual_words = {
            "city", "street", "office", "home", "beach", "mountain", "forest",
            "ocean", "sky", "sunset", "sunrise", "night", "morning", "evening",
            "rain", "snow", "storm", "cloud", "fire", "water", "forest",
            "desert", "jungle", "garden", "park", "market", "restaurant",
            "hospital", "school", "university", "factory", "warehouse",
            "computer", "phone", "laptop", "screen", "monitor", "keyboard",
            "car", "bike", "train", "bus", "plane", "ship", "boat",
            "food", "coffee", "tea", "wine", "beer", "meal", "breakfast",
            "lunch", "dinner", "vegetables", "fruits", "meat", "fish",
            "person", "people", "man", "woman", "child", "family", "friends",
            "team", "group", "crowd", "audience", "spectators",
        }

        # Extract words from text
        words = text.lower().split()
        words = [w.strip(".,!?;:\"'()[]{}") for w in words]
        words = [w for w in words if len(w) > 3]

        # Score words by video relevance
        scored: dict[str, float] = {}
        for i, word in enumerate(words):
            score = 1.0

            # Boost action words (high video potential)
            if word in video_action_words:
                score += 3.0

            # Boost visual words (concrete imagery)
            if word in visual_words:
                score += 2.0

            # Boost words near the start (topic-defining)
            if i < 5:
                score += 1.5

            # Boost noun-like words (4-8 chars often best)
            if 4 <= len(word) <= 8:
                score += 0.5

            scored[word] = score

        # Sort by score and get top keywords
        sorted_words = sorted(scored.items(), key=lambda x: -x[1])
        top_keywords = [w for w, _ in sorted_words[:5]]

        # Add bigrams (two-word phrases) for better context
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] in visual_words or words[i] in video_action_words:
                bigrams.append(f"{words[i]} {words[i+1]}")

        # Combine keywords with bigrams
        result = top_keywords + bigrams[:3]

        logger.debug(
            "Segment %d: extracted keywords=%s from text=%s",
            segment_index,
            result,
            text[:50],
        )
        return result[:6]  # Max 6 search terms

    def is_video_worthy(self, text: str) -> bool:
        """Check if text segment is likely to have good video matches.

        Returns True if the text describes:
        - Actions/events that can be visually depicted
        - Concrete objects/places that stock video can represent
        - Scenes or activities
        """
        # Skip very short or abstract text
        if len(text.split()) < 5:
            return False

        # Text describing emotions/abstract concepts is hard to visualize
        abstract_patterns = [
            "feeling", "believe", "think that", "opinion", "maybe",
            "perhaps", "probably", "seems", "appear", "imagine",
        ]

        for pattern in abstract_patterns:
            if pattern in text.lower():
                return False

        return True

    def suggest_video_type(self, text: str, keywords: list[str]) -> str:
        """Suggest preferred video type based on content analysis.

        Returns: 'video', 'image', or 'both'
        """
        action_count = sum(1 for k in keywords if k in {
            "running", "walking", "talking", "working", "playing",
            "dancing", "cooking", "driving", "flying", "building",
        })

        if action_count >= 2:
            return "video"  # Dynamic content needs video
        elif action_count == 1:
            return "both"   # Either works
        else:
            return "image"  # Static content works fine with images


__all__ = ["SemanticSearchEnhancer"]
