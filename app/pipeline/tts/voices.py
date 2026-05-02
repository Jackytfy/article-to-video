"""Language -> Edge-TTS voice resolution.

Edge-TTS uses voice IDs like "zh-CN-XiaoxiaoNeural" or "en-US-AriaNeural".
This module maps short language codes to defaults and validates explicit IDs.

Full voice list: `edge-tts --list-voices`.
"""
from __future__ import annotations

# Sensible defaults per common locale. Pick neutral/clear voices.
_DEFAULTS: dict[str, str] = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "zh-TW": "zh-TW-HsiaoChenNeural",
    "en": "en-US-AriaNeural",
    "en-US": "en-US-AriaNeural",
    "en-GB": "en-GB-LibbyNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural",
}


def resolve_voice(lang_or_voice: str | None, fallback: str = "en-US-AriaNeural") -> str:
    """Resolve a language code or full voice id into a usable Edge-TTS voice id.

    Already-formatted voice ids (containing 'Neural') pass through unchanged.
    """
    if not lang_or_voice:
        return fallback
    if "Neural" in lang_or_voice:
        return lang_or_voice  # Already a full voice id.
    if lang_or_voice in _DEFAULTS:
        return _DEFAULTS[lang_or_voice]
    # Try language family prefix (e.g. "zh-Hans" -> "zh").
    prefix = lang_or_voice.split("-", 1)[0]
    if prefix in _DEFAULTS:
        return _DEFAULTS[prefix]
    return fallback


__all__ = ["resolve_voice"]
