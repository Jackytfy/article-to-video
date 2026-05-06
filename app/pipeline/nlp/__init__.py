"""NLP stage: pluggable backends + factory.

Selects an NLPBackend based on settings.nlp_backend.
Heavy backend imports are lazy so the unused ones never load.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.pipeline.nlp.base import NLPBackend


def make_backend(
    backend: str | None = None,
    settings: Settings | None = None,
) -> NLPBackend:
    """Build the configured NLPBackend.

    Args:
        backend: optional override. Options:
            - "ollama": Local Ollama server (default, free)
            - "llm": Claude/OpenAI (requires API keys)
            - "local": Offline jieba+regex
            - "deepseek": DeepSeek API (OpenAI-compatible)
            - "qwen": 通义千问 via DashScope
            - "zhipu": 智谱 GLM (OpenAI-compatible)
            - "spark": 讯飞星火 (WebSocket)
            - "wenxin": 百度文心一言 ERNIE
        settings: optional injected settings (testing).
    """
    settings = settings or get_settings()
    backend = backend or settings.nlp_backend

    if backend == "ollama":
        from app.pipeline.nlp.ollama import OllamaNLPBackend

        return OllamaNLPBackend(
            host=settings.ollama_host,
            model=settings.ollama_model,
        )

    if backend == "local":
        from app.pipeline.nlp.local import LocalNLPBackend

        return LocalNLPBackend()

    if backend == "llm":
        from app.pipeline.nlp.llm import LLMNLPBackend

        return LLMNLPBackend(
            model=settings.llm_model,
            anthropic_key=settings.anthropic_api_key,
            openai_key=settings.openai_api_key,
        )

    if backend == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError(
                "DeepSeek backend requires DEEPSEEK_API_KEY. "
                "Get one at https://platform.deepseek.com/"
            )
        from app.pipeline.nlp.deepseek import DeepSeekNLPBackend

        return DeepSeekNLPBackend(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )

    if backend == "qwen":
        if not settings.dashscope_api_key:
            raise ValueError(
                "通义千问 backend requires DASHSCOPE_API_KEY. "
                "Get one at https://dashscope.console.aliyun.com/"
            )
        from app.pipeline.nlp.qwen import QwenNLPBackend

        return QwenNLPBackend(
            api_key=settings.dashscope_api_key,
            model=settings.qwen_model,
        )

    if backend == "zhipu":
        if not settings.zhipu_api_key:
            raise ValueError(
                "智谱 GLM backend requires ZHIPU_API_KEY. "
                "Get one at https://open.bigmodel.cn/"
            )
        from app.pipeline.nlp.zhipu import ZhipuNLPBackend

        return ZhipuNLPBackend(
            api_key=settings.zhipu_api_key,
            model=settings.zhipu_model,
        )

    if backend == "spark":
        if not (settings.spark_app_id and settings.spark_api_key):
            raise ValueError(
                "讯飞星火 backend requires SPARK_APP_ID, SPARK_API_KEY, "
                "and SPARK_API_SECRET. Get them at https://www.xfyun.cn/"
            )
        from app.pipeline.nlp.spark import SparkNLPBackend

        return SparkNLPBackend(
            app_id=settings.spark_app_id,
            api_key=settings.spark_api_key,
            api_secret=settings.spark_api_secret or "",
            model=settings.spark_model,
        )

    if backend == "wenxin":
        if not (settings.wenxin_api_key and settings.wenxin_secret_key):
            raise ValueError(
                "百度文心一言 backend requires WENXIN_API_KEY and WENXIN_SECRET_KEY. "
                "Get them at https://cloud.baidu.com/"
            )
        from app.pipeline.nlp.wenxin import WenxinNLPBackend

        return WenxinNLPBackend(
            api_key=settings.wenxin_api_key,
            secret_key=settings.wenxin_secret_key,
            model=settings.wenxin_model,
        )

    raise ValueError(
        f"Unknown NLP backend: {backend!r}. "
        f"Valid options: ollama, llm, local, deepseek, qwen, zhipu, spark, wenxin"
    )


__all__ = ["make_backend"]
