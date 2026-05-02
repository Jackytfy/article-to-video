# syntax=docker/dockerfile:1.6
# Article -> Video pipeline. Single-stage: small enough that multi-stage
# overhead doesn't pay back for this workload.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1

# System deps:
#   ffmpeg     -> MoviePy render
#   fonts-noto-cjk -> CJK subtitle rendering (Microsoft YaHei not available on Linux)
#   fonts-dejavu-core -> latin fallback
#   curl       -> uv installer
#   ca-certificates -> outbound HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
        fonts-dejavu-core \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Lockfile copy first for layer caching.
COPY pyproject.toml ./
RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install --system fastapi "uvicorn[standard]" \
        "pydantic>=2.9" "pydantic-settings>=2.6" \
        moviepy ffmpeg-python imageio-ffmpeg \
        edge-tts httpx pillow mutagen python-dotenv \
        ollama jieba

ENV PATH="/opt/venv/bin:$PATH"

COPY app/ ./app/
COPY scripts/ ./scripts/

# Non-root user.
RUN groupadd -g 1000 a2v && useradd -m -u 1000 -g a2v a2v \
    && mkdir -p /app/assets/cache /app/assets/output /app/assets/bgm \
    && chown -R a2v:a2v /app
USER a2v

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
