"""Disk-backed media cache + downloader.

- Files keyed by SHA256(url) -> deterministic filename.
- Async download via httpx; idempotent if file already on disk.
- Concurrency-safe enough for single-process use; do not share across procs.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


def _filename_for(url: str, fallback_ext: str = "") -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    ext = Path(url.split("?", 1)[0]).suffix.lower() or fallback_ext
    return f"{digest}{ext}"


class MediaCache:
    """Async, file-based asset cache."""

    def __init__(self, root: Path, timeout_s: float = 30.0) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._timeout_s = timeout_s
        self._locks: dict[str, asyncio.Lock] = {}

    def path_for(self, url: str, *, content_type: str | None = None) -> Path:
        ext = ""
        if content_type:
            ext = mimetypes.guess_extension(content_type.split(";", 1)[0]) or ""
        return self._root / _filename_for(url, fallback_ext=ext)

    async def fetch(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> Path:
        """Download `url` if not cached; return local path."""
        target = self.path_for(url)
        if target.exists() and target.stat().st_size > 0:
            return target

        lock = self._locks.setdefault(target.name, asyncio.Lock())
        async with lock:
            if target.exists() and target.stat().st_size > 0:
                return target  # raced, already done

            owns_client = client is None
            client = client or httpx.AsyncClient(timeout=self._timeout_s)
            try:
                logger.info("Downloading %s -> %s", url, target.name)
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()

                # Re-resolve extension if URL had none.
                if not target.suffix:
                    target = self.path_for(
                        url, content_type=resp.headers.get("content-type")
                    )

                # Direct write — atomic rename via .part is brittle on Windows
                # under AV/indexing locks. Idempotency is guarded by the lock
                # + size-check above.
                target.write_bytes(resp.content)
                return target
            finally:
                if owns_client:
                    await client.aclose()
