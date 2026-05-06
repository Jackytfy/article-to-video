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

# Valid MP4/Video file magic bytes (first 4 bytes should be these for MP4/M4V)
_VIDEO_MAGIC = (
    b"ftyp",  # Standard MP4/M4V
    b"moov",  # MOV format
    b"\x00\x00\x00",  # Some formats start with size
)
# For JPEG/PNG images
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG",  # PNG
)


def _is_valid_media_file(path: Path, content_type: str | None) -> bool:
    """Check if file has valid magic bytes for media content."""
    if path.stat().st_size < 8:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        # Check video magic bytes
        if content_type and "video" in content_type:
            for magic in _VIDEO_MAGIC:
                if header.startswith(magic) or magic in header[:4]:
                    return True
            return False
        # Check image magic bytes
        if content_type and "image" in content_type:
            for magic in _IMAGE_MAGIC:
                if header.startswith(magic):
                    return True
            return False
        # Default: accept files with valid media headers
        for magic in (*_VIDEO_MAGIC, *_IMAGE_MAGIC):
            if header.startswith(magic) or magic in header[:4]:
                return True
        return False
    except Exception:
        return False


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
            # Validate existing cached file
            content_type = None
            if target.suffix in (".mp4", ".m4v", ".webm"):
                content_type = "video/mp4"
            elif target.suffix in (".jpg", ".jpeg", ".png"):
                content_type = "image/jpeg"
            if _is_valid_media_file(target, content_type):
                return target
            # Invalid cached file, delete and re-download
            try:
                target.unlink()
            except OSError:
                pass

        lock = self._locks.setdefault(target.name, asyncio.Lock())
        async with lock:
            # Re-check after acquiring lock
            if target.exists() and _is_valid_media_file(target, None):
                return target  # raced, already done

            owns_client = client is None
            client = client or httpx.AsyncClient(timeout=self._timeout_s)
            content_type = None
            try:
                logger.info("Downloading %s -> %s", url, target.name)
                resp = await client.get(url, follow_redirects=True, headers={
                    "Accept": "*/*",
                })
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                expected_size = int(resp.headers.get("content-length", 0))
                actual_size = len(resp.content)

                # Check for empty or suspiciously small response
                if actual_size < 1024:
                    raise ValueError(
                        f"Downloaded content too small ({actual_size} bytes): "
                        f"likely empty or error page from {url}"
                    )

                # Check size mismatch (indicates truncated download)
                if expected_size > 0 and abs(actual_size - expected_size) > 100:
                    raise ValueError(
                        f"Size mismatch: expected {expected_size}, got {actual_size} "
                        f"from {url}"
                    )

                # Re-resolve extension if URL had none.
                if not target.suffix:
                    target = self.path_for(url, content_type=content_type)

                # Write content
                target.write_bytes(resp.content)

                # Validate downloaded file
                if not _is_valid_media_file(target, content_type):
                    # Delete invalid file
                    try:
                        target.unlink()
                    except OSError:
                        pass
                    raise ValueError(
                        f"Downloaded file has invalid format: {target.name} "
                        f"(content-type: {content_type})"
                    )

                logger.info("Downloaded %s (%d bytes)", target.name, actual_size)
                return target
            except Exception as exc:
                logger.error("Failed to download %s: %s", url, exc)
                # Clean up partial/corrupt file
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    pass
                raise
            finally:
                if owns_client:
                    await client.aclose()
