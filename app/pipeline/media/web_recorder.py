"""Web-recorder media provider: yt-dlp video download + Playwright page recording.

Strategy:
1. **Primary (yt-dlp)**: Search & download real video clips from Bilibili / YouTube.
   If a segment is about "秦始皇", we search Bilibili for "秦始皇 纪录片",
   download a short segment, and return it as a MediaAsset.

2. **Fallback (Playwright)**: When yt-dlp finds nothing (or isn't installed),
   open relevant web pages in a headless browser, scroll through them while
   recording the viewport, producing a short video clip.

Both paths return standard MediaAsset objects that slot into the existing
pipeline (ranking, caching, compositing).

Configuration (in .env):
  - WEB_RECORDER_ENABLED=true          (default: false)
  - WEB_RECORDER_MODE=auto|ytdlp|playwright  (default: auto)
  - WEB_RECORDER_MAX_DURATION=15       (max clip seconds, default: 15)
  - BILIBILI_SESSDATA=                 (optional, for higher quality)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx

from app.pipeline.models import MediaAsset, MediaType, Orientation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ytdlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def _is_playwright_available() -> bool:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Bilibili search
# ---------------------------------------------------------------------------

_BILIBILI_VIDEO = "https://www.bilibili.com/video/{vid}"


async def _search_bilibili(
    keywords: list[str],
    limit: int = 5,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Search Bilibili for videos matching keywords.

    Strategy:
    1. Try Bilibili's popular/new video search API (no auth needed)
    2. Fallback: use yt-dlp's built-in Bilibili search
    3. Last resort: use Playwright to render the search page

    Returns list of dicts with keys: bvid, title, duration_s, url.
    """
    query = " ".join(keywords[:3])
    if not query.strip():
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.bilibili.com",
    }
    sessdata = os.environ.get("BILIBILI_SESSDATA", "")
    if sessdata:
        headers["Cookie"] = f"SESSDATA={sessdata}"

    # Strategy 1: Use Bilibili's grpc/search API (works without auth in many cases)
    try:
        results = await _search_bilibili_api(query, limit, headers, timeout_s)
        if results:
            return results
    except Exception as exc:
        logger.debug("Bilibili API search failed: %s", exc)

    # Strategy 2: yt-dlp search
    try:
        results = await _ytdlp_search_bilibili(query, limit, timeout_s)
        if results:
            return results
    except Exception as exc:
        logger.debug("yt-dlp Bilibili search failed: %s", exc)

    # Strategy 3: Playwright-rendered search
    try:
        results = await _playwright_search_bilibili(query, limit, timeout_s)
        if results:
            return results
    except Exception as exc:
        logger.debug("Playwright Bilibili search failed: %s", exc)

    return []


async def _search_bilibili_api(
    query: str,
    limit: int,
    headers: dict[str, str],
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Try Bilibili's web search API (may require cookies)."""
    import hashlib
    import time

    # First get a buvid3 cookie by visiting the main page
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        # Visit main page to get basic cookies
        await client.get("https://www.bilibili.com", headers=headers)

        # Try the search API with a simple params approach
        # Using the mobile API which is more lenient
        params = {
            "keyword": query,
            "page": 1,
            "page_size": limit,
        }
        mobile_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 11; Pixel 5) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Mobile Safari/537.36"
            ),
            "Referer": "https://m.bilibili.com",
        }
        if sessdata := os.environ.get("BILIBILI_SESSDATA", ""):
            mobile_headers["Cookie"] = f"SESSDATA={sessdata}"

        resp = await client.get(
            "https://api.bilibili.com/x/web-interface/search/type",
            params={**params, "search_type": "video"},
            headers={**headers, **mobile_headers},
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        if data.get("code") != 0:
            return []

        results: list[dict[str, Any]] = []
        for item in (data.get("data", {}).get("result") or [])[:limit]:
            bvid = item.get("bvid", "")
            if not bvid:
                continue
            duration_str = str(item.get("duration", "0:00"))
            parts = duration_str.split(":")
            duration_s = 0
            try:
                if len(parts) == 2:
                    duration_s = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    duration_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except (ValueError, IndexError):
                pass
            title = re.sub(r"<[^>]+>", "", str(item.get("title", "")))
            results.append({
                "bvid": bvid,
                "title": title,
                "duration_s": duration_s,
                "url": _BILIBILI_VIDEO.format(vid=bvid),
            })

        return results


async def _ytdlp_search_bilibili(
    query: str,
    limit: int = 5,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Use yt-dlp's built-in search to find Bilibili videos.

    yt-dlp supports 'bilisearch:QUERY' for searching Bilibili.
    The search returns AV IDs which yt-dlp can download directly.
    """
    if not _is_ytdlp_available():
        return []

    cmd = [
        "yt-dlp",
        f"bilisearch{limit}:{query}",
        "--flat-playlist",
        "--print", "%(id)s|%(url)s",
        "--no-warnings",
        "--quiet",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

        if proc.returncode != 0:
            return []

        results: list[dict[str, Any]] = []
        for line in stdout.decode(errors="replace").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 1)
            vid = parts[0].strip() if len(parts) >= 1 else ""
            url = parts[1].strip() if len(parts) >= 2 else ""

            if vid:
                # Use the URL from yt-dlp directly (it may be AV or BV format)
                if not url:
                    url = f"https://www.bilibili.com/video/av{vid}"
                results.append({
                    "bvid": vid,
                    "title": query,
                    "duration_s": 0,
                    "url": url,
                })

        return results[:limit]
    except Exception as exc:
        logger.warning("yt-dlp Bilibili search failed: %s", exc)
        return []


async def _playwright_search_bilibili(
    query: str,
    limit: int = 5,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Use Playwright to render Bilibili search page and extract video links."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return []

    search_url = f"https://search.bilibili.com/video?keyword={quote_plus(query)}"

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            await page.goto(search_url, wait_until="networkidle", timeout=timeout_s * 1000)
            # Wait for video results to render
            await page.wait_for_timeout(3000)

            # Extract video links from the rendered page
            links = await page.eval_on_selector_all(
                'a[href*="/video/BV"]',
                """els => els.map(el => ({
                    href: el.href,
                    title: el.textContent || ''
                }))"""
            )

            await context.close()
            await browser.close()

            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            for link in links:
                href = link.get("href", "")
                # Extract BV ID
                match = re.search(r"/video/(BV[a-zA-Z0-9]+)", href)
                if not match:
                    continue
                bvid = match.group(1)
                if bvid in seen:
                    continue
                seen.add(bvid)
                title = link.get("title", "").strip()[:100]
                results.append({
                    "bvid": bvid,
                    "title": title or query,
                    "duration_s": 0,
                    "url": _BILIBILI_VIDEO.format(vid=bvid),
                })
                if len(results) >= limit:
                    break

            return results
    except Exception as exc:
        logger.warning("Playwright Bilibili search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------

async def _ytdlp_download_segment(
    url: str,
    output_dir: Path,
    max_duration_s: int = 15,
    timeout_s: float = 120.0,
) -> Path | None:
    """Download a short video segment using yt-dlp.

    Downloads the video, then trims to max_duration_s using FFmpeg.
    Returns path to the trimmed MP4 file, or None on failure.
    """
    if not _is_ytdlp_available():
        logger.debug("yt-dlp not available, skipping download")
        return None

    output_template = str(output_dir / "ytdlp_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--max-filesize", "100M",
        "--no-warnings",
        "--quiet",
        # Add cookies from browser if available (helps with Bilibili)
        "--cookies-from-browser", "chrome",
        url,
    ]

    # If cookies-from-browser fails, try without it
    fallback_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--max-filesize", "100M",
        "--no-warnings",
        "--quiet",
        url,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

        # If cookies-from-browser failed, try without it
        if proc.returncode != 0:
            logger.debug("yt-dlp with cookies failed, trying without: %s", stderr.decode(errors="replace")[:200])
            proc = await asyncio.create_subprocess_exec(
                *fallback_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

        if proc.returncode != 0:
            logger.warning("yt-dlp failed for %s: %s", url, stderr.decode(errors="replace")[:200])
            return None

        # Find the downloaded file
        downloaded = list(output_dir.glob("ytdlp_*.mp4"))
        if not downloaded:
            logger.warning("yt-dlp produced no output file for %s", url)
            return None

        raw_path = downloaded[0]

        # Trim to max_duration_s using FFmpeg
        trimmed_path = output_dir / f"trimmed_{raw_path.name}"
        trim_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_path),
            "-t", str(max_duration_s),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(trimmed_path),
        ]

        proc2 = await asyncio.create_subprocess_exec(
            *trim_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc2.communicate()

        # Clean up raw file
        try:
            raw_path.unlink()
        except OSError:
            pass

        if trimmed_path.exists() and trimmed_path.stat().st_size > 1024:
            return trimmed_path

        # If trimming failed, try using the raw file
        if raw_path.exists():
            return raw_path

        return None

    except asyncio.TimeoutError:
        logger.warning("yt-dlp timed out for %s", url)
        return None
    except Exception as exc:
        logger.warning("yt-dlp download error for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Playwright page recording
# ---------------------------------------------------------------------------

async def _playwright_record_page(
    url: str,
    output_path: Path,
    duration_s: int = 10,
    width: int = 1280,
    height: int = 720,
    timeout_s: float = 30.0,
) -> Path | None:
    """Open a URL in headless browser, scroll through it, and record a video.

    Returns path to the recorded WebM file, or None on failure.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed; cannot record page")
        return None

    video_dir = output_path.parent
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                record_video_dir=str(video_dir),
                record_video_size={"width": width, "height": height},
            )
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
            except Exception as exc:
                logger.warning("Page load failed for %s: %s", url, exc)
                await context.close()
                await browser.close()
                return None

            # Wait a moment for images to load
            await page.wait_for_timeout(1500)

            # Smooth scroll through the page
            total_height = await page.evaluate("document.body.scrollHeight")
            viewport_height = await page.evaluate("window.innerHeight")
            scroll_step = max(viewport_height // 3, 100)
            current = 0
            scroll_interval = max(500, (duration_s * 1000) // max(1, total_height // scroll_step))

            while current < total_height:
                await page.evaluate(f"window.scrollBy(0, {scroll_step})")
                current += scroll_step
                await page.wait_for_timeout(scroll_interval)

            # Ensure minimum recording duration
            await page.wait_for_timeout(1000)

            # Close to finalize video
            await context.close()
            await browser.close()

            # Find the recorded video
            video_path = video_dir / f"{output_path.stem}.webm"
            # Playwright saves with a random name; find the latest .webm
            webm_files = sorted(
                video_dir.glob("*.webm"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if webm_files:
                recorded = webm_files[0]
                if recorded != video_path:
                    recorded.rename(video_path)
                return video_path

            return None

    except Exception as exc:
        logger.warning("Playwright recording failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Web search for finding relevant pages (for Playwright fallback)
# ---------------------------------------------------------------------------

async def _search_web_pages(
    keywords: list[str],
    limit: int = 3,
    timeout_s: float = 10.0,
) -> list[dict[str, str]]:
    """Search the web for pages related to keywords.

    Uses multiple search engines for reliability.
    Returns list of dicts: {title, url}.
    """
    query = " ".join(keywords[:3])
    if not query.strip():
        return []

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    # Try DuckDuckGo (no API key, reliable HTML results)
    try:
        ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.get(ddg_url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        # DuckDuckGo HTML results contain links in <a class="result__a" href="...">
        for match in re.finditer(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html
        ):
            url = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            # DDG uses redirect URLs; extract the real URL
            real_url_match = re.search(r'uddg=([^&]+)', url)
            if real_url_match:
                from urllib.parse import unquote
                url = unquote(real_url_match.group(1))
            # Skip non-http and low-quality results
            if not url.startswith("http"):
                continue
            if any(skip in url for skip in ("duckduckgo", "microsoft.com", "bing.com")):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if title and len(title) > 3:
                results.append({"title": title, "url": url})
            if len(results) >= limit:
                break
    except Exception as exc:
        logger.debug("DuckDuckGo search failed: %s", exc)

    # Fallback: Use Wikipedia search for educational/historical content
    if len(results) < limit:
        try:
            wiki_url = (
                f"https://zh.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={quote_plus(query)}"
                f"&format=json&srlimit={limit}"
            )
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
                resp = await client.get(wiki_url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                wiki_link = f"https://zh.wikipedia.org/wiki/{quote_plus(title)}"
                if wiki_link not in seen_urls:
                    seen_urls.add(wiki_link)
                    results.append({"title": title, "url": wiki_link})
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.debug("Wikipedia search failed: %s", exc)

    # Last resort: Baidu Baike (for Chinese content)
    if len(results) < limit:
        try:
            baidu_url = f"https://baike.baidu.com/item/{quote_plus(query)}"
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
                resp = await client.get(baidu_url, headers=headers)
                if resp.status_code == 200:
                    final_url = str(resp.url)
                    if final_url not in seen_urls:
                        seen_urls.add(final_url)
                        results.append({"title": query, "url": final_url})
        except Exception as exc:
            logger.debug("Baidu Baike search failed: %s", exc)

    return results[:limit]


# ---------------------------------------------------------------------------
# WebRecorderProvider
# ---------------------------------------------------------------------------

class WebRecorderProvider:
    """Media provider that combines yt-dlp video download with Playwright
    page recording to produce video clips matching script content.

    Priority:
    1. yt-dlp: Search Bilibili for real videos, download a short segment.
    2. Playwright: Search web for relevant pages, record browser scrolling.
    """

    name = "web_recorder"

    def __init__(
        self,
        max_duration_s: int = 15,
        recording_width: int = 1280,
        recording_height: int = 720,
        mode: str = "auto",  # auto | ytdlp | playwright
        temp_dir: Path | None = None,
    ) -> None:
        self._max_duration_s = max_duration_s
        self._rec_width = recording_width
        self._rec_height = recording_height
        self._mode = mode
        self._temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="web_recorder_"))
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._asset_counter = 0

    async def search(
        self,
        keywords: list[str],
        orientation: Orientation,
        media_type: MediaType,
        limit: int = 10,
    ) -> list[MediaAsset]:
        """Search for and produce video clips matching keywords.

        This provider always returns video assets. It actively downloads/records
        content, so results are real files on disk (local_path is set).
        """
        if not keywords:
            return []

        assets: list[MediaAsset] = []

        # Strategy 1: yt-dlp (download real videos from Bilibili)
        if self._mode in ("auto", "ytdlp"):
            ytdlp_assets = await self._ytdlp_search(keywords, orientation, limit)
            assets.extend(ytdlp_assets)

        # Strategy 2: Playwright (record web pages as video)
        if self._mode in ("auto", "playwright") and len(assets) < limit:
            pw_assets = await self._playwright_search(keywords, orientation, limit - len(assets))
            assets.extend(pw_assets)

        return assets[:limit]

    async def search_both(
        self,
        keywords: list[str],
        orientation: Orientation,
        limit_each: int = 5,
    ) -> list[MediaAsset]:
        """This provider only produces videos, so just delegates to search()."""
        return await self.search(keywords, orientation, "video", limit_each)

    # ---- yt-dlp path --------------------------------------------------------

    async def _ytdlp_search(
        self,
        keywords: list[str],
        orientation: Orientation,
        limit: int,
    ) -> list[MediaAsset]:
        """Search Bilibili and download short video segments."""
        if not _is_ytdlp_available():
            logger.debug("yt-dlp not installed; skipping video download path")
            return []

        # Search Bilibili
        bilibili_results = await _search_bilibili(keywords, limit=limit)
        if not bilibili_results:
            return []

        assets: list[MediaAsset] = []
        # Download videos concurrently (but limit concurrency)
        sem = asyncio.Semaphore(2)  # Max 2 concurrent downloads

        async def _download_one(item: dict[str, Any]) -> MediaAsset | None:
            async with sem:
                clip_dir = self._temp_dir / f"clip_{self._next_id()}"
                clip_dir.mkdir(parents=True, exist_ok=True)
                path = await _ytdlp_download_segment(
                    item["url"],
                    clip_dir,
                    max_duration_s=self._max_duration_s,
                )
                if path and path.exists():
                    return MediaAsset(
                        provider=self.name,
                        media_type="video",
                        url=item["url"],
                        width=self._rec_width,
                        height=self._rec_height,
                        duration_s=self._max_duration_s,
                        license="fair-use",
                        tags=tuple(keywords[:3]),
                        local_path=path,
                    )
                return None

        tasks = [_download_one(item) for item in bilibili_results[:limit]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, MediaAsset):
                assets.append(r)
            elif isinstance(r, Exception):
                logger.warning("yt-dlp download error: %s", r)

        return assets

    # ---- Playwright path ----------------------------------------------------

    async def _playwright_search(
        self,
        keywords: list[str],
        orientation: Orientation,
        limit: int,
    ) -> list[MediaAsset]:
        """Search the web and record relevant pages as video clips.

        Uses Playwright to both search and record. Takes screenshots during
        scrolling and combines them into a video with FFmpeg (avoids needing
        Playwright's built-in ffmpeg which has installation issues).
        """
        if not _is_playwright_available():
            logger.debug("Playwright not installed; skipping page recording path")
            return []

        # Adjust recording dimensions based on orientation
        w, h = self._rec_width, self._rec_height
        if orientation == "portrait":
            w, h = h, w  # 720x1280
        elif orientation == "square":
            w = h = min(w, h)

        query = " ".join(keywords[:3])

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return []

        assets: list[MediaAsset] = []

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": w, "height": h},
                    locale="zh-CN",
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                )

                # Pages to record: try multiple sources
                pages_to_record: list[tuple[str, str]] = []  # (title, url)

                # Source 1: Baidu search results
                page = await context.new_page()
                try:
                    search_url = f"https://www.baidu.com/s?wd={quote_plus(query)}"
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
                    await page.wait_for_timeout(2000)

                    # Extract result links
                    links = await page.eval_on_selector_all(
                        'h3 a[href]',
                        """els => els.map(el => ({
                            href: el.href,
                            title: el.textContent || ''
                        }))"""
                    )
                    for link in links[:limit]:
                        href = link.get("href", "")
                        title = link.get("title", "").strip()
                        if href and ("baidu.com/link?" in href or href.startswith("http")):
                            pages_to_record.append((title or query, href))
                except Exception as exc:
                    logger.debug("Baidu search failed: %s", exc)

                # Source 2: Baidu Baike (direct encyclopedia entry)
                try:
                    baike_url = f"https://baike.baidu.com/item/{quote_plus(query)}"
                    pages_to_record.append((query, baike_url))
                except Exception:
                    pass

                await page.close()

                # Record each page using screenshot-based approach
                for i, (title, url) in enumerate(pages_to_record[:limit]):
                    try:
                        rec_page = await context.new_page()
                        try:
                            await rec_page.goto(
                                url, wait_until="domcontentloaded", timeout=20000
                            )
                        except Exception:
                            await rec_page.close()
                            continue

                        await rec_page.wait_for_timeout(1500)

                        # Take screenshots during smooth scroll
                        clip_id = self._next_id()
                        frame_dir = self._temp_dir / f"frames_{clip_id}"
                        frame_dir.mkdir(parents=True, exist_ok=True)

                        total_h = await rec_page.evaluate("document.body.scrollHeight")
                        vp_h = await rec_page.evaluate("window.innerHeight")
                        scroll_step = max(vp_h // 3, 100)

                        # Calculate frames: aim for ~2 fps over max_duration_s
                        total_frames = self._max_duration_s * 2
                        scroll_positions = list(range(0, total_h, scroll_step))
                        # Sample evenly from scroll positions
                        if len(scroll_positions) > total_frames:
                            step = len(scroll_positions) // total_frames
                            scroll_positions = scroll_positions[::step][:total_frames]

                        for frame_idx, scroll_y in enumerate(scroll_positions):
                            await rec_page.evaluate(f"window.scrollTo(0, {scroll_y})")
                            await rec_page.wait_for_timeout(100)  # Shorter pause for faster capture
                            frame_path = frame_dir / f"frame_{frame_idx:04d}.jpg"
                            await rec_page.screenshot(path=str(frame_path), type="jpeg", quality=80)

                        await rec_page.close()

                        # Combine screenshots into video with FFmpeg
                        mp4_path = await self._frames_to_video(
                            frame_dir, clip_id, w, h, fps=2
                        )
                        if mp4_path and mp4_path.exists():
                            assets.append(MediaAsset(
                                provider=self.name,
                                media_type="video",
                                url=url,
                                width=w,
                                height=h,
                                duration_s=self._max_duration_s,
                                license="fair-use",
                                tags=tuple(keywords[:3]),
                                local_path=mp4_path,
                            ))

                    except Exception as exc:
                        logger.debug("Page recording failed for %s: %s", url, exc)
                        try:
                            await rec_page.close()
                        except Exception:
                            pass

                await context.close()
                await browser.close()

        except Exception as exc:
            logger.warning("Playwright recording session failed: %s", exc)

        return assets

    async def _frames_to_video(
        self,
        frame_dir: Path,
        clip_id: int,
        width: int,
        height: int,
        fps: int = 2,
    ) -> Path | None:
        """Combine screenshot frames into an MP4 video using FFmpeg.

        Uses ultrafast preset for speed; tries NVENC first if available.
        """
        output_path = self._temp_dir / f"page_{clip_id}.mp4"

        # Build cmd: try NVENC first (much faster), fallback to libx264
        use_nvenc = await self._supports_nvenc()
        if use_nvenc:
            codec = "h264_nvenc"
            preset_args = ["-preset", "p1"]
        else:
            codec = "libx264"
            preset_args = ["-preset", "ultrafast"]

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", codec,
            *preset_args,
            "-b:v", "1M",           # Low bitrate for screenshot videos
            "-pix_fmt", "yuv420p" if not use_nvenc else "nv12",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            # Clean up frame images
            shutil.rmtree(frame_dir, ignore_errors=True)
            if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024:
                return output_path
            # NVENC may fail (driver issues); fallback to libx264
            if use_nvenc:
                logger.warning("NVENC failed for frames->video, retrying with libx264")
                return await self._frames_to_video_fallback(frame_dir, clip_id, width, height, fps)
        except asyncio.TimeoutError:
            logger.warning("FFmpeg frames->video timed out for clip %d", clip_id)
        except Exception as exc:
            logger.warning("FFmpeg frames->video failed: %s", exc)
        shutil.rmtree(frame_dir, ignore_errors=True)
        return None

    async def _frames_to_video_fallback(
        self,
        frame_dir: Path,
        clip_id: int,
        width: int,
        height: int,
        fps: int = 2,
    ) -> Path | None:
        """Fallback: libx264 ultrafast."""
        output_path = self._temp_dir / f"page_{clip_id}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-b:v", "1M",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=180)
            if output_path.exists() and output_path.stat().st_size > 1024:
                return output_path
        except Exception as exc:
            logger.warning("FFmpeg fallback frames->video failed: %s", exc)
        return None

    async def _supports_nvenc(self) -> bool:
        """Check if ffmpeg supports h264_nvenc."""
        if not hasattr(self, "_nvenc_cached"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-encoders",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                self._nvenc_cached = b"h264_nvenc" in stdout
            except Exception:
                self._nvenc_cached = False
        return self._nvenc_cached

    # ---- Helpers ------------------------------------------------------------

    def _next_id(self) -> int:
        self._asset_counter += 1
        return self._asset_counter

    async def _convert_to_mp4(self, webm_path: Path) -> Path | None:
        """Convert WebM to MP4 using FFmpeg for pipeline compatibility."""
        mp4_path = webm_path.with_suffix(".mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(webm_path),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(mp4_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if mp4_path.exists() and mp4_path.stat().st_size > 1024:
                # Clean up WebM
                try:
                    webm_path.unlink()
                except OSError:
                    pass
                return mp4_path
        except Exception as exc:
            logger.warning("FFmpeg WebM->MP4 conversion failed: %s", exc)
        return None

    def cleanup(self) -> None:
        """Remove temporary files created by this provider."""
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
