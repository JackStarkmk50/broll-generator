"""
Pinterest search for b-roll images and videos.

Uses Pinterest's web interface with proper session handling.
Falls back to parsing embedded JSON from search pages.
Pinterest actively blocks automated access, so this module
degrades gracefully — Pexels remains the reliable fallback.
"""
import json
import re
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup


@dataclass
class PinterestResult:
    pin_id: str
    title: str
    description: str
    image_url: str
    source_url: str
    width: int = 0
    height: int = 0
    is_video: bool = False
    video_url: Optional[str] = None
    board_name: str = ""


class PinterestSearcher:
    """
    Searches Pinterest for b-roll reference images and videos.

    Strategy:
    1. Try Pinterest's internal resource API (JSON)
    2. Fall back to HTML page parsing with embedded data extraction
    3. If all fails, return empty list (caller should use Pexels)
    """

    SEARCH_PAGE = "https://www.pinterest.com/search/pins/"
    RESOURCE_API = "https://www.pinterest.com/resource/BaseSearchResource/get/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        })
        self._cookies_initialized = False

    def _init_cookies(self):
        """Visit Pinterest to grab session cookies and CSRF token."""
        if self._cookies_initialized:
            return
        try:
            resp = self.session.get("https://www.pinterest.com/", timeout=10)
            if resp.status_code == 200:
                self._cookies_initialized = True
        except Exception as e:
            print(f"[Pinterest] Cookie init failed: {e}")

    def search(self, query: str, limit: int = 20) -> List[PinterestResult]:
        """
        Search Pinterest for pins matching query.
        Returns list of PinterestResult with image/video URLs.
        """
        self._init_cookies()

        # Strategy 1: Resource API (JSON)
        results = self._search_api(query, limit)
        if results:
            return results[:limit]

        # Strategy 2: HTML page parsing
        results = self._search_html(query, limit)
        if results:
            return results[:limit]

        print(f"[Pinterest] No results for '{query}'. Pinterest may be blocking requests.")
        return []

    def _search_api(self, query: str, limit: int) -> List[PinterestResult]:
        """Try Pinterest's internal BaseSearchResource API."""
        try:
            # Get CSRF token from cookies
            csrf = self.session.cookies.get("csrftoken", "")

            headers = {
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": csrf,
                "Accept": "application/json",
                "Referer": f"https://www.pinterest.com/search/pins/?q={query}",
            }

            params = {
                "source_url": f"/search/pins/?q={query}",
                "data": json.dumps({
                    "options": {
                        "query": query,
                        "scope": "pins",
                        "page_size": min(limit, 25),
                    },
                    "context": {},
                }),
            }

            resp = self.session.get(
                self.RESOURCE_API,
                params=params,
                headers=headers,
                timeout=15,
            )

            if resp.status_code == 200:
                data = resp.json()
                return self._parse_api_response(data)

        except Exception as e:
            print(f"[Pinterest] API search error: {e}")

        return []

    def _parse_api_response(self, data: dict) -> List[PinterestResult]:
        """Parse Pinterest BaseSearchResource JSON response."""
        results = []
        try:
            resource = data.get("resource_response", {})
            pins = resource.get("data", {}).get("results", [])

            for pin in pins:
                if not isinstance(pin, dict):
                    continue
                if pin.get("type") != "pin":
                    continue

                images = pin.get("images", {})
                # Prefer original size, fall back to 736px
                orig = images.get("orig", {}) or images.get("736x", {}) or {}
                image_url = orig.get("url", "")
                if not image_url:
                    continue

                # Check for video
                is_video = bool(pin.get("is_video") or pin.get("videos"))
                video_url = self._extract_video_url(pin)

                title = (
                    pin.get("title", "")
                    or pin.get("grid_title", "")
                    or (pin.get("rich_summary", {}) or {}).get("display_name", "")
                    or ""
                )
                description = pin.get("description", "") or pin.get("closeup_description", "") or ""

                results.append(PinterestResult(
                    pin_id=str(pin.get("id", "")),
                    title=title[:200] if title else "Pinterest Pin",
                    description=description[:500],
                    image_url=image_url,
                    source_url=f"https://www.pinterest.com/pin/{pin.get('id', '')}/",
                    width=orig.get("width", 0),
                    height=orig.get("height", 0),
                    is_video=is_video,
                    video_url=video_url,
                    board_name=(pin.get("board", {}) or {}).get("name", ""),
                ))

        except Exception as e:
            print(f"[Pinterest] API parse error: {e}")

        return results

    def _extract_video_url(self, pin: dict) -> Optional[str]:
        """Extract best video URL from pin data."""
        try:
            videos = pin.get("videos", {})
            if not videos:
                return None
            formats = videos.get("video_list", {})
            # Prefer higher quality
            for key in ["V_720P", "V_EXP7", "V_480P", "V_360P"]:
                if key in formats:
                    url = formats[key].get("url")
                    if url:
                        return url
            # Take any available
            for fmt in formats.values():
                if isinstance(fmt, dict) and fmt.get("url"):
                    return fmt["url"]
        except Exception:
            pass
        return None

    def _search_html(self, query: str, limit: int) -> List[PinterestResult]:
        """Fall back to parsing Pinterest search HTML page."""
        try:
            resp = self.session.get(
                self.SEARCH_PAGE,
                params={"q": query},
                timeout=15,
            )
            if resp.status_code != 200:
                return []

            # Pinterest embeds pin data in script tags as JSON
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []

            # Strategy A: Find __INITIAL_STATE__ or __PWS_DATA__
            for script in soup.find_all("script", {"type": "application/json"}):
                text = script.string or ""
                if len(text) < 100:
                    continue
                try:
                    blob = json.loads(text)
                    pins = self._extract_pins_from_blob(blob)
                    if pins:
                        results.extend(pins)
                except (json.JSONDecodeError, ValueError):
                    continue

            if results:
                return results[:limit]

            # Strategy B: Find embedded JSON in script tags (no type attr)
            for script in soup.find_all("script"):
                text = script.string or ""
                for pattern in [
                    r'window\.__INITIAL_STATE__\s*=\s*({.*?});?\s*</script>',
                    r'window\.__PWS_DATA__\s*=\s*({.*?});?\s*</script>',
                ]:
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        try:
                            blob = json.loads(match.group(1))
                            pins = self._extract_pins_from_blob(blob)
                            if pins:
                                results.extend(pins)
                        except (json.JSONDecodeError, ValueError):
                            continue

            if results:
                return results[:limit]

            # Strategy C: Regex extract image URLs from raw HTML
            image_urls = re.findall(
                r'https://i\.pinimg\.com/(?:originals|736x|564x)/[a-f0-9/]+\.\w+',
                resp.text,
            )
            seen = set()
            for i, url in enumerate(image_urls):
                if url not in seen:
                    seen.add(url)
                    results.append(PinterestResult(
                        pin_id=f"html_{i}",
                        title="Pinterest Pin",
                        description="",
                        image_url=url,
                        source_url=f"https://www.pinterest.com/search/pins/?q={query}",
                    ))
            return results[:limit]

        except Exception as e:
            print(f"[Pinterest] HTML search error: {e}")

        return []

    def _extract_pins_from_blob(self, blob: dict, depth: int = 0) -> List[PinterestResult]:
        """Recursively extract pin data from a nested JSON blob."""
        if depth > 5:
            return []

        results = []

        # Check if this dict looks like a pin
        if isinstance(blob, dict):
            if blob.get("type") == "pin" and blob.get("images"):
                images = blob["images"]
                orig = images.get("orig", {}) or images.get("736x", {}) or {}
                url = orig.get("url", "")
                if url:
                    results.append(PinterestResult(
                        pin_id=str(blob.get("id", "")),
                        title=blob.get("title", "") or blob.get("grid_title", "") or "Pinterest Pin",
                        description=blob.get("description", "")[:500],
                        image_url=url,
                        source_url=f"https://www.pinterest.com/pin/{blob.get('id', '')}/",
                        width=orig.get("width", 0),
                        height=orig.get("height", 0),
                        is_video=bool(blob.get("is_video") or blob.get("videos")),
                        video_url=self._extract_video_url(blob),
                    ))
                    return results

            # Recurse into dict values
            for v in blob.values():
                if isinstance(v, (dict, list)):
                    results.extend(self._extract_pins_from_blob(v, depth + 1))

        elif isinstance(blob, list):
            for item in blob:
                if isinstance(item, (dict, list)):
                    results.extend(self._extract_pins_from_blob(item, depth + 1))

        return results

    def get_pin_details(self, pin_id: str) -> Optional[PinterestResult]:
        """Fetch details for a specific pin by ID."""
        try:
            self._init_cookies()
            csrf = self.session.cookies.get("csrftoken", "")

            headers = {
                "X-Requested-With": "XMLHttpRequest",
                "X-CSRFToken": csrf,
                "Accept": "application/json",
            }

            params = {
                "source_url": f"/pin/{pin_id}/",
                "data": json.dumps({
                    "options": {"id": pin_id, "field_set_key": "detailed"},
                    "context": {},
                }),
            }

            resp = self.session.get(
                "https://www.pinterest.com/resource/PinResource/get/",
                params=params,
                headers=headers,
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                pin = data.get("resource_response", {}).get("data", {})
                if pin and pin.get("images"):
                    images = pin["images"]
                    orig = images.get("orig", {}) or images.get("736x", {}) or {}
                    return PinterestResult(
                        pin_id=str(pin.get("id", pin_id)),
                        title=pin.get("title", "") or "Pinterest Pin",
                        description=pin.get("description", "")[:500],
                        image_url=orig.get("url", ""),
                        source_url=f"https://www.pinterest.com/pin/{pin_id}/",
                        width=orig.get("width", 0),
                        height=orig.get("height", 0),
                        is_video=bool(pin.get("is_video") or pin.get("videos")),
                        video_url=self._extract_video_url(pin),
                    )

        except Exception as e:
            print(f"[Pinterest] Pin detail error: {e}")

        return None

    def download_image(self, url: str, save_path: str) -> bool:
        """Download an image from Pinterest to local path."""
        try:
            resp = self.session.get(url, timeout=20, stream=True)
            if resp.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                return True
        except Exception as e:
            print(f"[Pinterest] Download error: {e}")
        return False
