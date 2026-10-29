from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import requests
import re
from agent.state import StockCandidate
from config.settings import settings
from core.pinterest_searcher import PinterestSearcher
from core.query_reasoner import QueryReasoner, ReasonedQuery

PEXELS_VIDEOS_ENDPOINT = "https://api.pexels.com/videos/search"
PEXELS_PHOTOS_ENDPOINT = "https://api.pexels.com/v1/search"

# Curated royalty-free fallback catalog for offline or keyless testing
FALLBACK_STOCK_CATALOG = [
    {
        "id": "mock_vid_1",
        "type": "video",
        "title": "Cinematic misty mountain peaks and pine forest",
        "url": "https://www.pexels.com/video/misty-mountains-1",
        "download_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "preview_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&auto=format&fit=crop&q=80",
        "tags": ["mountain", "misty", "drone", "nature", "landscape", "aerial", "forest"],
    },
    {
        "id": "mock_vid_2",
        "type": "video",
        "title": "Futuristic neon-lit city cyber skyline",
        "url": "https://www.pexels.com/video/cyber-city-2",
        "download_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
        "preview_url": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800&auto=format&fit=crop&q=80",
        "tags": ["city", "neon", "cyber", "technology", "urban", "night", "skyline", "modern"],
    },
    {
        "id": "mock_img_1",
        "type": "image",
        "title": "Creative video editor working at dual monitor setup",
        "url": "https://www.pexels.com/photo/creative-studio-3",
        "download_url": "https://images.unsplash.com/photo-1531403009284-440f080d1e12?w=1920&auto=format&fit=crop&q=80",
        "preview_url": "https://images.unsplash.com/photo-1531403009284-440f080d1e12?w=800&auto=format&fit=crop&q=80",
        "tags": ["technology", "creators", "build", "stories", "studio", "computer", "editing"],
    },
    {
        "id": "mock_vid_3",
        "type": "video",
        "title": "Fast-paced corporate modern workflow and bustling street",
        "url": "https://www.pexels.com/video/bustling-street-4",
        "download_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
        "preview_url": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&auto=format&fit=crop&q=80",
        "tags": ["world", "fast-paced", "corporate", "speed", "modern", "business", "busy"],
    },
]


class StockSourcer:
    """
    Sourcer node: searches Pinterest and Pexels for b-roll candidates.
    Falls back to curated catalog when offline or no API key.
    """

    def __init__(self, api_key: Optional[str] = None, enable_pinterest: bool = True,
                 enable_pexels: bool = True, custom_instructions: str = ""):
        self.api_key = api_key or settings.PEXELS_API_KEY
        self.enable_pinterest = enable_pinterest and settings.ENABLE_PINTEREST
        self.enable_pexels = enable_pexels and settings.ENABLE_PEXELS
        self._pinterest = PinterestSearcher() if self.enable_pinterest else None
        self.reasoner = QueryReasoner(custom_instructions=custom_instructions)

    def generate_search_queries(self, beat_text: str, keywords: List[str]) -> List[str]:
        """Use QueryReasoner to analyze text and generate smart visual queries."""
        reasoned = self.reasoner.reason(beat_text, keywords)
        return reasoned.queries or ["cinematic b-roll"]

    def reason_and_explain(self, beat_text: str, keywords: List[str]) -> ReasonedQuery:
        """Full reasoning output (queries + analysis + mood + concepts)."""
        return self.reasoner.reason(beat_text, keywords)

    def search_all(self, query: str, per_page: int = 6,
                   exclude_urls: Optional[Set[str]] = None) -> List[StockCandidate]:
        """
        Search all enabled sources and merge results.
        exclude_urls: set of source_url or download_url already saved (dedup).
        """
        exclude_urls = exclude_urls or set()
        candidates: List[StockCandidate] = []

        # Pinterest search
        if self._pinterest:
            pinterest_results = self._search_pinterest(query)
            candidates.extend(pinterest_results)

        # Pexels search
        if self.enable_pexels:
            pexels_results = self.search_pexels(query, per_page=per_page)
            candidates.extend(pexels_results)

        # Fallback if nothing found
        if not candidates:
            candidates = self._search_fallback(query)

        # Dedup: remove already-saved assets
        if exclude_urls:
            before = len(candidates)
            candidates = [
                c for c in candidates
                if c.source_url not in exclude_urls and c.download_url not in exclude_urls
            ]
            filtered = before - len(candidates)
            if filtered > 0:
                print(f"[StockSourcer] Filtered {filtered} already-saved duplicates.")

        return candidates

    def _search_pinterest(self, query: str) -> List[StockCandidate]:
        """Search Pinterest and convert results to StockCandidate."""
        if not self._pinterest:
            return []

        try:
            pins = self._pinterest.search(query, limit=settings.PINTEREST_RESULTS_PER_QUERY)
            candidates = []
            for pin in pins:
                if not pin.image_url:
                    continue
                media_type = "video" if (pin.is_video and pin.video_url) else "image"
                download_url = pin.video_url if media_type == "video" else pin.image_url
                candidates.append(StockCandidate(
                    id=f"pinterest_{pin.pin_id}",
                    media_type=media_type,
                    title=pin.title or pin.description[:80] or f"Pinterest: {query}",
                    source_url=pin.source_url,
                    download_url=download_url,
                    preview_url=pin.image_url,
                    width=pin.width or 1920,
                    height=pin.height or 1080,
                    duration=None,
                    source="pinterest",
                ))
            return candidates
        except Exception as e:
            print(f"[StockSourcer] Pinterest search error: {e}")
            return []

    def search_pexels(self, query: str, per_page: int = 4) -> List[StockCandidate]:
        """Query Pexels Video and Photo APIs."""
        if not self.api_key:
            return self._search_fallback(query)

        headers = {"Authorization": self.api_key}
        candidates: List[StockCandidate] = []

        # Videos
        try:
            v_res = requests.get(
                PEXELS_VIDEOS_ENDPOINT,
                headers=headers,
                params={"query": query, "per_page": per_page, "orientation": "landscape"},
                timeout=10,
            )
            if v_res.status_code == 200:
                for v in v_res.json().get("videos", []):
                    v_files = v.get("video_files", [])
                    hd_files = [f for f in v_files if f.get("quality") == "hd"] or v_files
                    download_url = hd_files[0]["link"] if hd_files else None
                    preview_pic = v.get("image", "")
                    if download_url and preview_pic:
                        candidates.append(StockCandidate(
                            id=f"pexels_v_{v['id']}",
                            media_type="video",
                            title=f"Pexels Video {v['id']} ({query})",
                            source_url=v.get("url", ""),
                            download_url=download_url,
                            preview_url=preview_pic,
                            width=v.get("width", 1920),
                            height=v.get("height", 1080),
                            duration=v.get("duration", 5.0),
                            source="pexels",
                        ))
        except Exception as ve:
            print(f"[StockSourcer] Pexels video error: {ve}")

        # Photos
        try:
            p_res = requests.get(
                PEXELS_PHOTOS_ENDPOINT,
                headers=headers,
                params={"query": query, "per_page": per_page, "orientation": "landscape"},
                timeout=10,
            )
            if p_res.status_code == 200:
                for p in p_res.json().get("photos", []):
                    src = p.get("src", {})
                    download_url = src.get("large2x") or src.get("original") or src.get("large")
                    preview_url = src.get("medium") or src.get("small")
                    if download_url and preview_url:
                        candidates.append(StockCandidate(
                            id=f"pexels_p_{p['id']}",
                            media_type="image",
                            title=f"Pexels Photo by {p.get('photographer', 'Unknown')} ({query})",
                            source_url=p.get("url", ""),
                            download_url=download_url,
                            preview_url=preview_url,
                            width=p.get("width", 1920),
                            height=p.get("height", 1080),
                            source="pexels",
                        ))
        except Exception as pe:
            print(f"[StockSourcer] Pexels photo error: {pe}")

        if not candidates:
            return self._search_fallback(query)

        return candidates

    def _search_fallback(self, query: str) -> List[StockCandidate]:
        """Offline fallback using curated catalog."""
        query_words = set(query.lower().split())
        scored = []
        for item in FALLBACK_STOCK_CATALOG:
            overlap = len(query_words.intersection(set(item["tags"])))
            scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[StockCandidate] = []
        for _, item in scored:
            results.append(StockCandidate(
                id=item["id"],
                media_type=item["type"],
                title=item["title"],
                source_url=item["url"],
                download_url=item["download_url"],
                preview_url=item["preview_url"],
                duration=5.0 if item["type"] == "video" else None,
                source="fallback",
            ))
        return results
