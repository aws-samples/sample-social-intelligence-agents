"""YouTube Data API v3: collects trending video data."""

import logging
import re
from datetime import datetime, timedelta, timezone

from ._freshness import freshness_weight
from ._http import get_with_retry
from ._secrets import get_secret

logger = logging.getLogger(__name__)
_YOUTUBE_API_KEY_PATTERN = re.compile(r"^AIza[0-9A-Za-z_-]{35}$")


def _recent_cutoff(days: int = 90) -> str:
    """Return an ISO 8601 timestamp N days ago for the publishedAfter filter.

    Args:
        days: Number of days before now to use as the cutoff.

    Returns:
        ISO 8601 UTC timestamp string suitable for the YouTube API publishedAfter param.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return cutoff.strftime("%Y-%m-%dT00:00:00Z")


def handle(params: dict) -> dict:
    """Search YouTube for trending videos related to a topic.

    Args:
        params: query (str, max 500 chars), max_results (int 1-10, default 5)

    Returns:
        Dict with keys: videos (list), query (str), count (int), source (str).
        On error, adds an ``error`` key and returns empty videos.
    """
    query = str(params.get("query", ""))[:500]
    max_results = max(1, min(int(params.get("max_results", 5)), 10))

    try:
        api_key = get_secret("social-intel/youtube-api-key").strip()
    except Exception:
        logger.info("YouTube API key secret unavailable; skipping YouTube enrichment")
        return {"videos": [], "query": query, "count": 0, "source": "YouTube", "error": "not_configured"}
    if not _YOUTUBE_API_KEY_PATTERN.fullmatch(api_key):
        # The stack creates a random bootstrap secret. It is not a Google API key
        # and must not be sent to the API as though it were one.
        logger.info("YouTube API key is not configured; skipping YouTube enrichment")
        return {"videos": [], "query": query, "count": 0, "source": "YouTube", "error": "not_configured"}

    try:
        search_resp = get_with_retry(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "key": api_key,
                "q": query,
                "part": "snippet",
                "type": "video",
                "order": "viewCount",
                "maxResults": max_results,
                "publishedAfter": _recent_cutoff(),
            },
            timeout=30.0,
        )
        search_resp.raise_for_status()
        video_ids = [item["id"]["videoId"] for item in search_resp.json().get("items", [])]
        if not video_ids:
            return {"videos": [], "query": query, "count": 0, "source": "YouTube"}

        stats_resp = get_with_retry(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"key": api_key, "id": ",".join(video_ids), "part": "statistics,snippet"},
            timeout=30.0,
        )
        stats_resp.raise_for_status()

        videos = []
        for item in stats_resp.json().get("items", []):
            published_at = item["snippet"]["publishedAt"]
            videos.append(
                {
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "likes": int(item["statistics"].get("likeCount", 0)),
                    "published_at": published_at,
                    "url": f"https://youtube.com/watch?v={item['id']}",
                    "freshness_weight": freshness_weight(published_at),
                }
            )
        return {"videos": videos, "query": query, "count": len(videos), "source": "YouTube"}
    except Exception:
        logger.exception("YouTube API error for query '%s'", query)
        return {"videos": [], "query": query, "count": 0, "source": "YouTube", "error": "upstream_error"}
