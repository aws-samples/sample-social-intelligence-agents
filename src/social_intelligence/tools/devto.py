"""dev.to (Forem) API: developer community sentiment and trending content."""

import logging
import re

from ._freshness import freshness_weight
from ._http import get_with_retry

logger = logging.getLogger(__name__)


def handle(params: dict) -> dict:
    """Fetch trending developer articles by tag.

    Args:
        params: tag (str, alphanumeric/underscore/hyphen, max 50 chars),
            time_window (int days 1-365, default 7), limit (int 1-30, default 10)

    Returns:
        Dict with keys: articles (list), tag (str), count (int), source (str).
        On error, adds an ``error`` key and returns empty articles.
    """
    tag = (re.sub(r"[^a-zA-Z0-9_-]", "", str(params.get("tag", "ai"))) or "ai")[:50]
    time_window = max(1, min(int(params.get("time_window", 7)), 365))
    limit = max(1, min(int(params.get("limit", 10)), 30))

    try:
        resp = get_with_retry(
            "https://dev.to/api/articles",
            params={"tag": tag, "top": time_window, "per_page": limit},
            timeout=15.0,
        )
        resp.raise_for_status()

        articles = []
        for a in resp.json():
            articles.append(
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "reactions": a.get("positive_reactions_count", 0),
                    "comments": a.get("comments_count", 0),
                    "author": a.get("user", {}).get("name", ""),
                    "published": a.get("published_at", ""),
                    "tags": a.get("tag_list", []),
                    "reading_time": a.get("reading_time_minutes", 0),
                    "description": (a.get("description", "") or "")[:200],
                    "freshness_weight": freshness_weight(a.get("published_at", "")),
                }
            )
        return {"articles": articles, "tag": tag, "count": len(articles), "source": "dev.to"}
    except Exception:
        logger.exception("dev.to API error for tag '%s'", tag)
        return {"articles": [], "tag": tag, "count": 0, "source": "dev.to", "error": "upstream_error"}
