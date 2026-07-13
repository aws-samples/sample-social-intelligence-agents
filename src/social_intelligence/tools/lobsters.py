"""Lobste.rs JSON API: curated tech community discussions."""

import logging

from ._freshness import freshness_weight
from ._http import get_with_retry

logger = logging.getLogger(__name__)


def handle(params: dict) -> dict:
    """Fetch hottest stories from Lobste.rs.

    Args:
        params: limit (int 1-25, default 10)

    Returns:
        Dict with keys: stories (list), count (int), source (str).
        On error, adds an ``error`` key and returns empty stories.
    """
    limit = max(1, min(int(params.get("limit", 10)), 25))

    try:
        resp = get_with_retry(
            "https://lobste.rs/hottest.json",
            headers={"User-Agent": "AnyCompanyBot/1.0 (+https://example.com/bot)"},
            timeout=20.0,
            follow_redirects=True,
        )
        resp.raise_for_status()

        stories = []
        for s in resp.json()[:limit]:
            if not isinstance(s, dict):
                continue
            submitter = s.get("submitter_user", "")
            author = submitter.get("username", "") if isinstance(submitter, dict) else str(submitter)
            created_at = s.get("created_at", "")
            stories.append(
                {
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "score": s.get("score", 0),
                    "comments": s.get("comment_count", 0),
                    "author": author,
                    "tags": s.get("tags", []),
                    "created": created_at,
                    "freshness_weight": freshness_weight(created_at),
                }
            )
        return {"stories": stories, "count": len(stories), "source": "Lobste.rs"}
    except Exception:
        logger.exception("Lobste.rs API error")
        return {"stories": [], "count": 0, "source": "Lobste.rs", "error": "upstream_error"}
