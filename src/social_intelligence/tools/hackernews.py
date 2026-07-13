"""Hacker News Firebase API: discovers trending tech launches and discussions.

Security: HTTPS enforced for all Firebase API calls. TLS certificate validation
is enabled by default in httpx. Input parameters are validated and length-limited.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from ._freshness import freshness_weight
from ._http import get_with_retry

logger = logging.getLogger(__name__)

HN_BASE = "https://hacker-news.firebaseio.com/v0"
_VALID_CATEGORIES = {"top", "new", "best", "ask", "show"}
_FETCH_WORKERS = 10


def _fetch_item(sid: int) -> dict | None:
    """Fetch a single HN item by ID.

    Args:
        sid: Hacker News story ID.

    Returns:
        Parsed item dict, or None if the fetch fails or the item is not a story.
    """
    try:
        item_resp = get_with_retry(f"{HN_BASE}/item/{sid}.json", timeout=10.0)
        if item_resp.status_code != 200:
            return None
        item = item_resp.json()
        if not item or item.get("type") != "story":
            return None
        return item
    except (httpx.HTTPError, ValueError, KeyError):
        logger.debug("Failed to fetch HN item %d", sid)
        return None


def handle(params: dict) -> dict:
    """Fetch trending HN stories with optional keyword filtering.

    Args:
        params: category (str, one of top/new/best/ask/show), limit (int 1-30),
            keyword_filter (str, case-insensitive substring match on title)

    Returns:
        Dict with keys: stories (list), category (str), count (int), source (str).
        On error, adds an ``error`` key and returns empty stories.
    """
    category = params.get("category", "top")
    if category not in _VALID_CATEGORIES:
        return {"error": "Invalid category", "stories": [], "count": 0, "source": "Hacker News"}
    limit = max(1, min(int(params.get("limit", 10)), 30))
    keyword_filter = str(params.get("keyword_filter", ""))[:200]

    endpoint_map = {
        "top": "topstories",
        "new": "newstories",
        "best": "beststories",
        "ask": "askstories",
        "show": "showstories",
    }

    try:
        resp = get_with_retry(f"{HN_BASE}/{endpoint_map[category]}.json", timeout=15.0)
        resp.raise_for_status()
    except Exception:
        logger.exception("Hacker News API error for category '%s'", category)
        return {"stories": [], "category": category, "count": 0, "source": "Hacker News", "error": "upstream_error"}

    # Fetch the first 50 IDs, then resolve items in parallel
    story_ids = resp.json()[:50]

    items_by_id: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
        future_to_id = {executor.submit(_fetch_item, sid): sid for sid in story_ids}
        for future in as_completed(future_to_id):
            sid = future_to_id[future]
            try:
                item = future.result()
            except Exception:
                logger.debug("Unexpected error fetching HN item %d", sid)
                item = None
            if item is not None:
                items_by_id[sid] = item

    # Reconstruct in original score-descending order, respecting keyword filter and limit
    stories = []
    for sid in story_ids:
        if len(stories) >= limit:
            break
        item = items_by_id.get(sid)
        if item is None:
            continue
        title = item.get("title", "")
        if keyword_filter and keyword_filter.lower() not in title.lower():
            continue
        epoch = item.get("time", 0)
        stories.append(
            {
                "id": item["id"],
                "title": title,
                "url": item.get("url", f"https://news.ycombinator.com/item?id={item['id']}"),
                "score": item.get("score", 0),
                "author": item.get("by", ""),
                "comments": item.get("descendants", 0),
                "time": epoch,
                "freshness_weight": freshness_weight(epoch),
            }
        )

    return {"stories": stories, "category": category, "count": len(stories), "source": "Hacker News"}
