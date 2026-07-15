"""Wikipedia REST API: company/topic research enrichment."""

import logging
import re

import httpx

from ._http import get_with_retry

logger = logging.getLogger(__name__)


def handle(params: dict) -> dict:
    """Get a Wikipedia summary for a topic.

    Args:
        params: topic (str, max 200 chars; Unicode word chars, spaces, hyphens, parens)

    Returns:
        Dict with keys: title (str), extract (str), url (str), description (str),
        source (str). On error, adds an ``error`` key and returns empty strings.
    """
    topic = re.sub(r"[^\w\s()-]", "", str(params.get("topic", "")), flags=re.UNICODE)[:200]

    try:
        resp = get_with_retry(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}",
            headers={"User-Agent": "AnyCompanyBot/1.0 (social-intel-research)"},
            timeout=15.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info("Wikipedia has no summary for topic '%s'", topic)
            return {
                "title": "",
                "extract": "",
                "url": "",
                "description": "",
                "source": "Wikipedia",
            }
        logger.exception("Wikipedia API error for topic '%s'", topic)
        return {
            "title": "",
            "extract": "",
            "url": "",
            "description": "",
            "source": "Wikipedia",
            "error": "upstream_error",
        }
    except Exception:
        logger.exception("Wikipedia API error for topic '%s'", topic)
        return {
            "title": "",
            "extract": "",
            "url": "",
            "description": "",
            "source": "Wikipedia",
            "error": "upstream_error",
        }

    data = resp.json()
    return {
        "title": data.get("title", ""),
        "extract": data.get("extract", ""),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "description": data.get("description", ""),
        "source": "Wikipedia",
    }
