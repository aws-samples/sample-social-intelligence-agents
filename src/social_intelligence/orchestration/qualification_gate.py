"""Deterministic eligibility checks for outbound email generation.

The model decides how to score a prospect, but persistence independently verifies
the two policy requirements from the reference architecture: a minimum score and
corroboration from distinct supported sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_SOURCE_ALIASES = {
    "devto": "devto",
    "dev.to": "devto",
    "github": "github",
    "hackernews": "hackernews",
    "hackernewsapi": "hackernews",
    "hn": "hackernews",
    "lobsters": "lobsters",
    "producthunt": "producthunt",
    "producthuntapi": "producthunt",
    "reddit": "reddit",
    "stackoverflow": "stackoverflow",
    "stackexchange": "stackoverflow",
    "wikipedia": "wikipedia",
    "youtube": "youtube",
}
_SOURCE_KEYS = ("source", "source_name", "provider", "origin")
_URL_KEYS = ("url", "source_url", "link")


@dataclass(frozen=True)
class EmailQualification:
    """The server-computed eligibility of one prospect for email generation."""

    score_qualified: bool
    source_qualified: bool
    independent_sources: tuple[str, ...]

    @property
    def independent_source_count(self) -> int:
        """Return the number of distinct recognized sources in the evidence."""
        return len(self.independent_sources)

    @property
    def email_eligible(self) -> bool:
        """Return whether both qualification requirements are satisfied."""
        return self.score_qualified and self.source_qualified


def assess_email_eligibility(
    score: object,
    evidence: object,
    *,
    score_threshold: int,
    min_independent_sources: int,
) -> EmailQualification:
    """Compute email eligibility from a typed score and structured source evidence.

    Only the nine public sources used by this sample count toward corroboration.
    This avoids treating a prospect's own website or an arbitrary model-supplied
    label as an independent research source.
    """
    sources = tuple(sorted(_collect_independent_sources(evidence)))
    score_qualified = isinstance(score, int) and not isinstance(score, bool) and score >= score_threshold
    return EmailQualification(
        score_qualified=score_qualified,
        source_qualified=len(sources) >= min_independent_sources,
        independent_sources=sources,
    )


def _collect_independent_sources(value: object) -> set[str]:
    """Recursively extract recognized source identities from JSON-compatible evidence."""
    sources: set[str] = set()
    if isinstance(value, Mapping):
        source = _source_from_mapping(value)
        if source:
            sources.add(source)
        for nested_value in value.values():
            sources.update(_collect_independent_sources(nested_value))
    elif isinstance(value, list):
        for item in value:
            sources.update(_collect_independent_sources(item))
    return sources


def _source_from_mapping(item: Mapping[str, Any]) -> str:
    """Return one recognized source from a mapping, preferring an explicit source label."""
    for key in _SOURCE_KEYS:
        source = _normalize_source(item.get(key))
        if source:
            return source
    for key in _URL_KEYS:
        source = _source_from_url(item.get(key))
        if source:
            return source
    return ""


def _normalize_source(value: object) -> str:
    """Normalize one source label to a supported source identity."""
    if not isinstance(value, str):
        return ""
    normalized = "".join(char for char in value.lower() if char.isalnum() or char == ".")
    return _SOURCE_ALIASES.get(normalized, "")


def _source_from_url(value: object) -> str:
    """Recognize one supported source from a URL without trusting arbitrary domains."""
    if not isinstance(value, str):
        return ""
    hostname = (urlparse(value).hostname or "").lower()
    if _matches_domain(hostname, "github.com"):
        return "github"
    if _matches_domain(hostname, "ycombinator.com") or _matches_domain(hostname, "hacker-news.firebaseio.com"):
        return "hackernews"
    if _matches_domain(hostname, "lobste.rs"):
        return "lobsters"
    if _matches_domain(hostname, "producthunt.com"):
        return "producthunt"
    if _matches_domain(hostname, "reddit.com"):
        return "reddit"
    if _matches_domain(hostname, "stackoverflow.com") or _matches_domain(hostname, "stackexchange.com"):
        return "stackoverflow"
    if _matches_domain(hostname, "wikipedia.org"):
        return "wikipedia"
    if _matches_domain(hostname, "youtube.com") or _matches_domain(hostname, "youtu.be"):
        return "youtube"
    if _matches_domain(hostname, "dev.to"):
        return "devto"
    return ""


def _matches_domain(hostname: str, domain: str) -> bool:
    """Return whether hostname is the source domain or one of its subdomains."""
    return hostname == domain or hostname.endswith(f".{domain}")
