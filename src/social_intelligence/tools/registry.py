"""Tool registry: maps route names to handler functions.

Single source of truth for tool routing. Used by:
- Lambda handler (deployed): routes Amazon Bedrock AgentCore Gateway / Amazon API
  Gateway requests to handlers

To add a tool, follow CONTRIBUTING.md. This registry owns only route-name to
handler mapping; Gateway schema exposure and agent allow-lists are separate.

Security: All handlers are invoked through IAM-authenticated Lambda. Each handler
validates inputs (length limits, allowed values). The Lambda handler returns
generic 500 errors to callers; no stack traces or internal details are exposed.
See SECURITY.md for the full security architecture.
"""

from __future__ import annotations

from collections.abc import Callable

from . import (
    devto,
    github,
    hackernews,
    lobsters,
    producthunt,
    reddit,
    stackoverflow,
    wikipedia,
    youtube,
)

# Route name → handler function
# Tool route name -> callable. Gateway schema names map to these keys in the Lambda router.
ROUTES: dict[str, Callable[..., dict]] = {
    "hackernews": hackernews.handle,
    "youtube": youtube.handle,
    "devto": devto.handle,
    "wikipedia": wikipedia.handle,
    "github": github.handle,
    "lobsters": lobsters.handle,
    "producthunt": producthunt.handle,
    "reddit": reddit.handle,
    "stackoverflow": stackoverflow.handle,
}
