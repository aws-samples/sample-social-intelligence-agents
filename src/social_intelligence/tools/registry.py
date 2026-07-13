"""Tool registry: maps route names to handler functions.

Single source of truth for tool routing. Used by:
- Lambda handler (deployed): routes Amazon Bedrock AgentCore Gateway / Amazon API
  Gateway requests to handlers

Adding a new tool:
1. Create tools/<name>.py with a handle(params: dict) -> dict function
2. Add the route to ROUTES below
3. Add the tool definition to schemas/tool_schema.json

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
# Adding a tool = one line here + one file in tools/ + one path in tool_schema.json
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
