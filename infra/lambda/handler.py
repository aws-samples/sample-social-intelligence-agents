"""Lambda handler for the social intelligence tools API.

Thin router that dispatches requests to shared tool handlers in tools/.
Supports two invocation patterns:

1. Amazon API Gateway proxy integration: path-based routing via {proxy+}
2. Amazon Bedrock AgentCore Gateway Lambda target: tool name from context.client_context.custom
   Per AWS docs, the Gateway passes tool metadata in the Lambda context object,
   NOT in the event body. The event body contains only the tool's input parameters.

Adding a new tool requires NO changes here — just add it to tools/registry.py.

Security: IAM authentication is required on both Amazon API Gateway and Amazon
Bedrock AgentCore Gateway endpoints. Callers must sign requests with SigV4.
Unhandled exceptions return a generic 500 error — no stack traces or internal
details are exposed. See SECURITY.md for IAM policy examples and the full
shared responsibility model.
"""

import json
import logging

from tools.registry import ROUTES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Map tool schema operationIds to route keys
_SCHEMA_TO_ROUTE = {
    "hackernews_trending": "hackernews",
    "youtube_trending": "youtube",
    "devto_trending": "devto",
    "wikipedia_summary": "wikipedia",
    "github_search": "github",
    "lobsters_trending": "lobsters",
    "producthunt_trending": "producthunt",
    "reddit_search": "reddit",
    "stackoverflow_search": "stackoverflow",
}

# Gateway target prefix delimiter (triple underscore per AWS docs)
_GATEWAY_DELIMITER = "___"


def _resolve_route(tool_name: str):
    """Resolve a tool name to a route handler, handling Gateway prefixes."""
    # Direct match (API Gateway path-based routing)
    if tool_name in ROUTES:
        return ROUTES[tool_name]

    # Strip Gateway target prefix if present (target_name___tool_name)
    if _GATEWAY_DELIMITER in tool_name:
        tool_name = tool_name[tool_name.index(_GATEWAY_DELIMITER) + len(_GATEWAY_DELIMITER) :]

    # Map schema operationId to route key
    route_key = _SCHEMA_TO_ROUTE.get(tool_name, tool_name)
    return ROUTES.get(route_key)


def _get_tool_name(event, context):
    """Extract tool name from Amazon Bedrock AgentCore Gateway context or API Gateway path.

    Amazon Bedrock AgentCore Gateway passes tool metadata in context.client_context.custom,
    NOT in the event body. The event body contains only input parameters.
    """
    # Amazon Bedrock AgentCore Gateway: tool name in Lambda context metadata
    try:
        custom = context.client_context.custom
        if custom and "bedrockAgentCoreToolName" in custom:
            tool_name = custom["bedrockAgentCoreToolName"]
            logger.info("Gateway tool_name from context: %s", tool_name)
            return tool_name
    except (AttributeError, TypeError):
        pass

    # API Gateway proxy integration: route from path
    proxy = (event.get("pathParameters") or {}).get("proxy", "")
    if proxy:
        logger.info("API Gateway tool_name from path: %s", proxy)
        return proxy

    return ""


def handler(event, context):
    """Route requests to the appropriate tool handler."""
    tool_name = _get_tool_name(event, context)

    if not tool_name:
        logger.warning("No tool_name resolved. Event keys: %s", list(event.keys()))
        return _response(400, {"error": "No tool name provided"})

    # For API Gateway events, parse the body; for Gateway events, event IS the params.
    # Each tool handler validates and bounds its own inputs (length caps, allowed-value
    # sets, regex) at its boundary, so the router passes the params through without
    # field selection. See tools/registry.py and the per-tool handle() functions.
    if "body" in event:
        raw_body = event.get("body", {})
        params = json.loads(raw_body) if isinstance(raw_body, str) else (raw_body or {})
    else:
        params = event

    route_fn = _resolve_route(tool_name)
    if not route_fn:
        logger.warning("Unknown tool: %s", tool_name)
        return _response(404, {"error": f"Unknown tool: {tool_name}"})

    try:
        result = route_fn(params)
        return _response(200, result)
    except Exception:
        logger.exception("Tool '%s' failed", tool_name)
        return _response(500, {"error": "Internal server error"})


def _response(status: int, body: dict) -> dict:
    """Format API Gateway proxy response."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }
