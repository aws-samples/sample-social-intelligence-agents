"""End-to-end test — invoke Amazon Bedrock AgentCore Runtime and parse the EventStream response.

Requires AGENTCORE_AGENT_ARN environment variable. Skipped automatically when not set.

Usage:
    export AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime
    pytest tests/integration/test_runtime.py -v
"""

import json
import os
import re
import uuid

import pytest

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "default")
AGENT_ARN = os.environ.get("AGENTCORE_AGENT_ARN", "")

pytestmark = pytest.mark.skipif(not AGENT_ARN, reason="AGENTCORE_AGENT_ARN not set")


@pytest.fixture(scope="module")
def agentcore_client():
    import boto3

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    return session.client("bedrock-agentcore", region_name=REGION)


def test_invoke_graph_pattern(agentcore_client):
    """Invoke with graph pattern and verify SSE stream contains expected event types."""
    payload = json.dumps(
        {
            "prompt": "Find the top 3 trending stories on Hacker News right now. Just list them briefly.",
            "pattern": "graph",
        }
    ).encode()

    sid = f"test-{uuid.uuid4()}"
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_ARN,
        runtimeSessionId=sid,
        payload=payload,
    )

    body = response.get("body") or response.get("response")
    assert body is not None, "No response body"

    event_count = 0
    node_starts: set[str] = set()
    node_stops: set[str] = set()
    tool_calls: list[str] = []
    buffer = ""

    for chunk in body:
        buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            raw = line[6:] if line.startswith("data: ") else line
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError, TypeError:
                continue
            if not isinstance(parsed, dict):
                continue

            evt_type = parsed.get("type", "")
            event_count += 1

            if evt_type == "multiagent_node_start":
                node_starts.add(parsed.get("node_id", ""))
            elif evt_type == "multiagent_node_stop":
                node_stops.add(parsed.get("node_id", ""))
            elif evt_type == "multiagent_node_stream":
                event_data = parsed.get("event", {})
                if isinstance(event_data, dict):
                    inner = event_data.get("event", {})
                    if isinstance(inner, dict):
                        cbs = inner.get("contentBlockStart", {})
                        if isinstance(cbs, dict):
                            start = cbs.get("start", {})
                            if isinstance(start, dict) and "toolUse" in start:
                                tname = start["toolUse"].get("name", "")
                                if tname:
                                    tool_calls.append(re.sub(r"^[^_]+___", "", tname))

    assert event_count > 0, "No events received from stream"
    assert len(node_starts) > 0, "No node_start events"
    assert len(tool_calls) > 0, f"No tool calls detected (events: {event_count})"
