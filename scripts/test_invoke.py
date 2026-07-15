"""Quick test: invoke the deployed AgentCore Runtime with SSE streaming fix.

Usage:
    .venv/bin/python test_invoke.py [graph|swarm]

Requires the project venv — bedrock-agentcore registers the service model with botocore.
"""

import json
import os
import sys
import time
import uuid

try:
    import bedrock_agentcore  # noqa: F401 — registers the service model with botocore
except ImportError:
    print("ERROR: bedrock-agentcore package not found.")
    print("Run with the project venv: .venv/bin/python test_invoke.py [graph|swarm]")
    sys.exit(1)

from http.client import IncompleteRead

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ResponseStreamingError
from urllib3.exceptions import ProtocolError

# Verify the service model actually registered (requires botocore >= 1.42)
_available = boto3.Session().get_available_services()
if "bedrock-agentcore" not in _available:
    import botocore

    print(f"ERROR: bedrock-agentcore service not registered with botocore {botocore.__version__}.")
    print(f"  boto3={boto3.__version__}, botocore={botocore.__version__} (need >= 1.42)")
    print("Run with the project venv: .venv/bin/python test_invoke.py [graph|swarm]")
    sys.exit(1)

# Replace the fallback with your runtime ARN (see CDK output RuntimeArn).
AGENT_ARN = os.environ.get(
    "AGENTCORE_AGENT_ARN",
    "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/social_intel-XXXXXXXXXX",
)
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "default")


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "graph"

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client = session.client(
        "bedrock-agentcore",
        region_name=REGION,
        config=BotoConfig(read_timeout=900, retries={"total_max_attempts": 1}),
    )

    prompt = (
        "Find 3 recent AI tool launches on Hacker News relevant to SaaS and Developer Tools. "
        "Score each prospect and generate outreach emails for those scoring 60+."
    )
    payload = json.dumps({"prompt": prompt, "pattern": pattern}).encode()
    sid = str(uuid.uuid4())

    print(f"Invoking AgentCore Runtime (pattern={pattern})...")
    print(f"  ARN: {AGENT_ARN}")
    print(f"  Session: {sid}")
    print(f"  Prompt: {prompt[:80]}...")
    print()

    t0 = time.time()

    response = client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_ARN,
        runtimeSessionId=sid,
        payload=payload,
        contentType="application/json",
        accept="text/event-stream, application/json",
        qualifier="DEFAULT",
    )

    content_type = response.get("contentType", "")
    print(f"Response contentType: {content_type}")
    print(f"Response keys: {list(response.keys())}")
    print()

    body = response.get("response")
    if body is None:
        print("ERROR: No response body")
        return

    is_sse = "text/event-stream" in content_type
    print(f"SSE mode: {is_sse}")
    print(f"Body type: {type(body).__name__}")
    print(f"Body methods: {[m for m in dir(body) if not m.startswith('_') and callable(getattr(body, m, None))]}")
    print()

    bytes_read = 0
    line_count = 0
    event_counts: dict[str, int] = {}

    # Robust drain for both SSE and raw streams: iterate raw byte chunks and split
    # on newlines in a buffer. iter_lines() on a botocore StreamingBody can stop at
    # a chunk boundary that lacks a trailing newline, ending the stream early (it was
    # cutting off at node transitions). Buffering raw bytes and flushing the trailing
    # line at the end drains the complete response regardless of chunk boundaries.
    print("Draining stream (buffered byte chunks)...")

    def _handle_line(line: str) -> None:
        nonlocal line_count
        line = line.strip()
        # Skip blanks and SSE keepalive/comment lines (':' prefix per the SSE spec).
        if not line or line.startswith(":"):
            return
        line_count += 1
        data = line[6:] if line.startswith("data: ") else line
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError, TypeError:
            return
        if not isinstance(parsed, dict):
            return
        evt_type = parsed.get("type", "unknown")
        event_counts[evt_type] = event_counts.get(evt_type, 0) + 1
        elapsed = time.time() - t0
        if evt_type == "multiagent_node_start":
            print(f"  [{elapsed:6.1f}s] NODE START: {parsed.get('node_id', '?')}")
        elif evt_type in ("multiagent_node_stop", "multiagent_node_complete"):
            print(f"  [{elapsed:6.1f}s] NODE END:   {parsed.get('node_id', '?')}")
        elif evt_type == "multiagent_result":
            print(f"  [{elapsed:6.1f}s] PIPELINE COMPLETE")
        elif evt_type == "multiagent_node_stream":
            event_data = parsed.get("event", {})
            msg = event_data.get("message", {}) if isinstance(event_data, dict) else {}
            for block in msg.get("content", []) if isinstance(msg, dict) else []:
                tu = block.get("toolUse", {}) if isinstance(block, dict) else {}
                if tu and tu.get("name"):
                    print(f"  [{elapsed:6.1f}s]   TOOL: {tu['name']}")

    disconnected = False
    try:
        buffer = ""
        # iter_chunks() yields raw bytes without line-boundary assumptions; fall back
        # to iterating the body directly if the stream type lacks it.
        chunks = body.iter_chunks(chunk_size=4096) if hasattr(body, "iter_chunks") else body
        for chunk in chunks:
            raw = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
            bytes_read += len(raw)
            buffer += raw
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                _handle_line(line)
            if line_count and line_count % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  ... {line_count} lines, {bytes_read / 1024 / 1024:.1f} MB, {elapsed:.0f}s")
        # Flush any trailing line not terminated by a newline.
        if buffer.strip():
            _handle_line(buffer)
    except (ResponseStreamingError, ProtocolError, IncompleteRead) as e:
        # The AgentCore app cancels its producer when the response consumer disconnects.
        # A partial stream is not a successful invocation and must be retried.
        disconnected = True
        if buffer.strip():
            _handle_line(buffer)
        print(f"\n[client disconnect after {time.time() - t0:.0f}s - invocation cancelled] {type(e).__name__}")
    except Exception as e:
        print(f"\nStream error: {type(e).__name__}: {e}")
    if disconnected:
        print("NOTE: partial output is invalid. Rerun the invocation after resolving the stream error.")

    elapsed = time.time() - t0
    print()
    print("=== RESULTS ===")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Bytes read: {bytes_read:,} ({bytes_read / 1024 / 1024:.2f} MB)")
    print(f"Lines parsed: {line_count:,}")
    print(f"Event types: {json.dumps(event_counts, indent=2)}")


if __name__ == "__main__":
    main()
