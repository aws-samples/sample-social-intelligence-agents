"""Graph orchestration runner: invokes the deployed Amazon Bedrock AgentCore endpoint with pattern='graph'.

The Graph pattern uses a deterministic DAG:

    research ──┐
               ├──→ analysis (waits for BOTH) ──→ email (if score ≥ 60)
    search   ──┘

Usage:
    export AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime
    python -m social_intelligence.orchestration.graph_runner "Find recent AI tool launches"

Note: This is a local development and testing utility, not production code. It prints
the raw agent response to stdout for inspection. Do not run it against real prospect
data in a shared environment, since the response may include recalled content.
"""

import json
import os
import sys
import uuid

import bedrock_agentcore  # noqa: F401 (registers the service model with botocore)
import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
AGENT_ARN = os.environ["AGENTCORE_AGENT_ARN"]


def run_graph(prompt: str, session_id: str = "") -> str:
    """Invoke the deployed agent with graph pattern and stream results."""
    client = boto3.client("bedrock-agentcore", region_name=REGION)
    sid = session_id or str(uuid.uuid4())

    payload = json.dumps({"prompt": prompt, "pattern": "graph"}).encode()
    response = client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_ARN,
        runtimeSessionId=sid,
        payload=payload,
    )

    stream = response.get("response")
    if not stream:
        print("No response stream")
        return ""

    content = stream.read()
    text = content.decode("utf-8", errors="replace")
    # Print metadata only: do not log raw prospect data in shared environments.
    print(f"[graph_runner] Response received ({len(text)} chars)")
    return text


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Find recent AI tool launches and generate outreach emails"
    run_graph(prompt)
