"""Swarm orchestration runner: invokes the deployed Amazon Bedrock AgentCore endpoint with pattern='swarm'.

The Swarm pattern uses autonomous agent collaboration with dynamic handoffs.

Usage:
    export AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime
    python -m social_intelligence.orchestration.swarm_runner "Deep-dive on AI agent frameworks"

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

from social_intelligence.config import AWS_REGION

AGENT_ARN = os.environ["AGENTCORE_AGENT_ARN"]


def run_swarm(prompt: str, session_id: str = "") -> str:
    """Invoke the deployed agent with swarm pattern and stream results."""
    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    sid = session_id or str(uuid.uuid4())

    payload = json.dumps({"prompt": prompt, "pattern": "swarm"}).encode()
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
    print(f"[swarm_runner] Response received ({len(text)} chars)")
    return text


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Find recent AI tool launches and generate outreach emails"
    run_swarm(prompt)
