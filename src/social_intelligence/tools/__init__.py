"""Shared tool implementations for the social intelligence system.

This package contains two types of modules:

1. API tools (hackernews, youtube, devto, etc.): pure functions that call
   external APIs and return dicts. Deployed behind Lambda, discovered by
   agents via Amazon Bedrock AgentCore Gateway (MCP protocol).

2. Agent-side tools (brand_knowledge, email_renderer, dynamodb): tools that
   run in the agent process, not behind the Gateway/Lambda.

Architecture:
    Agent (Amazon Bedrock AgentCore Runtime)
        → Amazon Bedrock AgentCore Gateway (MCP, IAM auth)
            → Lambda handler imports from tools/*.py
        → Agent-side tools imported directly (@tool)

Security: All API tools run behind IAM-authenticated Lambda. External HTTP
calls use HTTPS with TLS certificate validation (httpx defaults). Input
parameters are validated and length-limited in each handler. See SECURITY.md
for the full threat model and shared responsibility details.
"""
