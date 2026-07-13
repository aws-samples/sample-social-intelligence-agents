"""Amazon Bedrock AgentCore Gateway construction for the social intelligence tools.

This module owns the Gateway and its Lambda targets. The main stack
(`stacks/social_intelligence_stack.py`) calls `build_tools_gateway` to create the
Gateway inside the single CloudFormation stack — no separate deployment.

Swap or add data sources here:
    1. Implement the handler in `src/social_intelligence/tools/<name>.py`
    2. Add the route in `src/social_intelligence/tools/registry.py`
    3. Add the tool definition to `src/social_intelligence/schemas/tool_schema.json`
    4. Register a new Lambda target below with `gateway.add_lambda_target(...)`

The Gateway uses IAM inbound auth (SigV4); the agent discovers tools over MCP.
"""

from __future__ import annotations

import os

import aws_cdk.aws_bedrock_agentcore_alpha as agentcore
from aws_cdk import aws_lambda as lambda_
from constructs import Construct


def build_tools_gateway(
    scope: Construct,
    *,
    tools_lambda: lambda_.IFunction,
    project_root: str,
) -> agentcore.Gateway:
    """Create the AgentCore Gateway and register the shared tools Lambda target.

    Args:
        scope: The construct scope (the stack) the Gateway belongs to.
        tools_lambda: The Lambda function hosting the shared tool handlers.
        project_root: Repository root, used to locate the tool schema asset.

    Returns:
        The configured Gateway. Call `add_lambda_target` on it to register
        additional data sources without touching the main stack.
    """
    gateway = agentcore.Gateway(
        scope,
        "ToolsGateway",
        gateway_name="social-intel-gateway",
        description="MCP gateway for social intelligence tools",
        authorizer_configuration=agentcore.GatewayAuthorizer.using_aws_iam(),
    )

    schema_path = os.path.join(
        project_root,
        "src",
        "social_intelligence",
        "schemas",
        "tool_schema.json",
    )
    tool_schema = agentcore.ToolSchema.from_local_asset(schema_path)

    gateway.add_lambda_target(
        "ToolsLambdaTarget",
        gateway_target_name="social-intel-tools",
        description="Lambda target hosting shared tool handlers",
        lambda_function=tools_lambda,
        tool_schema=tool_schema,
    )

    # TODO: Web Search MCP target — not available as a native CDK construct in
    # aws-cdk.aws-bedrock-agentcore-alpha==2.238.0a0.  The module exposes
    # `gateway.add_mcp_server_target(...)` which accepts an existing MCP server URL,
    # but there is no managed "web search" target type.  Once AWS publishes a hosted
    # web-search MCP endpoint (see AgentCore Gateway docs:
    # https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-MCPservers.html),
    # register it here:
    #
    # gateway.add_mcp_server_target(
    #     "WebSearchTarget",
    #     gateway_target_name="social-intel-web-search",
    #     description="Web search via AgentCore-hosted MCP server",
    #     mcp_server_config=agentcore.McpServerTargetConfiguration.create(
    #         "<WEB_SEARCH_MCP_SERVER_ENDPOINT_URL>"
    #     ),
    # )

    return gateway
