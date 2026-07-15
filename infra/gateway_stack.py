"""Amazon Bedrock AgentCore Gateway construction for the social intelligence tools.

This module owns the Gateway and its Lambda targets. The main stack
(`stacks/social_intelligence_stack.py`) calls `build_tools_gateway` to create the
Gateway inside the single CloudFormation stack — no separate deployment.

Swap or add data sources here:
    1. Implement the handler in `src/social_intelligence/tools/<name>.py`
    2. Add the route in `src/social_intelligence/tools/registry.py`
    3. Add the tool definition to `src/social_intelligence/schemas/tool_schema.json`
    4. Add the schema-to-route mapping only when its name differs from the route key
    5. Add the schema name to the responsible agent's allow-list in `entrypoint.py`

The Gateway uses IAM inbound auth (SigV4); the agent discovers tools over MCP.
"""

from __future__ import annotations

import os

from aws_cdk import aws_bedrockagentcore as agentcore
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
        The configured Gateway. The single Lambda target loads every tool declared
        in `tool_schema.json`.
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

    # AgentCore's managed WebSearch connector is currently available only in us-east-1.
    # The active eu-west-1 deployment cannot add its ``connectorId: "web-search"`` target.
    # The Trend agent accepts WebSearch automatically when a supported-region deployment
    # adds that MCP target in the future.

    return gateway
