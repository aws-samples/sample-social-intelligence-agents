"""Tests keeping the tool contract and its reference spec in agreement.

`tool_schema.json` is the wired contract: `infra/gateway_stack.py` hands it to the
Gateway target via `ToolSchema.from_local_asset`, so it alone drives tool discovery.
`openapi.yaml` documents the same HTTP surface but nothing loads it, which means
drift between the two is silent. These tests make it loud.
"""

import json
from importlib.resources import files

import yaml

SCHEMAS = files("social_intelligence.schemas")


def tool_schema_names() -> set[str]:
    """Return the tool names declared in the wired gateway contract."""
    tools = json.loads(SCHEMAS.joinpath("tool_schema.json").read_text(encoding="utf-8"))
    return {tool["name"] for tool in tools}


def openapi_operation_ids() -> set[str]:
    """Return the operation ids documented in the reference OpenAPI spec."""
    spec = yaml.safe_load(SCHEMAS.joinpath("openapi.yaml").read_text(encoding="utf-8"))
    return {operation["operationId"] for path in spec["paths"].values() for operation in path.values()}


def test_both_schemas_declare_tools() -> None:
    # Guard against a parse that silently yields nothing: an empty set would make
    # the drift test below pass without comparing anything.
    assert len(tool_schema_names()) == 9
    assert len(openapi_operation_ids()) == 9


def test_openapi_spec_documents_every_wired_tool() -> None:
    wired = tool_schema_names()
    documented = openapi_operation_ids()
    assert wired - documented == set(), "tool_schema.json declares tools missing from openapi.yaml"
    assert documented - wired == set(), "openapi.yaml documents tools absent from tool_schema.json"


def test_openapi_paths_match_operation_count() -> None:
    spec = yaml.safe_load(SCHEMAS.joinpath("openapi.yaml").read_text(encoding="utf-8"))
    assert len(spec["paths"]) == len(openapi_operation_ids())
