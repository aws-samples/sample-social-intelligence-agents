"""Regression tests for the live evaluation harness."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_quality.py"


def _load_eval_quality_module():
    spec = importlib.util.spec_from_file_location("eval_quality_test_module", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sse_parser_reassembles_split_event():
    module = _load_eval_quality_module()
    chunks = [
        b'data: {"type":"async_started","run_id":"abc"',
        b"}\n\n",
    ]

    events = list(module._iter_sse_json_events(chunks))

    assert events == [{"type": "async_started", "run_id": "abc"}]


def test_golden_score_ranges_are_type_specific():
    module = _load_eval_quality_module()

    assert (
        module._validate_golden_set(
            [
                {"prompt": "email", "expected_type": "email_draft", "min_score": 7},
                {"prompt": "low fit", "expected_type": "scored_prospect", "min_score": 0, "max_score": 49},
            ]
        )
        == []
    )

    errors = module._validate_golden_set(
        [
            {"prompt": "bad email", "expected_type": "email_draft", "min_score": 0},
            {"prompt": "bad range", "expected_type": "scored_prospect", "min_score": 80, "max_score": 49},
        ]
    )

    assert any("email_draft" in error for error in errors)
    assert any("cannot exceed" in error for error in errors)


def test_records_query_the_session_index_without_scanning():
    module = _load_eval_quality_module()
    table = MagicMock()
    row = {"email_body": "Hello", "score": 80, "product_name": "Acme", "dedup_partition": "LEAD"}
    table.query.return_value = {"Items": [row]}

    with patch("boto3.resource") as resource:
        resource.return_value.Table.return_value = table
        records = module._records_from_dynamodb("us-east-1", "run-123")

    assert records == [row]
    assert table.query.call_args.kwargs["IndexName"] == "session-id-discovered-at-index"
    table.scan.assert_not_called()


def test_start_background_run_returns_acknowledged_run_id():
    """A background run returns as soon as the runtime acknowledges it; results are then
    read from DynamoDB by run_id rather than by holding the response stream open."""
    module = _load_eval_quality_module()
    body = MagicMock()
    body.iter_chunks.return_value = [b'data: {"type":"async_started","run_id":"run-1"}\n\n']

    with patch("boto3.client") as client:
        client.return_value.invoke_agent_runtime.return_value = {"response": body}
        result = module._start_background_run(
            "arn:aws:bedrock-agentcore:us-east-1:123:runtime/test", "test", "us-east-1", "run-1"
        )

    assert result == "run-1"
    payload = json.loads(client.return_value.invoke_agent_runtime.call_args.kwargs["payload"])
    assert payload["background"] is True
    client_config = client.call_args.kwargs["config"]
    assert client_config.read_timeout == 300
    assert client_config.connect_timeout == 10
    body.close.assert_called_once()


def test_start_background_run_raises_without_acknowledgment():
    """If the runtime never sends async_started, the run is not confirmed started."""
    module = _load_eval_quality_module()
    body = MagicMock()
    body.iter_chunks.return_value = [b'data: {"type":"multiagent_node_start"}\n\n']

    with patch("boto3.client") as client:
        client.return_value.invoke_agent_runtime.return_value = {"response": body}
        try:
            module._start_background_run(
                "arn:aws:bedrock-agentcore:us-east-1:123:runtime/test", "test", "us-east-1", "run-1"
            )
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "did not acknowledge" in str(exc)


def test_eval_patterns_reject_invalid_value_and_expand_both():
    module = _load_eval_quality_module()

    assert module._eval_patterns(" both ") == ["graph", "swarm"]
    try:
        module._eval_patterns("grahp")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "EVAL_PATTERN" in str(exc)


def test_swarm_scored_prospect_reads_persisted_scores_without_lead_fallback():
    module = _load_eval_quality_module()
    item = {
        "prompt": "Find consumer mobile gaming apps and score them",
        "expected_type": "scored_prospect",
        "min_score": 0,
        "max_score": 49,
    }

    with patch.object(module, "_start_background_run"):
        with patch.object(
            module,
            "_wait_for_terminal_run",
            return_value=("succeeded", [{"score": 30}]),
        ) as wait:
            result = module._evaluate_prompt(item, 0, "arn", "us-east-1", MagicMock(), pattern="swarm")

    assert result["pass"] is True
    assert result["metric"] == 30.0
    wait.assert_called_once()


def test_wait_for_terminal_run_returns_failed_status_without_settling():
    module = _load_eval_quality_module()
    marker = {"product_name": "__run_status__:failed"}

    with patch.object(module, "_records_from_dynamodb", return_value=[marker]):
        with patch.object(module.time, "sleep") as sleep:
            status, records = module._wait_for_terminal_run("us-east-1", "run-1", timeout_s=1260, delay_s=15)

    assert status == "failed"
    assert records == [marker]
    sleep.assert_not_called()


def test_evaluate_prompt_rejects_failed_background_run():
    module = _load_eval_quality_module()
    item = {"prompt": "Find prospects", "expected_type": "scored_prospect", "min_score": 0}

    with patch.object(module, "_start_background_run"):
        with patch.object(module, "_wait_for_terminal_run", return_value=("failed", [])):
            result = module._evaluate_prompt(item, 0, "arn", "us-east-1", MagicMock())

    assert result["pass"] is False
    assert result["metric"] == 0.0


def test_evaluate_prompt_does_not_count_graph_recovery_as_swarm_success():
    """A successful fallback preserves availability but is not a native Swarm pass."""
    module = _load_eval_quality_module()
    item = {"prompt": "Find prospects", "expected_type": "scored_prospect", "min_score": 0}
    marker = {"product_name": "__run_status__:succeeded"}

    with patch.object(module, "_start_background_run"):
        with patch.object(module, "_wait_for_terminal_run", return_value=("succeeded", [marker])):
            with patch.object(module, "_run_execution_path", return_value="graph_recovery"):
                result = module._evaluate_prompt(item, 0, "arn", "us-east-1", MagicMock(), pattern="swarm")

    assert result["pass"] is False
    assert result["execution_path"] == "graph_recovery"


def test_judge_rejects_out_of_range_scores():
    module = _load_eval_quality_module()
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": module._JUDGE_TOOL_NAME,
                            "input": {"relevance": 100, "grounding": 100},
                        }
                    }
                ]
            }
        }
    }

    assert module._judge_email(client, "prompt", "body", "{}") == {"relevance": 0, "grounding": 0}


def test_judge_forces_and_reads_structured_tool_scores():
    module = _load_eval_quality_module()
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": module._JUDGE_TOOL_NAME,
                            "input": {"relevance": 8, "grounding": 9},
                        }
                    }
                ]
            }
        }
    }

    assert module._judge_email(client, "prompt", "body", "{}") == {"relevance": 8, "grounding": 9}
    assert client.converse.call_args.kwargs["toolConfig"] == module._JUDGE_TOOL_CONFIG


def test_judge_schema_avoids_unsupported_numeric_range_keywords():
    """Bedrock Converse strict tool schemas reject integer minimum/maximum constraints."""
    module = _load_eval_quality_module()
    properties = module._JUDGE_TOOL_CONFIG["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]

    for score_name in ("relevance", "grounding"):
        score_schema = properties[score_name]
        assert score_schema["type"] == "integer"
        assert "minimum" not in score_schema
        assert "maximum" not in score_schema


def test_live_limit_submits_only_the_requested_number_of_prompts():
    module = _load_eval_quality_module()
    items = [
        {"prompt": "first", "expected_type": "scored_prospect", "min_score": 0},
        {"prompt": "second", "expected_type": "scored_prospect", "min_score": 0},
    ]

    with patch.dict("os.environ", {"AGENTCORE_AGENT_ARN": "arn:test", "EVAL_CONCURRENCY": "1"}, clear=False):
        with patch.object(module, "_load_golden_set", return_value=items):
            with patch("boto3.client"):
                with patch.object(
                    module,
                    "_evaluate_prompt",
                    return_value={
                        "prompt": "first",
                        "pattern": "graph",
                        "type": "scored_prospect",
                        "min_score": 0,
                        "max_score": None,
                        "metric": 1.0,
                        "pass": True,
                    },
                ) as evaluate:
                    assert module.run_live(max_prompts=1) is True

    evaluate.assert_called_once()


def test_live_start_selects_a_specific_golden_entry():
    module = _load_eval_quality_module()
    items = [
        {"prompt": "first", "expected_type": "scored_prospect", "min_score": 0},
        {"prompt": "second", "expected_type": "scored_prospect", "min_score": 0},
    ]

    with patch.dict("os.environ", {"AGENTCORE_AGENT_ARN": "arn:test", "EVAL_CONCURRENCY": "1"}, clear=False):
        with patch.object(module, "_load_golden_set", return_value=items):
            with patch("boto3.client"):
                with patch.object(
                    module,
                    "_evaluate_prompt",
                    return_value={
                        "prompt": "second",
                        "pattern": "graph",
                        "type": "scored_prospect",
                        "min_score": 0,
                        "max_score": None,
                        "metric": 1.0,
                        "pass": True,
                    },
                ) as evaluate:
                    assert module.run_live(max_prompts=1, start_at=2) is True

    assert evaluate.call_args.args[0]["prompt"] == "second"
    assert evaluate.call_args.args[1] == 1
