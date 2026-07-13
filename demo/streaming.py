"""Amazon Bedrock AgentCore streaming client and SSE event parser."""

import json
import time
import uuid
from collections.abc import Generator

import bedrock_agentcore  # noqa: F401 — registers the service model with botocore
from botocore.config import Config as BotoConfig
from config import AGENT_ARN, AWS_REGION
from data import get_boto_session

# Update the status banner every N events so the UI never looks frozen.
_STATUS_UPDATE_INTERVAL = 3

# Maximum wall-clock seconds for the entire streaming response. After this,
# the loop breaks and returns whatever partial results have been collected.
# AgentCore's own timeout is 60 min; this caps the Streamlit wait at 25 min.
_STREAM_WALL_CLOCK_TIMEOUT = 1500.0  # 25 minutes


def invoke_agentcore_streaming(
    prompt: str,
    pattern: str = "graph",
    events: list[dict] | None = None,
    session_id: str = "",
    live_container=None,
    status_placeholder=None,
) -> str:
    """Invoke the deployed Amazon Bedrock AgentCore agent with response streaming.

    Args:
        prompt: User prompt to send to the agent.
        pattern: Orchestration pattern ('graph' or 'swarm').
        events: Mutable list to append parsed events to.
        session_id: Optional session ID for memory integration.
        live_container: Optional Streamlit container for live event rendering.
        status_placeholder: Optional ``st.empty()`` placeholder updated after
            every few events so users see live progress on long-running pipelines.

    Returns:
        Final result text from the pipeline.
    """
    if events is None:
        events = []

    if not AGENT_ARN:
        _append_event(
            events,
            {"type": "error", "message": "AGENTCORE_AGENT_ARN env var not set.", "ts": time.time()},
            live_container,
            status_placeholder,
        )
        return ""

    session = get_boto_session()
    client = session.client(
        "bedrock-agentcore",
        region_name=AWS_REGION,
        config=BotoConfig(read_timeout=900, retries={"max_attempts": 0}),
    )

    payload = json.dumps(
        {"prompt": prompt, "pattern": pattern, **({"session_id": session_id} if session_id else {})}
    ).encode()
    sid = session_id or str(uuid.uuid4())

    _append_event(
        events,
        {"type": "info", "message": f"Invoking Amazon Bedrock AgentCore ({pattern})...", "ts": time.time()},
        live_container,
        status_placeholder,
    )

    response = client.invoke_agent_runtime(
        agentRuntimeArn=AGENT_ARN,
        runtimeSessionId=sid,
        payload=payload,
        contentType="application/json",
        accept="text/event-stream, application/json",
        qualifier="DEFAULT",
    )

    final_parts: list[str] = []
    active_tools: dict[str, str] = {}
    started_nodes: set[str] = set()
    ended_nodes: set[str] = set()

    body = response.get("response")
    if body is None:
        _append_event(
            events,
            {"type": "error", "message": "No response body", "ts": time.time()},
            live_container,
            status_placeholder,
        )
        return ""

    content_type = response.get("contentType", "")
    is_sse = "text/event-stream" in content_type

    try:
        if is_sse and hasattr(body, "iter_lines"):
            line_iter: Generator[str, None, None] = (
                ln.decode("utf-8", errors="replace") if isinstance(ln, bytes) else str(ln)
                for ln in body.iter_lines(chunk_size=1024)
            )
        else:
            # Fallback: chunk-based iteration with manual line splitting
            line_iter = _chunk_to_lines(body)

        stream_deadline = time.monotonic() + _STREAM_WALL_CLOCK_TIMEOUT

        for line in line_iter:
            # Wall-clock timeout — break out of a stalled or very long stream
            if time.monotonic() > stream_deadline:
                _append_event(
                    events,
                    {
                        "type": "info",
                        "message": f"Stream timeout ({int(_STREAM_WALL_CLOCK_TIMEOUT)}s) — partial results shown above",
                        "ts": time.time(),
                    },
                    live_container,
                    status_placeholder,
                )
                break

            line = line.strip()
            if not line:
                continue

            raw = line[6:] if line.startswith("data: ") else line

            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            evt_type = parsed.get("type", "")

            if evt_type == "multiagent_node_start":
                nid = parsed.get("node_id", "")
                if nid not in started_nodes:
                    started_nodes.add(nid)
                    _append_event(
                        events,
                        {"type": "node_start", "node_id": nid, "ts": time.time()},
                        live_container,
                        status_placeholder,
                    )

            elif evt_type in ("multiagent_node_stop", "multiagent_node_complete"):
                nid = parsed.get("node_id", "")
                if nid not in ended_nodes:
                    ended_nodes.add(nid)
                    _append_event(
                        events,
                        {"type": "node_end", "node_id": nid, "ts": time.time()},
                        live_container,
                        status_placeholder,
                    )

            elif evt_type == "multiagent_handoff":
                for nid in parsed.get("from_node_ids", []):
                    if nid not in ended_nodes:
                        ended_nodes.add(nid)
                        _append_event(
                            events,
                            {"type": "node_end", "node_id": nid, "ts": time.time()},
                            live_container,
                            status_placeholder,
                        )
                for nid in parsed.get("to_node_ids", []):
                    if nid not in started_nodes:
                        started_nodes.add(nid)
                        _append_event(
                            events,
                            {"type": "node_start", "node_id": nid, "ts": time.time()},
                            live_container,
                            status_placeholder,
                        )

            elif evt_type == "multiagent_node_stream":
                _parse_node_stream(parsed, events, active_tools, live_container, status_placeholder)

            elif evt_type == "multiagent_result":
                _append_event(
                    events,
                    {"type": "pipeline_end", "ts": time.time()},
                    live_container,
                    status_placeholder,
                )
                result = parsed.get("result")
                if result:
                    final_parts.append(str(result)[:5000])
    except Exception as stream_err:
        err_name = type(stream_err).__name__
        if "IncompleteRead" in err_name or "ResponseStreamingError" in str(stream_err):
            _append_event(
                events,
                {"type": "info", "message": "Stream ended -- results shown above", "ts": time.time()},
                live_container,
                status_placeholder,
            )
        else:
            msg = f"Stream error: {err_name} -- partial results shown above"
            _append_event(
                events,
                {"type": "error", "message": msg, "ts": time.time()},
                live_container,
                status_placeholder,
            )

    # Mark any nodes that started but never received a stop/complete event
    for nid in started_nodes - ended_nodes:
        _append_event(
            events,
            {"type": "node_end", "node_id": nid, "ts": time.time()},
            live_container,
            status_placeholder,
        )

    return "".join(final_parts)


def _chunk_to_lines(body) -> Generator[str, None, None]:
    """Convert a chunk-based stream into a line iterator (fallback for non-SSE responses)."""
    buffer = ""
    for chunk in body:
        buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer.strip():
        yield buffer


def _append_event(
    events: list[dict],
    evt: dict,
    live_container=None,
    status_placeholder=None,
) -> None:
    """Append event and optionally render it live in a Streamlit container.

    Also updates ``status_placeholder`` every :data:`_STATUS_UPDATE_INTERVAL`
    events with a running count and elapsed time so the UI never appears frozen
    during long-running pipelines.
    """
    events.append(evt)
    if live_container is not None:
        from components import render_event

        with live_container:
            render_event(evt)

    if status_placeholder is not None and len(events) % _STATUS_UPDATE_INTERVAL == 0:
        start_ts = events[0].get("ts", time.time())
        elapsed_s = time.time() - start_ts
        mins, secs = divmod(int(elapsed_s), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        tool_count = sum(1 for e in events if e.get("type") == "tool_end")
        node_count = sum(1 for e in events if e.get("type") == "node_end")
        status_placeholder.markdown(
            f'<div style="background:#1a202c;border-radius:8px;padding:14px 18px;margin:8px 0;'
            f'border:1px solid #2d3748;text-align:center;">'
            f'<span style="color:#63b3ed;font-size:13px;font-weight:500;">'
            f"🔄 Pipeline running · {node_count} phases · {tool_count} tool calls · {elapsed_str}"
            f"</span></div>",
            unsafe_allow_html=True,
        )


def _parse_node_stream(
    parsed: dict,
    events: list[dict],
    active_tools: dict[str, str],
    live_container=None,
    status_placeholder=None,
) -> None:
    """Extract tool calls, tool results, and structured output from a node stream event."""
    node_id = parsed.get("node_id", "")
    event_data = parsed.get("event", {})
    if not isinstance(event_data, dict):
        return

    msg = event_data.get("message")
    if isinstance(msg, dict):
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            tu = block.get("toolUse", {})
            if tu and tu.get("name"):
                tool_use_id = tu.get("toolUseId", "")
                if tool_use_id:
                    active_tools[tool_use_id] = tu["name"]
                tool_input = tu.get("input", {})
                summary = (
                    {
                        k: (str(v)[:120] + "..." if len(str(v)) > 120 else str(v))
                        for k, v in (tool_input if isinstance(tool_input, dict) else {}).items()
                        if k in list(tool_input)[:3]
                    }
                    if isinstance(tool_input, dict)
                    else {}
                )
                tool_evt = {
                    "type": "tool_start",
                    "tool_name": tu["name"],
                    "input_summary": summary,
                    "node_id": node_id,
                    "ts": time.time(),
                }
                _append_event(events, tool_evt, live_container, status_placeholder)
            tr = block.get("toolResult", {})
            if tr and tr.get("toolUseId"):
                result_text = ""
                for rc in tr.get("content", []):
                    if isinstance(rc, dict) and "text" in rc:
                        result_text = rc["text"][:8000]
                        break
                _append_event(
                    events,
                    {
                        "type": "tool_end",
                        "tool_name": active_tools.pop(tr["toolUseId"], "unknown"),
                        "status": tr.get("status", "success"),
                        "result_preview": result_text,
                        "node_id": node_id,
                        "ts": time.time(),
                    },
                    live_container,
                    status_placeholder,
                )

    so = event_data.get("structured_output")
    if so and isinstance(so, dict):
        _append_event(
            events,
            {"type": "structured_output", "node_id": node_id, "keys": list(so.keys()), "ts": time.time()},
            live_container,
            status_placeholder,
        )

    inner = event_data.get("event", {})
    if isinstance(inner, dict):
        cbs = inner.get("contentBlockStart", {})
        if isinstance(cbs, dict):
            start = cbs.get("start", {})
            if isinstance(start, dict) and "toolUse" in start:
                tid = start["toolUse"].get("toolUseId", "")
                tname = start["toolUse"].get("name", "")
                if tid and tname:
                    active_tools[tid] = tname
                    # Emit tool_start for streaming tool calls that arrive via contentBlockStart
                    if not any(
                        e.get("type") == "tool_start"
                        and e.get("tool_name") == tname
                        and e.get("node_id") == node_id
                        and abs(e.get("ts", 0) - time.time()) < 2
                        for e in events[-10:]
                    ):
                        _append_event(
                            events,
                            {
                                "type": "tool_start",
                                "tool_name": tname,
                                "input_summary": {},
                                "node_id": node_id,
                                "ts": time.time(),
                            },
                            live_container,
                            status_placeholder,
                        )
