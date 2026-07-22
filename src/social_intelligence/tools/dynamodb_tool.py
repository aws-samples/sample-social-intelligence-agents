"""DynamoDB lead storage tools: persist and deduplicate discovered prospects.

Stores leads in the 'social-intel-leads' table with prospect_id (partition key)
and discovered_at (sort key). Agents check for existing leads before processing
to avoid duplicate work and accumulate leads over time.
"""

import json
import logging
import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from math import isfinite

from botocore.exceptions import ClientError
from pydantic import ValidationError
from strands import tool

from social_intelligence.config import AWS_REGION, EMAIL_SCORE_THRESHOLD, MAX_LEADS_PER_RUN, MIN_INDEPENDENT_SOURCES
from social_intelligence.orchestration.qualification_gate import EmailQualification, assess_email_eligibility
from social_intelligence.schemas.models import PersistedScore, ScorePersistenceRequest
from social_intelligence.tools.grounding_gate import verify_email_claims

logger = logging.getLogger(__name__)

LEADS_TABLE = os.environ.get("LEADS_TABLE_NAME", "social-intel-leads")
RECENT_LEADS_INDEX = "dedup-partition-discovered-at-index"
# Scored-prospect records use a distinct dedup partition so they are attributable to a
# run (via the session_id GSI) for evaluation WITHOUT appearing in the LEAD skip list or
# the email dedup path. They capture every analysis score, including sub-threshold ones
# that never become emailed leads.
SCORE_DEDUP_PARTITION = "SCORE"
LEAD_DEDUP_PARTITION = "LEAD"
RUN_DEDUP_PARTITION = "RUN"
# Run-status rows are visible through the existing session-id GSI because product_name
# is projected there. They let the eval harness distinguish a completed empty run from
# one that is still producing records without changing an existing GSI projection.
RUN_STATUS_PRODUCT_PREFIX = "__run_status__:"
MAX_EVIDENCE_JSON_BYTES = 64 * 1024
_DEDUP_MARKER_SORT_KEY = "__dedup_marker__"
_DEDUP_MARKER_PREFIX = "dedup::"


@dataclass
class _RunState:
    """Mutable request state shared by Strands child tasks for one invocation."""

    leads_stored: int = 0
    lead_slots_reserved: int = 0
    score_persistence_calls: int = 0
    scores_requested: int = 0
    scores_persisted: int = 0
    email_eligible_scores: int = 0
    session_id: str = ""
    isolate: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class RunOutputMetrics:
    """Persisted-output counters used to validate one orchestration run."""

    score_persistence_calls: int
    scores_requested: int
    scores_persisted: int
    email_eligible_scores: int
    leads_stored: int


# AgentCore serves concurrent requests in one process and Strands executes tool calls
# in child asyncio tasks. A ContextVar gives each request its own state object while
# allowing child tasks to share that object's per-run counter and identifiers.
_run_state: ContextVar[_RunState | None] = ContextVar("social_intelligence_run_state", default=None)


def _current_run_state() -> _RunState:
    """Return this invocation's state, initializing it for direct tool calls."""
    state = _run_state.get()
    if state is None:
        state = _RunState()
        _run_state.set(state)
    return state


def reset_lead_counter() -> None:
    """Start a fresh request-scoped counter while retaining explicit run settings."""
    previous = _run_state.get()
    _run_state.set(
        _RunState(
            session_id=previous.session_id if previous else "",
            isolate=previous.isolate if previous else False,
        )
    )


def set_run_session_id(session_id: str) -> None:
    """Set the session id stamped onto leads stored during this run.

    Call at the start of each pipeline invocation. The value is written to every
    lead's ``session_id`` attribute so a caller can attribute leads to a specific
    run even when multiple runs write to the table concurrently.

    Args:
        session_id: Caller-supplied session identifier, or empty to clear it.
    """
    state = _current_run_state()
    with state.lock:
        state.session_id = session_id or ""


# When True, dedup reads (existing-lead checks) are scoped to the invocation session instead of
# the whole table. A fresh run then sees only its own leads, so concurrent runs against a
# shared table do not starve each other of prospects. Off by default: production dedup is
# table-wide. Intended for eval harnesses running many independent prompts in parallel.
def set_run_isolation(isolate: bool) -> None:
    """Enable or disable per-run dedup isolation for this invocation.

    Args:
        isolate: When True, existing-lead checks are scoped to the current run session.
    """
    state = _current_run_state()
    with state.lock:
        state.isolate = bool(isolate)


def run_isolation_enabled() -> bool:
    """Return whether per-run dedup isolation is active for this invocation."""
    state = _current_run_state()
    with state.lock:
        return state.isolate


def get_run_session_id() -> str:
    """Return the run session id stamped onto this invocation's leads (empty if unset)."""
    state = _current_run_state()
    with state.lock:
        return state.session_id


def get_run_output_metrics() -> RunOutputMetrics:
    """Return a consistent snapshot of this invocation's persisted output counters."""
    state = _current_run_state()
    with state.lock:
        return RunOutputMetrics(
            score_persistence_calls=state.score_persistence_calls,
            scores_requested=state.scores_requested,
            scores_persisted=state.scores_persisted,
            email_eligible_scores=state.email_eligible_scores,
            leads_stored=state.leads_stored,
        )


def _decimal_default(obj):
    """JSON serializer for DynamoDB Decimal types."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == int(obj) else float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Lazy singleton table cache: avoid creating a new boto3 resource on every call.
# Keyed by table name so the leads and frontier tables share one code path.
_table_cache: dict[str, object] = {}
_table_lock = threading.Lock()


def _lazy_table(table_name: str):
    """Return a cached boto3 DynamoDB Table resource, creating it on first use.

    Lazy-imports boto3 (cold-start) and double-checked-locks the per-name cache
    so a warm Lambda/Runtime container reuses the same resource across calls.
    """
    table = _table_cache.get(table_name)
    if table is None:
        with _table_lock:
            table = _table_cache.get(table_name)
            if table is None:
                import boto3

                table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(table_name)
                _table_cache[table_name] = table
    return table


def _get_table():
    """Get the leads DynamoDB table resource (lazy singleton)."""
    return _lazy_table(LEADS_TABLE)


def _get_frontier_table():
    """Get the frontier DynamoDB table resource, or None when FRONTIER_TABLE_NAME is unset."""
    frontier_table_name = os.environ.get("FRONTIER_TABLE_NAME", "")
    if not frontier_table_name:
        return None
    return _lazy_table(frontier_table_name)


PRODUCT_NAME_INDEX = "product-name-index"


def _find_by_product_name(table, product_name: str) -> dict | None:
    """Find a lead matching product_name (case-insensitive) via the GSI.

    Queries the `product-name-index` GSI on `product_name_lower` (O(1) lookup).
    The GSI projects keys only, so a hit triggers a follow-up get_item on the
    base table to return the full record. Returns the first match or None.
    """
    normalized = product_name.lower().strip()
    if not normalized:
        return None
    isolate = run_isolation_enabled()
    run_session_id = get_run_session_id()
    try:
        # The GSI is KEYS_ONLY, so query up to a small page of key matches, then fetch
        # each full item from the base table. Under per-run isolation, skip matches that
        # belong to a different run so concurrent eval runs do not dedup each other away.
        resp = table.query(
            IndexName=PRODUCT_NAME_INDEX,
            KeyConditionExpression="product_name_lower = :pn",
            ExpressionAttributeValues={":pn": normalized},
            Limit=10 if isolate else 1,
        )
        for key in resp.get("Items", []):
            full = table.get_item(Key={"prospect_id": key["prospect_id"], "discovered_at": key["discovered_at"]}).get(
                "Item"
            )
            item = full or key
            if isolate and str(item.get("session_id", "")) != run_session_id:
                continue  # a different run's lead — not a duplicate for this run
            return item
        return None
    except ClientError:
        logger.debug("product_name GSI query failed", exc_info=True)
        return None


def _dedup_marker_item(
    *,
    kind: str,
    value: str,
    scope: str,
    expires_at: int,
) -> dict[str, str | int]:
    """Build an opaque, deterministic marker used to reserve one lead identity."""
    digest = sha256(f"{scope}\x1f{kind}\x1f{value}".encode("utf-8")).hexdigest()
    return {
        "prospect_id": f"{_DEDUP_MARKER_PREFIX}{digest}",
        "discovered_at": _DEDUP_MARKER_SORT_KEY,
        "record_type": "dedup_marker",
        "expires_at": expires_at,
    }


def _store_lead_atomically(table, lead_item: dict, *, expires_at: int) -> None:
    """Persist a lead and immutable product/prospect reservations in one transaction.

    DynamoDB conditions apply to a complete composite key, so a conditional put on a
    timestamped lead row cannot guard a second row with the same prospect ID. Stable
    marker keys solve that identity problem without changing the existing lead-table
    schema or its indexes.
    """
    session_id = get_run_session_id()
    scope = f"run:{session_id}" if run_isolation_enabled() and session_id else "global"
    product_marker = _dedup_marker_item(
        kind="product",
        value=str(lead_item["product_name"]).lower().strip(),
        scope=scope,
        expires_at=expires_at,
    )
    prospect_marker = _dedup_marker_item(
        kind="prospect",
        value=str(lead_item["prospect_id"]).strip(),
        scope=scope,
        expires_at=expires_at,
    )
    # ``Table.meta.client`` is the DynamoDB resource's high-level client. It
    # serializes native Python values before sending the request, so passing
    # AttributeValue dictionaries here would serialize them a second time.
    table.meta.client.transact_write_items(
        TransactItems=[
            {
                "Put": {
                    "TableName": table.name,
                    "Item": product_marker,
                    "ConditionExpression": "attribute_not_exists(prospect_id)",
                }
            },
            {
                "Put": {
                    "TableName": table.name,
                    "Item": prospect_marker,
                    "ConditionExpression": "attribute_not_exists(prospect_id)",
                }
            },
            {
                "Put": {
                    "TableName": table.name,
                    "Item": lead_item,
                    "ConditionExpression": "attribute_not_exists(prospect_id) AND attribute_not_exists(discovered_at)",
                }
            },
        ]
    )


def _is_dedup_transaction_conflict(error: ClientError) -> bool:
    """Return whether a transaction was cancelled by an identity reservation."""
    code = error.response.get("Error", {}).get("Code")
    if code != "TransactionCanceledException":
        return code == "ConditionalCheckFailedException"
    reasons = error.response.get("CancellationReasons", [])
    return any(reason.get("Code") == "ConditionalCheckFailed" for reason in reasons if isinstance(reason, dict))


@dataclass(frozen=True)
class _EmailDraftValidation:
    """Validated email-draft fields passed to the lead persistence transaction."""

    canonical_evidence_json: str
    grounding_score: float
    qualification: EmailQualification
    rejection: dict[str, object] | None = None


def _validate_email_draft(
    *,
    score: int,
    email_body: str,
    evidence_json: str,
) -> _EmailDraftValidation:
    """Canonicalize and validate an optional email draft before persistence.

    The persistence boundary owns this check so model-provided score, source, and
    grounding claims cannot bypass the configured qualification policy.
    """
    qualification = assess_email_eligibility(
        score,
        [],
        score_threshold=EMAIL_SCORE_THRESHOLD,
        min_independent_sources=MIN_INDEPENDENT_SOURCES,
    )
    canonical_evidence_json = ""
    grounding_score = 1.0

    if email_body.strip():
        if not str(evidence_json).strip():
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {"stored": False, "reason": "evidence_json is required for an email draft"},
            )
        try:
            evidence = json.loads(
                str(evidence_json),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
            )
            if not isinstance(evidence, (dict, list)):
                return _EmailDraftValidation(
                    canonical_evidence_json,
                    grounding_score,
                    qualification,
                    {"stored": False, "reason": "evidence_json must be a JSON object or array"},
                )
            canonical_evidence_json = json.dumps(evidence, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        except TypeError, ValueError, json.JSONDecodeError:
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {"stored": False, "reason": "evidence_json must be valid JSON"},
            )

        if len(canonical_evidence_json.encode("utf-8")) > MAX_EVIDENCE_JSON_BYTES:
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {"stored": False, "reason": f"evidence_json exceeds {MAX_EVIDENCE_JSON_BYTES} bytes"},
            )

        qualification = assess_email_eligibility(
            score,
            evidence,
            score_threshold=EMAIL_SCORE_THRESHOLD,
            min_independent_sources=MIN_INDEPENDENT_SOURCES,
        )
        if not qualification.score_qualified:
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {
                    "stored": False,
                    "reason": f"email drafts require score >= {EMAIL_SCORE_THRESHOLD}",
                    "score": score,
                },
            )
        if not qualification.source_qualified:
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {
                    "stored": False,
                    "reason": f"email drafts require {MIN_INDEPENDENT_SOURCES} independent sources",
                    "independent_source_count": qualification.independent_source_count,
                },
            )

        # Re-run the exact production grounding check at the persistence boundary. A
        # caller-reported score is untrusted and cannot bypass this check.
        try:
            verification = json.loads(verify_email_claims(email_body, canonical_evidence_json))
            if not isinstance(verification, dict):
                raise ValueError("grounding verification must return a JSON object")
            grounding_score = float(verification["grounding_score"])
        except KeyError, TypeError, ValueError, json.JSONDecodeError:
            logger.warning("Lead persistence could not verify draft grounding", exc_info=True)
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {"stored": False, "reason": "grounding verification failed"},
            )

        if not isfinite(grounding_score) or not 0.0 <= grounding_score <= 1.0:
            logger.warning("Lead persistence received invalid grounding score")
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {"stored": False, "reason": "grounding verification failed"},
            )
        if verification.get("unsupported_claims"):
            logger.info("Grounding gate blocked lead with unsupported claims")
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {
                    "stored": False,
                    "reason": "failed grounding gate",
                    "grounding_score": grounding_score,
                    "unsupported_claims": verification.get("unsupported_claims", []),
                },
            )

    # Grounding gate: refuse low-confidence leads when the threshold is configured.
    grounding_min_raw = os.environ.get("GROUNDING_MIN_SCORE", "")
    if grounding_min_raw:
        try:
            grounding_min = float(grounding_min_raw)
            if not 0.0 <= grounding_min <= 1.0:
                return _EmailDraftValidation(
                    canonical_evidence_json,
                    grounding_score,
                    qualification,
                    {"stored": False, "reason": "GROUNDING_MIN_SCORE must be between 0 and 1"},
                )
            if grounding_score < grounding_min:
                logger.info("Grounding gate blocked lead: score=%.3f < min=%.3f", grounding_score, grounding_min)
                return _EmailDraftValidation(
                    canonical_evidence_json,
                    grounding_score,
                    qualification,
                    {
                        "stored": False,
                        "reason": "failed grounding gate",
                        "grounding_score": grounding_score,
                    },
                )
        except ValueError:
            logger.warning("GROUNDING_MIN_SCORE env var is not a valid float: '%s'", grounding_min_raw)
            return _EmailDraftValidation(
                canonical_evidence_json,
                grounding_score,
                qualification,
                {"stored": False, "reason": "GROUNDING_MIN_SCORE must be a number from 0 to 1"},
            )

    return _EmailDraftValidation(canonical_evidence_json, grounding_score, qualification)


@tool
def claim_url(claim_key: str) -> str:
    """Atomically claim a prospect URL or ID in the frontier table to prevent duplicate processing.

    Agents should call this BEFORE fetching or processing a prospect URL or ID.
    If the claim succeeds (claimed=true), the current runtime owns this item for
    the next 30 minutes. A repeated call from the same run renews its claim. If
    the claim fails (claimed=false), another active run owns it and the agent
    should skip to the next prospect. Expired claims are safely replaced even
    when DynamoDB TTL has not yet deleted the old item.

    When the FRONTIER_TABLE_NAME environment variable is not set, the tool returns
    claimed=true with reason "frontier disabled" so default single-agent behaviour
    is unchanged (no DynamoDB call is made).

    Args:
        claim_key: Unique identifier for the prospect, e.g. the HN story ID or URL.

    Returns:
        JSON string with keys:
            claimed (bool): True if this agent now owns the item.
            reason (str, optional): "already claimed" or "frontier disabled".
    """
    normalized_key = str(claim_key or "").strip()
    if not normalized_key:
        return json.dumps({"claimed": False, "reason": "claim_key is required"})

    table = _get_frontier_table()
    if table is None:
        return json.dumps({"claimed": True, "reason": "frontier disabled"})

    now = datetime.now(timezone.utc)
    expires_at = int((now + timedelta(minutes=30)).timestamp())
    owner_id = get_run_session_id() or "anonymous"

    try:
        table.put_item(
            Item={
                "claim_key": normalized_key,
                "owner_id": owner_id,
                "claimed_at": now.isoformat(),
                "expires_at": expires_at,
            },
            ConditionExpression="attribute_not_exists(claim_key) OR expires_at < :now OR owner_id = :owner",
            ExpressionAttributeValues={":now": int(now.timestamp()), ":owner": owner_id},
        )
        return json.dumps({"claimed": True})
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return json.dumps({"claimed": False, "reason": "already claimed"})
        logger.error("Frontier claim error: %s", e.response["Error"]["Code"])
        raise


@tool
def store_lead(
    prospect_id: str,
    product_name: str,
    score: int = 0,
    confidence: float = 0.0,
    reasoning: str = "",
    enrichment_summary: str = "",
    email_subject: str = "",
    email_body: str = "",
    source_url: str = "",
    author: str = "",
    signal_strength: str = "moderate",
    top_trends: list[str] | None = None,
    data_quality: str = "medium",
    evidence_json: str = "",
) -> str:
    """Store a scored lead in DynamoDB for persistence and deduplication.

    Use this AFTER the analysis agent has scored a prospect. Stores the complete
    lead record including score, enrichment data, and email draft so future
    invocations can skip already-processed prospects.

    Hard limits enforced by this tool (not overridable by the agent):
    - Maximum 3 leads per pipeline run (configurable via MAX_LEADS_PER_RUN env var)
    - Duplicate detection by product_name (case-insensitive) in addition to prospect_id
    - Grounding gate: the exact email is rechecked against ``evidence_json`` before
      writing. Unsupported claims and guardrail intervention are refused.
    - Grounding threshold: when GROUNDING_MIN_SCORE env var is set, its value is
      applied to the server-computed grounding score
    - Email eligibility: drafts require score >= EMAIL_SCORE_THRESHOLD and evidence
      from MIN_INDEPENDENT_SOURCES distinct supported sources
    - Human-in-the-loop: when EMAIL_APPROVAL_REQUIRED is true, leads are stored with
      status 'pending_review' instead of 'new', so a reviewer approves outreach first

    Stored items include an expires_at TTL attribute (now + 365 days, Unix epoch int)
    so the table can enforce automatic expiry via DynamoDB TTL on that attribute.

    Args:
        prospect_id: Unique identifier (e.g. HN story ID).
        product_name: Product or project name.
        score: Relevance score 0-100 from the analysis agent.
        confidence: Confidence in the score (0.0-1.0).
        reasoning: Analysis reasoning summary.
        enrichment_summary: Key enrichment findings.
        email_subject: Generated email subject line.
        email_body: Generated email body text.
        source_url: Primary source URL for the prospect.
        author: Author or maker username.
        signal_strength: Overall signal strength (strong/moderate/weak).
        top_trends: Ordered trend strings for this prospect.
        data_quality: Data quality assessment (high/medium/low).
        evidence_json: JSON source evidence used to verify the draft. Required when
            email_body is non-empty; stored with the lead for audit and live eval.
    Note:
        Persistence atomically writes the lead with two opaque reservation records:
        one for the canonical product name and one for the prospect ID. This prevents
        concurrent writers from creating duplicate timestamped lead rows.
    """
    if not str(prospect_id).strip() or not str(product_name).strip():
        return json.dumps({"stored": False, "reason": "prospect_id and product_name are required"})
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        return json.dumps({"stored": False, "reason": "score must be an integer from 0 to 100"})
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not isfinite(float(confidence)):
        return json.dumps({"stored": False, "reason": "confidence must be a finite number"})
    if not 0.0 <= float(confidence) <= 1.0:
        return json.dumps({"stored": False, "reason": "confidence must be between 0 and 1"})

    draft_validation = _validate_email_draft(
        score=score,
        email_body=email_body,
        evidence_json=evidence_json,
    )
    if draft_validation.rejection:
        return json.dumps(draft_validation.rejection)

    normalized_trends = _normalize_string_list(top_trends)
    state = _current_run_state()
    with state.lock:
        occupied_slots = state.leads_stored + state.lead_slots_reserved
        if occupied_slots >= MAX_LEADS_PER_RUN:
            logger.info(
                "Lead cap reached (%d stored, %d reserved, limit %d), skipping store",
                state.leads_stored,
                state.lead_slots_reserved,
                MAX_LEADS_PER_RUN,
            )
            return json.dumps(
                {
                    "stored": False,
                    "reason": f"Lead cap reached ({MAX_LEADS_PER_RUN} per run). Skipping.",
                    "prospect_id": prospect_id,
                }
            )
        # Reserve before DynamoDB checks and writes so parallel Strands tool calls
        # cannot all observe the same remaining capacity.
        state.lead_slots_reserved += 1
    reservation_active = True

    try:
        table = _get_table()
        isolate = run_isolation_enabled()
        run_session_id = get_run_session_id()

        # Dedup pass 1: check if a lead with the same product_name already exists.
        # (catches cases where the same product gets different prospect_ids across runs)
        if product_name:
            existing = _find_by_product_name(table, product_name)
            if existing:
                logger.info("Duplicate product_name detected (existing vs new prospect). Skipping.")
                return json.dumps(
                    {
                        "stored": False,
                        "reason": f"Duplicate: '{product_name}' already exists as {existing['prospect_id']}",
                        "existing_prospect_id": existing["prospect_id"],
                        "prospect_id": prospect_id,
                    }
                )

        # Dedup pass 2: query by prospect_id (PK) regardless of product_name. This
        # avoids a transaction for known historical duplicates; the atomic reservation
        # below closes the race between this read and the write.
        try:
            pid_resp = table.query(
                KeyConditionExpression="prospect_id = :pid",
                ExpressionAttributeValues={":pid": prospect_id},
                Limit=10 if isolate else 1,
            )
            pid_items = pid_resp.get("Items", [])
            if isolate:
                pid_items = [item for item in pid_items if str(item.get("session_id", "")) == run_session_id]
            if pid_items:
                logger.info("Duplicate prospect_id found by PK query. Skipping.")
                return json.dumps(
                    {
                        "stored": False,
                        "reason": "Duplicate: prospect_id already exists",
                        "prospect_id": prospect_id,
                    }
                )
        except ClientError:
            # Non-fatal: fall through to the conditional put, which is the last-resort guard.
            logger.debug("prospect_id PK pre-check query failed", exc_info=True)

        # Human-in-the-loop gate: when EMAIL_APPROVAL_REQUIRED is enabled, persist leads
        # as 'pending_review' so a human approves outreach before any email is sent.
        # Default ('new') matches the frozen-blog behavior when the flag is unset.
        approval_required = os.environ.get("EMAIL_APPROVAL_REQUIRED", "").strip().lower() in ("true", "1", "yes")
        lead_status = "pending_review" if approval_required else "new"

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires_at = int((now + timedelta(days=365)).timestamp())
        item = {
            "prospect_id": prospect_id,
            "discovered_at": now_iso,
            "product_name": product_name,
            "product_name_lower": product_name.lower().strip(),
            "score": score,
            "confidence": Decimal(str(confidence)),
            "grounding_score": Decimal(str(draft_validation.grounding_score)),
            "independent_source_count": draft_validation.qualification.independent_source_count,
            "email_eligible": draft_validation.qualification.email_eligible,
            "reasoning": reasoning,
            "enrichment_summary": enrichment_summary,
            "email_subject": email_subject,
            "email_body": email_body,
            "evidence_json": draft_validation.canonical_evidence_json,
            "source_url": source_url,
            "author": author,
            "signal_strength": signal_strength,
            "top_trends": normalized_trends,
            "data_quality": data_quality,
            "status": lead_status,
            "dedup_partition": LEAD_DEDUP_PARTITION,
            "updated_at": now_iso,
            "expires_at": expires_at,
            "session_id": run_session_id,
        }
        _store_lead_atomically(table, item, expires_at=expires_at)
        with state.lock:
            state.lead_slots_reserved -= 1
            state.leads_stored += 1
            leads_stored = state.leads_stored
            leads_remaining = max(0, MAX_LEADS_PER_RUN - state.leads_stored - state.lead_slots_reserved)
        reservation_active = False
        logger.info(
            "Stored lead %d/%d (score=%d, status=%s)",
            leads_stored,
            MAX_LEADS_PER_RUN,
            score,
            lead_status,
        )
        return json.dumps(
            {
                "stored": True,
                "prospect_id": prospect_id,
                "score": score,
                "status": lead_status,
                "independent_source_count": draft_validation.qualification.independent_source_count,
                "leads_stored_this_run": leads_stored,
                "leads_remaining": leads_remaining,
            }
        )
    except ClientError as e:
        if _is_dedup_transaction_conflict(e):
            logger.info("Atomic dedup reservation already exists, skipping.")
            return json.dumps(
                {
                    "stored": False,
                    "reason": "Duplicate: product or prospect identity already exists (atomic check).",
                    "prospect_id": prospect_id,
                }
            )
        logger.error("DynamoDB store error [%s]: %s", e.response["Error"]["Code"], e)
        return json.dumps({"stored": False, "error": "storage_error"})
    finally:
        if reservation_active:
            with state.lock:
                state.lead_slots_reserved = max(0, state.lead_slots_reserved - 1)


def _normalize_string_list(values: list[str] | None) -> list[str]:
    """Return non-empty strings while tolerating direct legacy calls with CSV input."""
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    return [str(value).strip() for value in values if str(value).strip()]


def persist_analysis_scores(prospects: list[dict]) -> int:
    """Persist analysis-stage scored prospects for this run so evaluation survives a
    dropped response stream.

    Each prospect is written under the SCORE dedup partition and stamped with the run
    session_id, so an evaluation client reads them back via the session_id GSI even when
    the SSE connection closes before the pipeline finishes. These records are separate
    from emailed leads: they capture every scored prospect, including sub-threshold ones,
    and never enter the LEAD skip list or email dedup path.

    Best-effort and non-fatal: a persistence error is logged and the pipeline continues.

    Args:
        prospects: Scored prospect dicts (score plus optional product_name, confidence,
            icp_fit, data_quality, signal_strength) from the analysis structured output.

    Returns:
        The number of score records written.
    """
    state = _current_run_state()
    with state.lock:
        state.scores_requested += len(prospects)
    if not prospects:
        return 0
    session_id = get_run_session_id()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    expires_at = int((now + timedelta(days=365)).timestamp())
    written = 0
    try:
        table = _get_table()
        for index, raw_prospect in enumerate(prospects):
            try:
                prospect = PersistedScore.model_validate(raw_prospect)
            except ValidationError:
                logger.warning("Skipping score record with invalid score contract", exc_info=True)
                continue
            score = prospect.score
            # A synthetic sort key keeps every score for the run distinct without relying
            # on the (possibly missing) prospect_id, while remaining attributable by session.
            raw_pid = prospect.prospect_id
            prospect_digest = sha256(raw_pid.encode("utf-8")).hexdigest()[:16]
            qualification = assess_email_eligibility(
                score,
                [evidence.model_dump(mode="json") for evidence in prospect.evidence],
                score_threshold=EMAIL_SCORE_THRESHOLD,
                min_independent_sources=MIN_INDEPENDENT_SOURCES,
            )
            item = {
                # Include the list index because the source data can contain duplicate
                # prospect IDs. Without it, same-batch records share both table keys and
                # DynamoDB overwrites one score silently.
                "prospect_id": f"score::{session_id}::{index}::{prospect_digest}",
                "scored_prospect_id": raw_pid,
                "discovered_at": now_iso,
                "product_name": prospect.product_name,
                "score": score,
                "confidence": Decimal(str(prospect.confidence)),
                "icp_fit": prospect.icp_fit,
                "score_breakdown": prospect.score_breakdown.model_dump(mode="json"),
                "data_quality": prospect.data_quality,
                "signal_strength": prospect.signal_strength,
                "independent_source_count": qualification.independent_source_count,
                "email_eligible": qualification.email_eligible,
                "dedup_partition": SCORE_DEDUP_PARTITION,
                "session_id": session_id,
                "updated_at": now_iso,
                "expires_at": expires_at,
            }
            table.put_item(Item=item)
            written += 1
            with state.lock:
                state.scores_persisted += 1
                if qualification.email_eligible:
                    state.email_eligible_scores += 1
    except ClientError as exc:
        logger.warning("Failed to persist analysis scores [%s]: %s", exc.response["Error"]["Code"], exc)
    return written


def _summarize_validation_errors(exc: ValidationError, *, limit: int = 5) -> list[str]:
    """Render Pydantic validation errors as compact, agent-actionable field paths.

    Args:
        exc: The raised validation error for the score-persistence payload.
        limit: Maximum number of error strings to return, to keep the tool result small.

    Returns:
        Up to ``limit`` strings of the form ``"prospects.0.prospect_id: <message>"``.
    """
    summaries: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        summaries.append(f"{location}: {error['msg']}")
        if len(summaries) >= limit:
            break
    return summaries


_PERSIST_SCORED_PROSPECTS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prospects": {
            "type": "array",
            "minItems": 0,
            "maxItems": 5,
            "description": (
                "Every scored prospect, including sub-threshold prospects. Each item must include "
                "prospect_id, score, icp_fit, score_breakdown, and evidence."
            ),
            "items": {
                "type": "object",
                "description": (
                    "A score record with bounded score_breakdown fields and source-backed evidence. "
                    "Pass objects directly, not a JSON-encoded string."
                ),
            },
        },
    },
    "required": ["prospects"],
}


@tool(inputSchema=_PERSIST_SCORED_PROSPECTS_INPUT_SCHEMA)
def persist_scored_prospects(prospects: list[dict[str, object]]) -> str:
    """Persist all swarm analysis scores before handing prospects to email generation.

    Swarm agents cannot use Pydantic structured output because that ends their turn before
    the handoff. The analyst sends every score here before handing off, which records
    qualified and sub-threshold prospects for durable evaluation. The input intentionally
    validates only the score-persistence fields; callers may include a complete
    ``ScoredProspectList`` and its email-oriented fields are ignored.

    Args:
        prospects: Native JSON array of score objects. Each prospect requires
            ``prospect_id``, integer ``score``, ``icp_fit``, a valid ``score_breakdown``,
            and source-backed ``evidence``. Pass objects directly, never a JSON string.

    Returns:
        JSON with ``stored`` and ``persisted``. ``stored`` is false if the input is
        malformed or not every valid prospect could be written.
    """
    try:
        score_request = ScorePersistenceRequest.model_validate({"prospects": prospects})
    except ValidationError as exc:
        # Score arithmetic is now canonicalized, so a failure here means a structurally
        # unusable row (e.g. missing prospect_id/score_breakdown or an out-of-range
        # category). Surface the specific field paths so the analyst can repair the exact
        # prospect instead of retrying the whole batch blindly.
        return json.dumps(
            {
                "stored": False,
                "reason": "prospects must match the score persistence contract",
                "errors": _summarize_validation_errors(exc),
                "persisted": 0,
            }
        )
    except TypeError, ValueError:
        return json.dumps(
            {
                "stored": False,
                "reason": "prospects must be a native array of score objects",
                "persisted": 0,
            }
        )

    prospects = [prospect.model_dump(mode="json") for prospect in score_request.prospects]
    state = _current_run_state()
    with state.lock:
        state.score_persistence_calls += 1
    persisted = persist_analysis_scores(prospects)
    return json.dumps({"stored": persisted == len(prospects), "persisted": persisted})


def persist_run_status(succeeded: bool, *, execution_path: str = "") -> None:
    """Persist a terminal background-run marker for completion-aware evaluation polling.

    Args:
        succeeded: Whether the full pipeline completed without an unhandled error.
        execution_path: The orchestrator path that produced the terminal result.
            ``graph_recovery`` indicates a Swarm contract failure recovered by Graph.
    """
    session_id = get_run_session_id()
    if not session_id:
        return

    now = datetime.now(timezone.utc)
    try:
        _get_table().put_item(
            Item={
                "prospect_id": f"run::{session_id}",
                "discovered_at": now.isoformat(),
                "product_name": f"{RUN_STATUS_PRODUCT_PREFIX}{'succeeded' if succeeded else 'failed'}",
                "dedup_partition": RUN_DEDUP_PARTITION,
                "session_id": session_id,
                "execution_path": execution_path,
                "updated_at": now.isoformat(),
                "expires_at": int((now + timedelta(days=7)).timestamp()),
            }
        )
    except ClientError as exc:
        logger.warning("Failed to persist run completion marker [%s]: %s", exc.response["Error"]["Code"], exc)


@tool
def check_existing_leads(prospect_id: str = "", product_name: str = "", limit: int = 20) -> str:
    """Check DynamoDB for existing leads to avoid duplicate discovery.

    Use this BEFORE running full discovery to see what prospects have already
    been processed. Checks by prospect_id OR product_name (case-insensitive).
    If both are provided, checks prospect_id first, then falls back to product_name.

    Args:
        prospect_id: Optional specific prospect ID to check.
        product_name: Optional product name to check (case-insensitive fuzzy match).
        limit: Maximum number of recent leads to return (for scan mode).
    """
    try:
        table = _get_table()
        isolate = run_isolation_enabled()
        run_session_id = get_run_session_id()

        if prospect_id:
            resp = table.query(
                KeyConditionExpression="prospect_id = :pid",
                ExpressionAttributeValues={":pid": prospect_id},
                ScanIndexForward=False,
                Limit=10 if isolate else 1,
            )
            items = resp.get("Items", [])
            if isolate:
                items = [item for item in items if str(item.get("session_id", "")) == run_session_id]
            if items:
                lead = items[0]
                discovered_at = lead.get("discovered_at", "")
                # Calculate age in days for staleness assessment
                age_days = -1
                if discovered_at:
                    try:
                        discovered_dt = datetime.fromisoformat(discovered_at)
                        age_days = (datetime.now(timezone.utc) - discovered_dt).days
                    except ValueError, TypeError:
                        pass
                return json.dumps(
                    {
                        "exists": True,
                        "prospect_id": lead["prospect_id"],
                        "product_name": lead.get("product_name", ""),
                        "score": lead.get("score", 0),
                        "discovered_at": discovered_at,
                        "status": lead.get("status", ""),
                        "age_days": age_days,
                        "stale": age_days > 7,
                    },
                    default=_decimal_default,
                )
            # prospect_id not found: fall through to product_name check if available
            if not product_name:
                return json.dumps({"exists": False, "prospect_id": prospect_id})

        # Check by product_name (case-insensitive)
        if product_name:
            existing = _find_by_product_name(table, product_name)
            if existing:
                discovered_at = existing.get("discovered_at", "")
                age_days = -1
                if discovered_at:
                    try:
                        discovered_dt = datetime.fromisoformat(discovered_at)
                        age_days = (datetime.now(timezone.utc) - discovered_dt).days
                    except ValueError, TypeError:
                        pass
                return json.dumps(
                    {
                        "exists": True,
                        "matched_by": "product_name",
                        "prospect_id": existing["prospect_id"],
                        "product_name": existing.get("product_name", ""),
                        "score": existing.get("score", 0),
                        "discovered_at": discovered_at,
                        "status": existing.get("status", ""),
                        "age_days": age_days,
                        "stale": age_days > 7,
                    },
                    default=_decimal_default,
                )
            return json.dumps({"exists": False, "product_name": product_name})

        # Query the dedicated GSI for the newest leads without a table scan.
        try:
            capped_limit = max(1, min(int(limit), 50))
        except TypeError, ValueError:
            capped_limit = 20
        resp = table.query(
            IndexName=RECENT_LEADS_INDEX,
            KeyConditionExpression="dedup_partition = :partition",
            ExpressionAttributeValues={":partition": LEAD_DEDUP_PARTITION},
            ScanIndexForward=False,
            Limit=capped_limit,
        )
        leads = resp.get("Items", [])
        if isolate:
            leads = [item for item in leads if str(item.get("session_id", "")) == run_session_id]

        recent_leads = [
            {
                "prospect_id": item["prospect_id"],
                "product_name": item.get("product_name", ""),
                "score": item.get("score", 0),
                "discovered_at": item.get("discovered_at", ""),
            }
            for item in leads
        ]
        return json.dumps(
            {
                "leads": recent_leads,
                "count": len(recent_leads),
                "source": "DynamoDB",
            },
            default=_decimal_default,
        )
    except ClientError as e:
        logger.error("DynamoDB query error [%s]: %s", e.response["Error"]["Code"], e)
        return json.dumps({"error": "storage_error", "leads": []})
