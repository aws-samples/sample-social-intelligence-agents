"""DynamoDB lead storage tools: persist and deduplicate discovered prospects.

Stores leads in the 'social-intel-leads' table with prospect_id (partition key)
and discovered_at (sort key). Agents check for existing leads before processing
to avoid duplicate work and accumulate leads over time.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from botocore.exceptions import ClientError
from strands import tool

from social_intelligence.config import AWS_REGION

logger = logging.getLogger(__name__)

LEADS_TABLE = os.environ.get("LEADS_TABLE_NAME", "social-intel-leads")
MAX_LEADS_PER_RUN = int(os.environ.get("MAX_LEADS_PER_RUN", "3"))

# In-process counter: tracks how many leads this pipeline run has stored.
# Reset each time the AgentCore Runtime creates a new session.
_leads_stored_this_run: int = 0


def reset_lead_counter():
    """Reset the per-run lead counter. Call at the start of each pipeline invocation."""
    global _leads_stored_this_run  # noqa: PLW0603
    _leads_stored_this_run = 0


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
    try:
        resp = table.query(
            IndexName=PRODUCT_NAME_INDEX,
            KeyConditionExpression="product_name_lower = :pn",
            ExpressionAttributeValues={":pn": normalized},
            Limit=1,
        )
        items = resp.get("Items", [])
        if not items:
            return None
        key = items[0]
        # KEYS_ONLY projection: fetch the full item from the base table.
        full = table.get_item(Key={"prospect_id": key["prospect_id"], "discovered_at": key["discovered_at"]})
        return full.get("Item") or key
    except ClientError:
        logger.debug("product_name GSI query failed", exc_info=True)
        return None


@tool
def claim_url(claim_key: str) -> str:
    """Atomically claim a prospect URL or ID in the frontier table to prevent duplicate processing.

    Agents should call this BEFORE fetching or processing a prospect URL or ID.
    If the claim succeeds (claimed=true), the agent is the exclusive processor for
    this item for the next 30 minutes. If the claim fails (claimed=false), another
    agent instance has already claimed it and the agent should skip to the next
    prospect.

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
    table = _get_frontier_table()
    if table is None:
        return json.dumps({"claimed": True, "reason": "frontier disabled"})

    now = datetime.now(timezone.utc)
    expires_at = int((now + timedelta(minutes=30)).timestamp())

    try:
        table.put_item(
            Item={"claim_key": claim_key, "expires_at": expires_at},
            ConditionExpression="attribute_not_exists(claim_key)",
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
    top_trends: str = "",
    data_quality: str = "medium",
    grounding_score: float = 1.0,
) -> str:
    """Store a scored lead in DynamoDB for persistence and deduplication.

    Use this AFTER the analysis agent has scored a prospect. Stores the complete
    lead record including score, enrichment data, and email draft so future
    invocations can skip already-processed prospects.

    Hard limits enforced by this tool (not overridable by the agent):
    - Maximum 3 leads per pipeline run (configurable via MAX_LEADS_PER_RUN env var)
    - Duplicate detection by product_name (case-insensitive) in addition to prospect_id
    - Grounding gate: when GROUNDING_MIN_SCORE env var is set, leads with a
      grounding_score below that threshold are refused (stored=false)
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
        top_trends: Comma-separated top trends for this prospect.
        data_quality: Data quality assessment (high/medium/low).
        grounding_score: Grounding confidence from the analysis step (0.0-1.0).
            When GROUNDING_MIN_SCORE env var is set and this value is below the
            threshold, the lead is refused without writing to DynamoDB.

    Note:
        TOCTOU / dedup limitation: the composite key (prospect_id + discovered_at
        timestamp) means the ``attribute_not_exists(prospect_id)`` conditional write
        only guards exact PK+SK collisions. Cross-run and concurrent same-prospect
        dedup relies on the product_name GSI pre-check (_find_by_product_name) and
        the prospect_id PK query pre-check; neither is atomic against all concurrent
        writers.
    """
    global _leads_stored_this_run  # noqa: PLW0603

    # Clamp grounding_score to [0.0, 1.0] so an agent cannot pass 99.0 to bypass the gate.
    grounding_score = max(0.0, min(1.0, float(grounding_score)))

    # Grounding gate: refuse low-confidence leads when the threshold is configured.
    grounding_min_raw = os.environ.get("GROUNDING_MIN_SCORE", "")
    if grounding_min_raw:
        try:
            grounding_min = float(grounding_min_raw)
            if grounding_score < grounding_min:
                logger.info(
                    "Grounding gate blocked lead: score=%.3f < min=%.3f",
                    grounding_score,
                    grounding_min,
                )
                return json.dumps(
                    {
                        "stored": False,
                        "reason": "failed grounding gate",
                        "grounding_score": grounding_score,
                    }
                )
        except ValueError:
            logger.warning("GROUNDING_MIN_SCORE env var is not a valid float: '%s'", grounding_min_raw)

    # Hard cap: reject if we've already stored enough this run
    if _leads_stored_this_run >= MAX_LEADS_PER_RUN:
        logger.info("Lead cap reached (%d/%d), skipping store", _leads_stored_this_run, MAX_LEADS_PER_RUN)
        return json.dumps(
            {
                "stored": False,
                "reason": f"Lead cap reached ({MAX_LEADS_PER_RUN} per run). Skipping.",
                "prospect_id": prospect_id,
            }
        )

    try:
        table = _get_table()

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

        # Dedup pass 2: query by prospect_id (PK) regardless of product_name.
        # The SK (discovered_at) is a write-time timestamp, so two concurrent writers
        # for the same prospect_id target DIFFERENT items, so the attribute_not_exists
        # condition on the put_item below does NOT prevent both from succeeding.
        # This explicit pre-check closes that gap deterministically.
        try:
            pid_resp = table.query(
                KeyConditionExpression="prospect_id = :pid",
                ExpressionAttributeValues={":pid": prospect_id},
                Limit=1,
            )
            pid_items = pid_resp.get("Items", [])
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
            "reasoning": reasoning,
            "enrichment_summary": enrichment_summary,
            "email_subject": email_subject,
            "email_body": email_body,
            "source_url": source_url,
            "author": author,
            "signal_strength": signal_strength,
            "top_trends": top_trends,
            "data_quality": data_quality,
            "status": lead_status,
            "updated_at": now_iso,
            "expires_at": expires_at,
        }
        # Final race-guard: attribute_not_exists(prospect_id) catches the narrow window
        # where two concurrent writers pass both pre-checks and then race to the same PK+SK.
        # Because SK is a unique write-time timestamp, two writers almost always target
        # DIFFERENT items and BOTH conditions pass, so this guard cannot prevent all
        # duplicates on its own. The primary dedup is the product_name GSI pre-check
        # (_find_by_product_name) and the prospect_id PK query above.
        table.put_item(Item=item, ConditionExpression="attribute_not_exists(prospect_id)")
        _leads_stored_this_run += 1
        logger.info(
            "Stored lead %d/%d (score=%d, status=%s)",
            _leads_stored_this_run,
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
                "leads_stored_this_run": _leads_stored_this_run,
                "leads_remaining": MAX_LEADS_PER_RUN - _leads_stored_this_run,
            }
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            logger.info("Atomic dedup: prospect_id already stored, skipping.")
            return json.dumps(
                {
                    "stored": False,
                    "reason": "Duplicate: prospect_id already exists (atomic check).",
                    "prospect_id": prospect_id,
                }
            )
        logger.error("DynamoDB store error [%s]: %s", e.response["Error"]["Code"], e)
        return json.dumps({"stored": False, "error": "storage_error"})


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

        if prospect_id:
            resp = table.query(
                KeyConditionExpression="prospect_id = :pid",
                ExpressionAttributeValues={":pid": prospect_id},
                ScanIndexForward=False,
                Limit=1,
            )
            items = resp.get("Items", [])
            if items:
                lead = items[0]
                discovered_at = lead.get("discovered_at", "")
                # Calculate age in days for staleness assessment
                age_days = -1
                if discovered_at:
                    try:
                        discovered_dt = datetime.fromisoformat(discovered_at)
                        age_days = (datetime.now(timezone.utc) - discovered_dt).days
                    except (ValueError, TypeError):
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
                    except (ValueError, TypeError):
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

        # Scan for recent leads (for dedup list)
        capped_limit = min(limit, 50)
        leads = []
        scan_kwargs: dict = {"Limit": 100}  # page size, not total limit
        while True:
            resp = table.scan(**scan_kwargs)
            for item in resp.get("Items", []):
                leads.append(
                    {
                        "prospect_id": item["prospect_id"],
                        "product_name": item.get("product_name", ""),
                        "score": item.get("score", 0),
                        "discovered_at": item.get("discovered_at", ""),
                    }
                )
            # Stop if we have enough or no more pages
            if len(leads) >= capped_limit * 3 or "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

        # Sort by discovered_at descending and return top N
        leads.sort(key=lambda x: x.get("discovered_at", ""), reverse=True)
        return json.dumps(
            {
                "leads": leads[:capped_limit],
                "count": len(leads),
                "source": "DynamoDB",
            },
            default=_decimal_default,
        )
    except ClientError as e:
        logger.error("DynamoDB query error [%s]: %s", e.response["Error"]["Code"], e)
        return json.dumps({"error": "storage_error", "leads": []})
