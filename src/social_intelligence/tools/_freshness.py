"""Temporal freshness decay helper for trend weighting.

Returns a float weight based on how recent a piece of content is relative
to the current UTC time. Used by tool handlers to populate freshness_weight
on returned items so downstream agents receive a pre-computed number instead
of relying on LLM arithmetic.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def freshness_weight(epoch_or_iso: int | float | str | None) -> float:
    """Compute a temporal decay weight from a Unix epoch or ISO 8601 timestamp.

    Thresholds:
        < 24 h  -> 1.5
        < 72 h  -> 1.2
        < 168 h -> 1.0
        >= 168 h -> 0.5

    Args:
        epoch_or_iso: Unix epoch (int or float) or ISO 8601 string. None or
            unparseable input returns the neutral weight 1.0.

    Returns:
        Decay weight as a float.
    """
    if epoch_or_iso is None:
        return 1.0

    try:
        if isinstance(epoch_or_iso, (int, float)):
            ts = datetime.fromtimestamp(float(epoch_or_iso), tz=timezone.utc)
        else:
            raw = str(epoch_or_iso).strip()
            if not raw:
                return 1.0
            ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        logger.debug("freshness_weight: could not parse %r, returning 1.0", epoch_or_iso)
        return 1.0

    now = datetime.now(timezone.utc)
    age_hours = (now - ts).total_seconds() / 3600.0

    if age_hours < 24:
        return 1.5
    if age_hours < 72:
        return 1.2
    if age_hours < 168:
        return 1.0
    return 0.5
