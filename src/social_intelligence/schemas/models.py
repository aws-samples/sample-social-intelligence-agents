"""Pydantic contracts shared by the Strands agents.

Every downstream stage receives the prospect identifier plus its supporting evidence.
Keeping that evidence in the contracts lets the analysis and email stages make
grounded decisions without relying on hidden tool-call history.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceItem(BaseModel):
    """A source-backed fact that may be cited in analysis or outreach."""

    source: str = Field(
        description="Data source: hackernews, youtube, devto, "
        "github, lobsters, wikipedia, reddit, stackoverflow, producthunt",
    )
    url: str = Field(default="", description="Source URL")
    fact: str = Field(default="", description="Concrete, source-backed fact suitable for citation")
    observed_at: str = Field(default="", description="ISO timestamp or date when the fact was observed")
    metric_name: str = Field(default="", description="Metric represented by metric_value, such as stars or votes")
    metric_value: str = Field(default="", description="Metric value exactly as reported by the source")


class TrendItem(EvidenceItem):
    """A trend signal collected during prospect discovery."""

    topic: str = Field(default="", description="Trend topic or title")
    engagement: int = Field(default=0, description="Engagement metric (views, likes, score)")
    timestamp: str = Field(default="", description="ISO timestamp for temporal decay weighting")
    intent_signals: list[str] = Field(
        default_factory=list,
        description="Detected intent signals: recommendation_seeking, "
        "competitor_frustration, product_launch, purchase_intent",
    )
    freshness_weight: float = Field(
        default=1.0,
        description="Temporal decay weight from source timestamp: 1.5 <24h, 1.2 <72h, 1.0 <168h, 0.5 older",
    )


class ProspectProfile(BaseModel):
    """A prospect and the source signals that justify researching it further."""

    prospect_id: str = Field(description="Unique identifier (e.g. HN story ID)")
    product_name: str = Field(description="Product or project name")
    tagline: str = Field(default="")
    author: str = Field(default="", description="Author or maker username")
    source_url: str = Field(default="", description="Canonical URL for the discovery source")
    category: str = Field(default="")
    community_score: int = Field(default=0, description="Community score, votes, or upvotes")
    signal_strength: Literal["strong", "moderate", "weak"] = Field(
        default="moderate",
        description="strong, moderate, or weak",
    )
    trend_signals: list[TrendItem] = Field(
        default_factory=list,
        max_length=3,
        description="Source-backed trend and buying-intent signals for this prospect",
    )


class TrendData(BaseModel):
    """Output of the Trend Research Agent: structured output model."""

    prospects: list[ProspectProfile] = Field(
        default_factory=list,
        max_length=5,
        description="Up to five discovered prospects",
    )


class ProspectEnrichment(BaseModel):
    """Prospect-specific research produced by the Search Specialist Agent."""

    prospect_id: str = Field(description="Identifier from TrendData")
    product_name: str = Field(description="Product or project name from TrendData")
    background: str = Field(max_length=240, default="", description="Concise factual context")
    recent_news: str = Field(
        max_length=240,
        default="",
        description="Recent source-backed development, or empty when unavailable",
    )
    competitors: list[str] = Field(default_factory=list, max_length=3, description="Up to three competitors")
    oss_summary: str = Field(max_length=240, default="", description="Concise open-source activity summary")
    talking_points: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Up to two specific, factual outreach talking points",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=3,
        description="Facts and metrics from the sources used for this enrichment",
    )


class EnrichmentData(BaseModel):
    """Output of the Search Specialist Agent: structured output model."""

    prospects: list[ProspectEnrichment] = Field(
        default_factory=list,
        max_length=5,
        description="Enrichment keyed to each prospect identifier from TrendData",
    )


class ScoreBreakdown(BaseModel):
    """Bounded category contributions used to calculate one prospect score."""

    topical_alignment: int = Field(ge=0, le=25, description="Topical alignment contribution, 0-25")
    timing_relevance: int = Field(ge=0, le=20, description="Timing relevance contribution, 0-20")
    engagement_potential: int = Field(ge=0, le=20, description="Engagement potential contribution, 0-20")
    intent_signal_strength: int = Field(ge=0, le=20, description="Intent-signal contribution, 0-20")
    data_quality: int = Field(ge=0, le=15, description="Data-quality contribution, 0-15")
    icp_adjustment: Literal[-10, 0, 10] = Field(description="ICP-fit adjustment applied after category totals")

    def total_score(self) -> int:
        """Return the capped score specified by the analysis-agent contract."""
        base_score = (
            self.topical_alignment
            + self.timing_relevance
            + self.engagement_potential
            + self.intent_signal_strength
            + self.data_quality
        )
        return min(100, max(0, base_score + self.icp_adjustment))


# The ICP-fit label deterministically fixes the adjustment applied after category totals.
_ICP_ADJUSTMENTS: dict[str, Literal[-10, 0, 10]] = {"strong": 10, "medium": 0, "weak": -10}


def _validate_score_calculation(score: int, icp_fit: str, score_breakdown: ScoreBreakdown) -> None:
    """Reject a score whose components or ICP adjustment do not match its total."""
    expected_adjustment = _ICP_ADJUSTMENTS[icp_fit]
    if score_breakdown.icp_adjustment != expected_adjustment:
        raise ValueError(f"icp_adjustment must be {expected_adjustment} for icp_fit={icp_fit!r}")
    expected_score = score_breakdown.total_score()
    if score != expected_score:
        raise ValueError(f"score must equal the capped score_breakdown total ({expected_score})")


def _canonical_score(icp_fit: str, score_breakdown: ScoreBreakdown) -> tuple[int, ScoreBreakdown]:
    """Derive the authoritative score and ICP adjustment from bounded contributions.

    The ``score_breakdown`` categories are the source of truth: each is independently
    range-validated, so the auditable 0-100 total is a pure function of them plus the
    ICP-fit label. Rather than rejecting a caller whose top-line ``score`` or
    ``icp_adjustment`` drifts from that total — an easy arithmetic slip for an LLM that
    would otherwise drop the entire batch — we recompute both from the breakdown. The
    persisted score therefore always recomputes from its components, preserving the
    auditable-scoring contract without brittle rejection.

    Args:
        icp_fit: The ICP-fit label that fixes the post-total adjustment.
        score_breakdown: Bounded per-category contributions.

    Returns:
        The canonical (score, score_breakdown) with ``icp_adjustment`` normalized to the
        value implied by ``icp_fit``.
    """
    normalized = score_breakdown.model_copy(update={"icp_adjustment": _ICP_ADJUSTMENTS[icp_fit]})
    return normalized.total_score(), normalized


class ScoredProspect(BaseModel):
    """A prospect scored for outreach, retaining the facts needed for grounding."""

    prospect_id: str = Field(description="Prospect identifier from TrendData")
    product_name: str = Field(default="", description="Product or project name")
    source_url: str = Field(default="", description="Canonical discovery source URL")
    author: str = Field(default="", description="Author or maker username from discovery")
    score: int = Field(ge=0, le=100, description="Relevance score 0-100")
    score_breakdown: ScoreBreakdown = Field(description="Bounded components that deterministically calculate score")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the score")
    reasoning: str = Field(max_length=480, description="Concise, source-backed score rationale")
    top_trends: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three trends most relevant for email personalization",
    )
    intent_signals: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three detected buying-intent signals",
    )
    icp_fit: Literal["strong", "medium", "weak"] = Field(
        default="medium",
        description="ICP fit: strong, medium, or weak",
    )
    data_quality: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Assessment: high, medium, or low",
    )
    signal_strength: Literal["strong", "moderate", "weak"] = Field(
        default="moderate",
        description="Combined evidence strength: strong, moderate, or weak",
    )
    enrichment_summary: str = Field(
        default="",
        max_length=480,
        description="Concise factual enrichment summary for persistence",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=4,
        description="Up to four source-backed facts allowed to support email claims",
    )

    @model_validator(mode="after")
    def validate_score_calculation(self) -> "ScoredProspect":
        """Ensure the stored total matches its bounded category contributions."""
        _validate_score_calculation(self.score, self.icp_fit, self.score_breakdown)
        return self


class ScoredProspectList(BaseModel):
    """Output of the Analysis Agent: list of scored prospects."""

    prospects: list[ScoredProspect] = Field(
        description="All scored prospects. Include every prospect from research, not just the top one."
    )


class PersistedScore(BaseModel):
    """Minimal, durable subset of a scored prospect needed by the evaluation store.

    Swarm persistence retains the evidence needed for deterministic qualification but
    does not depend on email-only content such as outreach reasoning. Extra
    full-contract fields are deliberately ignored, so the analyst may pass either this
    compact representation or a complete ``ScoredProspect`` object.
    """

    model_config = ConfigDict(extra="ignore")

    prospect_id: str = Field(min_length=1, max_length=256, description="Stable discovery identifier")
    score: int = Field(ge=0, le=100, description="Relevance score from 0 to 100")
    product_name: str = Field(default="", max_length=512, description="Product or project name")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    icp_fit: Literal["strong", "medium", "weak"] = Field(default="medium")
    score_breakdown: ScoreBreakdown = Field(
        description="Bounded contributions that independently validate the persisted score."
    )
    data_quality: str = Field(default="", max_length=32)
    signal_strength: str = Field(default="", max_length=32)
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=4,
        description="Source-backed facts retained to compute deterministic email eligibility.",
    )

    @model_validator(mode="after")
    def canonicalize_score(self) -> "PersistedScore":
        """Recompute the persisted score from its bounded components.

        Unlike the Graph structured-output path — which can transparently retry on a
        mismatch — the Swarm analyst persists through a plain tool call with no
        structured-output retry. A one-point arithmetic slip or an ``icp_adjustment``
        that disagrees with ``icp_fit`` would otherwise reject the whole batch and force
        blind self-correction until the tool budget is exhausted. The breakdown
        categories are individually range-validated and are the source of truth, so we
        derive the authoritative score and ICP adjustment from them. The stored score
        still recomputes exactly from its components, keeping scoring auditable.
        """
        self.score, self.score_breakdown = _canonical_score(self.icp_fit, self.score_breakdown)
        return self


class ScorePersistenceRequest(BaseModel):
    """Validated payload for durable swarm-analysis score persistence."""

    model_config = ConfigDict(extra="ignore")

    prospects: list[PersistedScore] = Field(
        max_length=5,
        description=(
            "Every prospect scored by the swarm analyst, including low-score prospects. "
            "Empty is valid when no prospects were found."
        ),
    )


class EmailDraft(BaseModel):
    """A single email draft for one prospect."""

    prospect_id: str = Field(description="Prospect identifier")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text, under 150 words")
    personalization_tokens: list[str] = Field(
        default_factory=list,
        description="Specific data points referenced in the email (e.g. 'HN score: 342')",
    )
    brand_compliant: bool = Field(default=True, description="Whether email passes brand guidelines check")


class EmailDraftList(BaseModel):
    """Output of the Email Generation Agent: list of email drafts."""

    drafts: list[EmailDraft] = Field(
        description="One email draft per prospect that meets the score and independent-source requirements."
    )
