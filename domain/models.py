from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, NamedTuple

from pydantic import BaseModel, Field, model_validator

from shared.constants import SourceType, VisualOrientation


def generate_item_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


class CollectedItem(BaseModel):
    item_id: str = ""
    source_type: SourceType
    title: str
    url: str
    text: str = ""
    author: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_item_id(self) -> CollectedItem:
        if not self.item_id and self.url:
            self.item_id = hashlib.sha256(self.url.encode()).hexdigest()[:16]
        return self

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CollectedItem):
            return False
        return self.url == other.url


class RankedItem(BaseModel):
    item: CollectedItem
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class DigestItem(BaseModel):
    """One story in the digest, as plain prose. The Threads renderer lays it out; the LLM never
    writes markup itself."""

    title: str  # Korean display title
    url: str
    source_tag: str = ""  # e.g. "r/LocalLLaMA", "@karpathy", "arxiv.org" — set by code
    metrics: str = ""  # e.g. "조회 12,000" — set by code
    body: str  # 2-4 sentences: what it is and why it matters
    implication: str = ""  # one sharp closing line (Gruber voice)


class DigestContent(BaseModel):
    """Structured digest the digest LLM returns. `lead` is the columnist take connecting
    the headline to its trend arc; the headline is items[0], the story the lead is about and the
    one the daily visual depicts."""

    lead: str
    items: list[DigestItem] = Field(default_factory=list)


class RankingHealth(BaseModel):
    """How complete the ranking pass was: how many batches failed every retry, and how many of the
    day's candidates ended up scored.

    A partly-ranked pool is not a failed run — the digest still publishes off what did rank — but it
    must not read as a clean success either: a throttled batch silently deletes ~40 candidates from
    the day, which is invisible in the digest itself."""

    batches_total: int = 0
    batches_failed: int = 0
    items_total: int = 0
    items_scored: int = 0
    items_lost: int = 0

    @property
    def coverage(self) -> float:
        """Share of the day's candidates that ended up scored; 1.0 when there was nothing to rank."""
        return self.items_scored / self.items_total if self.items_total else 1.0

    @property
    def degraded(self) -> bool:
        """True when candidates were LOST — a batch that failed every retry.

        A `coverage < min_coverage_ratio` arm sat beside this. It was added for the batch whose
        response never parsed twice over, leaving every counter at zero while the alerting stayed
        silent — but the same commit made that case RAISE, so the arm was dead by its own docstring.
        Measured over 30 production runs, per-day coverage is bimodal (0.93-1.0 or 0.0), so the
        tuned ratio never separated anything the batches_failed count did not already catch."""
        return self.batches_failed > 0

    def summary(self) -> str:
        return (
            f"{self.batches_failed}/{self.batches_total} ranking batches failed permanently; "
            f"{self.items_lost} of {self.items_total} candidates never reached the digest "
            f"({self.items_scored} scored, coverage {self.coverage:.0%})"
        )


class DigestResult(BaseModel):
    digest_text: str
    # Defaults to [] so the memory snapshot can persist the digest result WITHOUT re-embedding
    # the ranked list (it's stored once at the snapshot's top level); load rebuilds it there.
    ranked_items: list[RankedItem] = Field(default_factory=list)
    content: DigestContent | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Set by the pipeline after ranking, so the delivery/alerting layer can report a digest built on
    # an incomplete candidate pool. None means "not recorded" (older snapshots, direct construction).
    ranking_health: RankingHealth | None = None


class VisualBrief(BaseModel):
    # No max_length: the title only feeds a log line, so a cap could only cost the day's visual.
    title: str = Field(min_length=1)
    # The Images API limit for the GPT image models (4000 is dall-e-3's): a four-panel brief overran it.
    prompt: str = Field(min_length=1, max_length=32000)
    # Required, no default: a default let a brief that left the field out silently become portrait,
    # and the visual pipeline then recorded that fallback as if it had been chosen.
    orientation: VisualOrientation
    # Posted as the image's alt text: the strip's Korean dialogue otherwise exists only as pixels.
    alt_text: str = Field(
        default="",
        description="The comic for a screen reader, in Korean: one sentence per panel, with its line of dialogue.",
    )


class CommentReply(BaseModel):
    reply: bool = Field(description="False to leave the comment unanswered.")
    text: str = Field(default="", description="The reply to post, when `reply` is true.")


class ImageAsset(BaseModel):
    """A representative image downloaded from a source page (its og:image / twitter:image),
    carried through to Threads delivery. `source_url` is the article the image belongs
    to (shown for attribution); `image_url` is where the image bytes came from."""

    data: bytes
    source_url: str
    image_url: str
    content_type: str = "image/png"
    alt: str = ""


class ThreadsDelivery(NamedTuple):
    """How much of the intended post set actually landed. `posted`/`expected` count the ROOT plus
    one per reply, so a 5-story digest is 6 expected posts.

    Deliberately NOT a bool: "the root went up" and "the digest went up" are different outcomes,
    and a partial reply chain (4 of 6 posts on 2026-07-17) used to be reported as plain success.
    CALLERS: a NamedTuple is ALWAYS truthy, so never branch on the value itself — use `published`.
    """

    posted: int
    expected: int
    # Whether the root that went up actually carried the day's image. Defaults to False so every
    # existing construction (a failure verdict, a research report) is unchanged; the visual Lambda
    # alerts on it, because "posted 6/6" says nothing about a text-only fallback that
    # silently dropped the visual.
    with_image: bool = False
    # Nothing is KNOWN to have posted, but the root may be live: its publish step failed in a way
    # that can come back for a post that went up (an error status, a timeout). Callers keep the
    # date's marker so a replay cannot post a second root.
    uncertain: bool = False
    # The live root, and replies Threads published as standalone posts instead (2026-10-27): the
    # operator's alert names them, because only they can delete one or reply the story by hand.
    root_id: str = ""
    detached: tuple[str, ...] = ()

    @property
    def published(self) -> bool:
        """True when what landed is usable: the root, plus at least one reply when replies were
        intended. An all-replies-failed root is a lone image with no stories, never a digest."""
        return self.posted > 1 if self.expected > 1 else self.posted > 0

    @property
    def partial(self) -> bool:
        """Published, but some intended posts are missing — the reader sees a truncated chain."""
        return self.published and self.posted < self.expected

    def summary(self) -> str:
        return f"{self.posted}/{self.expected} posts"


class TrendEvidence(BaseModel):
    date: str  # YYYY-MM-DD, stamped by code from the digest date (never the LLM)
    summary: str


class TrendStatus(str, Enum):
    ACTIVE = "active"
    COOLING = "cooling"


class Trend(BaseModel):
    id: str  # stable slug; identity survives title rephrasing
    title: str
    status: TrendStatus = TrendStatus.ACTIVE
    first_seen: str = ""
    last_seen: str = ""
    evidence: list[TrendEvidence] = Field(default_factory=list)

    def momentum(self, today: date, half_life_days: float) -> float:
        # Recency-decayed evidence count: each piece of evidence contributes
        # 0.5 ** (age_days / half_life). Recent, frequently-cited trends rank highest.
        if half_life_days <= 0:
            return float(len(self.evidence))
        total = 0.0
        for ev in self.evidence:
            try:
                age = (today - date.fromisoformat(ev.date)).days
            except ValueError:
                continue
            if age < 0:
                age = 0
            total += 0.5 ** (age / half_life_days)
        return total


class TrendMemory(BaseModel):
    trends: list[Trend] = Field(default_factory=list)

    def by_id(self, trend_id: str) -> Trend | None:
        return next((t for t in self.trends if t.id == trend_id), None)


class SourceStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    FAILED = "failed"
    # The source served items from its S3 park file, but that file is too old (a stalled local
    # sync) or could not be read at all. The run still produced items, so it is not a FAILURE —
    # but it must not read as a healthy OK either, which is how a dead cron stayed invisible.
    STALE = "stale"
    # The source produced items, on time, from only a FRACTION of its inputs (most of its feeds
    # failed). Also not a failure and not OK: a source can shrink from 40 accounts to 3 and still
    # look perfectly healthy, which is how X quietly stopped contributing.
    DEGRADED = "degraded"


class SourceHealth(BaseModel):
    name: str
    item_count: int = 0
    status: SourceStatus = SourceStatus.OK
    detail: str | None = None


class HealthReport(BaseModel):
    sources: list[SourceHealth] = Field(default_factory=list)

    @property
    def failed_sources(self) -> list[str]:
        return [s.name for s in self.sources if s.status == SourceStatus.FAILED]

    @property
    def stale_sources(self) -> list[str]:
        return [s.name for s in self.sources if s.status == SourceStatus.STALE]

    @property
    def degraded_sources(self) -> list[str]:
        return [s.name for s in self.sources if s.status == SourceStatus.DEGRADED]

    @property
    def empty_sources(self) -> list[str]:
        """Sources that ran cleanly and returned NOTHING. Not a failure on its own (reddit/x are
        legitimately quiet some days), so callers decide which of these matter — see
        CollectorsConfig.alert_on_empty."""
        return [s.name for s in self.sources if s.status == SourceStatus.EMPTY]

    def summary(self) -> str:
        return "\n".join(
            f"[{s.status.value.upper()}] {s.name}: {s.item_count} items" + (f" — {s.detail}" if s.detail else "")
            for s in self.sources
        )
