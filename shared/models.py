from __future__ import annotations

import hashlib
import re
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import SourceType, VisualOrientation
from .logger import logger

# An XML/HTML-style tag: needs a letter after '<' and a closing '>', so prose like "<2%" survives.
_MARKUP_TAG_RE = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*?)?/?>")


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
    categories: list[str] = Field(default_factory=list)
    # True for a candidate handed over ONLY as merge backfill: the source-slot guarantees are
    # enforced on the first top_n items (the ones the reader actually gets), and these extras exist
    # so the editor can still reach top_n distinct stories after merging same-event items. Fully
    # usable — just not part of the diversified core. Defaults to False, so a snapshot stored before
    # this field existed still loads.
    backfill: bool = False


class DigestItem(BaseModel):
    """One story in the digest, as plain prose. Renderers add channel-specific markup
    (Slack Block Kit, Threads plain text); the LLM never writes Slack mrkdwn itself."""

    title: str  # Korean display title
    url: str
    source_tag: str = ""  # e.g. "r/LocalLLaMA", "@karpathy", "arxiv.org" — set by code
    metrics: str = ""  # e.g. "👍 +44", "▶️ 12,000" — set by code
    body: str  # 2-4 sentences: what it is and why it matters
    implication: str = ""  # one sharp closing line (Gruber voice)


class DigestContent(BaseModel):
    """Structured digest the digest LLM returns. `lead` is the columnist take connecting
    the headline to its trend arc; `headline_index` (1-based into items) is the story the
    lead is about and the one the daily visual depicts."""

    lead: str
    headline_index: int = 1
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
    # Coverage the run was judged against (pipeline.ranking_min_coverage_ratio), carried so the
    # verdict below needs no config access. 0.0 — the default for a directly constructed or older
    # persisted health — means "coverage alone never degrades the run".
    min_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def coverage(self) -> float:
        """Share of the day's candidates that ended up scored; 1.0 when there was nothing to rank."""
        return self.items_scored / self.items_total if self.items_total else 1.0

    @property
    def degraded(self) -> bool:
        """True when candidates were LOST — a batch that failed every retry — or when the pass
        scored less than min_coverage_ratio of the day's candidates. The coverage arm matters
        because a batch whose response never parsed twice over used to leave every counter at
        zero, so the alerting stayed silent while a whole batch vanished from the pool."""
        return self.batches_failed > 0 or self.coverage < self.min_coverage_ratio

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
    total_collected: int = 0
    total_ranked: int = 0
    # Set by the pipeline after ranking, so the delivery/alerting layer can report a digest built on
    # an incomplete candidate pool. None means "not recorded" (older snapshots, direct construction).
    ranking_health: RankingHealth | None = None


class VisualBrief(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    caption: str = Field(min_length=1, max_length=600)
    prompt: str = Field(min_length=1, max_length=4000)

    # The synopsis chooses the aspect ratio that fits the visual (a wide 4-panel strip,
    # a square meme, a tall infographic); the generator maps it to a supported size. The vocabulary
    # lives in shared/constants (VisualOrientation) because pipeline.image_sizes must offer exactly
    # these keys — a hardcoded Literal here let the two drift silently.
    orientation: VisualOrientation = "portrait"

    @field_validator("title", "caption", mode="after")
    @classmethod
    def _drop_markup(cls, value: str) -> str:
        """Strip tag-like markup from the human-facing fields. These are plain prose bound for a
        post caption, but a structured-output slip can trail the model's own scaffolding into the
        string: the 2026-08-17 caption ended with `</caption>\\n<parameter name="orientation">`
        and that markup was published verbatim. The pattern requires a letter after `<` and a
        closing `>`, so ordinary prose like "<2%" or "a < b" is left alone."""
        return _MARKUP_TAG_RE.sub("", value).strip()

    @model_validator(mode="after")
    def _drop_bled_field_value(self) -> VisualBrief:
        """Drop a trailing line that is just ANOTHER field's value.

        The 2026-08-17 leak (`</caption>\\n<parameter name="orientation">landscape`) was the tagged
        form of a recurring structured-output slip: the model runs the next field into the previous
        string. A 2026-08-18 local run produced the TAG-LESS form — a caption ending in a bare
        `\\nportrait` — which the markup strip above cannot see. The rule is derived, not a word
        list: a prose field must not END with a standalone line that is one of the orientation
        field's ALLOWED values, so prose that merely contains the word survives.

        Compared against every value of the Literal, not just the parsed `orientation`: on 08-17 the
        bled word was 'landscape' while orientation had fallen back to the default 'portrait', so a
        self-comparison did not match and the leak was published."""
        candidates = {v.casefold() for v in get_args(type(self).model_fields["orientation"].annotation)}
        for name in ("title", "caption"):
            value = getattr(self, name)
            head, sep, last = value.rpartition("\n")
            if sep and last.strip().casefold() in candidates:
                setattr(self, name, head.strip())
                logger.warning("Dropped a bled '%s' value from VisualBrief.%s", last.strip(), name)
        return self


class ImageAsset(BaseModel):
    """A representative image downloaded from a source page (its og:image / twitter:image),
    carried through to Slack/Threads delivery. `source_url` is the article the image belongs
    to (shown for attribution); `image_url` is where the image bytes came from."""

    data: bytes
    source_url: str
    image_url: str
    content_type: str = "image/png"
    alt: str = ""


class TrendEvidence(BaseModel):
    date: str  # YYYY-MM-DD, stamped by code from the digest date (never the LLM)
    summary: str
    item_id: str = ""
    url: str = ""


class TrendStatus(str, Enum):
    ACTIVE = "active"
    COOLING = "cooling"
    ARCHIVED = "archived"


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

    def search(self, query: str, *, today: date, half_life_days: float, top_k: int) -> list[Trend]:
        """Return up to top_k active/cooling trends most relevant to the query, ranked by
        (distinct query-term hits, then momentum). Empty query → top trends by momentum.
        Deterministic term matching — no embeddings."""
        terms = {t for t in query.lower().split() if t}
        candidates = [t for t in self.trends if t.status != TrendStatus.ARCHIVED]

        def hits(trend: Trend) -> int:
            if not terms:
                return 0
            hay = (trend.title + " " + " ".join(ev.summary for ev in trend.evidence)).lower()
            return sum(1 for term in terms if term in hay)

        scored = [(trend, hits(trend)) for trend in candidates]
        if terms:
            scored = [(trend, h) for trend, h in scored if h > 0]
        scored.sort(key=lambda pair: (pair[1], pair[0].momentum(today, half_life_days)), reverse=True)
        return [trend for trend, _ in scored[:top_k]]


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
    def has_failures(self) -> bool:
        return any(s.status == SourceStatus.FAILED for s in self.sources)

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
