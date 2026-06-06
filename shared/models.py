from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .constants import SourceType


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


class DigestResult(BaseModel):
    digest_text: str
    ranked_items: list[RankedItem]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_collected: int = 0
    total_ranked: int = 0


class VisualBrief(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    caption: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=4000)
    # The synopsis chooses the aspect ratio that fits the visual (a wide 4-panel strip,
    # a square meme, a tall infographic); the generator maps it to a supported size.
    orientation: Literal["square", "landscape", "portrait"] = "portrait"


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

    def summary(self) -> str:
        return "\n".join(
            f"[{s.status.value.upper()}] {s.name}: {s.item_count} items" + (f" — {s.detail}" if s.detail else "")
            for s in self.sources
        )
