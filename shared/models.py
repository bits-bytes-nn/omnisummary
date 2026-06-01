from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Any

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
            f"[{s.status.value.upper()}] {s.name}: {s.item_count} items"
            + (f" — {s.detail}" if s.detail else "")
            for s in self.sources
        )
