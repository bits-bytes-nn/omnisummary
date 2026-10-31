from __future__ import annotations

from dataclasses import dataclass, field

from domain.models import DigestContent, DigestResult, RankedItem


@dataclass
class DigestSnapshot:
    """The digest persisted per date: the ranked items and the digest result.

    Readers (the visual Lambda, cross-day dedup, recall) only ever use these two. Each RankedItem
    already embeds its CollectedItem, so the snapshot is self-contained; a separate collected-items
    map stored every ranked item twice and was what overflowed the AgentCore Memory 100k-char event
    limit on busy days."""

    ranked_items: list[RankedItem] = field(default_factory=list)
    digest: DigestResult | None = None

    @property
    def content(self) -> DigestContent | None:
        return self.digest.content if self.digest else None

    def to_dict(self) -> dict:
        return {
            "ranked_items": [ri.model_dump(mode="json") for ri in self.ranked_items],
            # ranked_items is excluded from the embedded digest result — it is already persisted at
            # the top level; re-embedding it doubled the snapshot.
            "digest_result": self.digest.model_dump(mode="json", exclude={"ranked_items"}) if self.digest else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DigestSnapshot:
        """Snapshots written before collected_items was dropped still load: unknown keys are ignored."""
        digest_data = data.get("digest_result")
        return cls(
            ranked_items=[RankedItem.model_validate(ri) for ri in data.get("ranked_items", [])],
            digest=DigestResult.model_validate(digest_data) if digest_data else None,
        )
