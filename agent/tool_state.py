from __future__ import annotations

from shared import DigestContent, DigestResult, RankedItem, logger


class DigestStateManager:
    """The digest snapshot persisted per date: the ranked items and the digest result.

    Readers (the visual Lambda, cross-day dedup, recall_digest) only ever use these two. Each
    RankedItem already embeds its CollectedItem, so the snapshot is self-contained; a separate
    collected-items map stored every ranked item twice and was what overflowed the AgentCore Memory
    100k-char event limit on busy days."""

    def __init__(self) -> None:
        self._ranked_items: list[RankedItem] = []
        self._digest_result: DigestResult | None = None

    def store_digest(self, ranked: list[RankedItem], digest: DigestResult) -> None:
        self._ranked_items = ranked
        self._digest_result = digest
        logger.info("Stored digest state: %d ranked items", len(self._ranked_items))

    def get_ranked_items(self) -> list[RankedItem]:
        return self._ranked_items

    def get_content(self) -> DigestContent | None:
        return self._digest_result.content if self._digest_result else None

    def export_state(self) -> dict:
        return {
            "ranked_items": [ri.model_dump(mode="json") for ri in self._ranked_items],
            # ranked_items is excluded from the embedded digest result — it is already persisted at
            # the top level; re-embedding it doubled the snapshot.
            "digest_result": (
                self._digest_result.model_dump(mode="json", exclude={"ranked_items"}) if self._digest_result else None
            ),
        }

    @classmethod
    def load_from_dict(cls, data: dict) -> DigestStateManager:
        """Snapshots written before collected_items was dropped still load: unknown keys are ignored."""
        manager = cls()
        manager._ranked_items = [RankedItem.model_validate(ri) for ri in data.get("ranked_items", [])]
        digest_data = data.get("digest_result")
        if digest_data:
            manager._digest_result = DigestResult.model_validate(digest_data)
        return manager
