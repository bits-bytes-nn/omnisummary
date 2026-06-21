from __future__ import annotations

from shared import CollectedItem, DigestContent, DigestResult, RankedItem, logger


class DigestStateManager:
    def __init__(self) -> None:
        self._collected_items: dict[str, CollectedItem] = {}
        self._ranked_items: list[RankedItem] = []
        self._digest_result: DigestResult | None = None

    def store_digest(
        self,
        items: list[CollectedItem],
        ranked: list[RankedItem],
        digest: DigestResult,
    ) -> None:
        self._collected_items = {item.item_id: item for item in items}
        self._ranked_items = ranked
        self._digest_result = digest
        logger.info(
            "Stored digest state: %d collected items, %d ranked items",
            len(self._collected_items),
            len(self._ranked_items),
        )

    def get_item_by_number(self, number: int) -> RankedItem | None:
        idx = number - 1
        if 0 <= idx < len(self._ranked_items):
            return self._ranked_items[idx]
        return None

    def get_item(self, item_id: str) -> CollectedItem | None:
        return self._collected_items.get(item_id)

    def get_ranked_items(self) -> list[RankedItem]:
        return self._ranked_items

    def get_content(self) -> DigestContent | None:
        return self._digest_result.content if self._digest_result else None

    def get_item_count(self) -> int:
        return len(self._ranked_items)

    def export_state(self) -> dict:
        # Only the ranked items (and the collected items they reference) are ever read
        # back by the follow-up agent; persisting all collected items blew past the
        # AgentCore Memory 100k-char event limit on busy days. Each RankedItem already
        # embeds its CollectedItem, so the ranked set is self-contained.
        ranked_ids = {ri.item.item_id for ri in self._ranked_items}
        return {
            "collected_items": {
                item_id: item.model_dump(mode="json")
                for item_id, item in self._collected_items.items()
                if item_id in ranked_ids
            },
            "ranked_items": [ri.model_dump(mode="json") for ri in self._ranked_items],
            # Exclude ranked_items from the embedded digest result — it's already persisted at
            # the top level above; re-embedding it doubled the snapshot and tripped the 100k cap.
            "digest_result": (
                self._digest_result.model_dump(mode="json", exclude={"ranked_items"}) if self._digest_result else None
            ),
        }

    def clear(self) -> None:
        self._collected_items.clear()
        self._ranked_items.clear()
        self._digest_result = None

    @classmethod
    def load_from_dict(cls, data: dict) -> DigestStateManager:
        manager = cls()

        collected = data.get("collected_items", {})
        if isinstance(collected, dict):
            manager._collected_items = {
                item_id: CollectedItem.model_validate(item_data) for item_id, item_data in collected.items()
            }
        elif isinstance(collected, list):
            for item_data in collected:
                item = CollectedItem.model_validate(item_data)
                manager._collected_items[item.item_id] = item

        if not manager._collected_items and "items" in data:
            for item_data in data["items"]:
                item = CollectedItem.model_validate(item_data)
                manager._collected_items[item.item_id] = item

        ranked = data.get("ranked_items", [])
        manager._ranked_items = [RankedItem.model_validate(ri) for ri in ranked]

        digest_data = data.get("digest_result")
        if digest_data:
            manager._digest_result = DigestResult.model_validate(digest_data)

        return manager
