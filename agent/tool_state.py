from __future__ import annotations

import json
from pathlib import Path

from shared import CollectedItem, DigestResult, RankedItem, logger


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

    def get_digest_text(self) -> str:
        return self._digest_result.digest_text if self._digest_result else ""

    def get_item_count(self) -> int:
        return len(self._ranked_items)

    def load_from(self, other: DigestStateManager) -> None:
        self._collected_items = dict(other._collected_items)
        self._ranked_items = list(other._ranked_items)
        self._digest_result = other._digest_result

    def export_state(self) -> dict:
        return {
            "collected_items": {
                item_id: item.model_dump(mode="json") for item_id, item in self._collected_items.items()
            },
            "ranked_items": [ri.model_dump(mode="json") for ri in self._ranked_items],
            "digest_result": self._digest_result.model_dump(mode="json") if self._digest_result else None,
        }

    def clear(self) -> None:
        self._collected_items.clear()
        self._ranked_items.clear()
        self._digest_result = None

    def save_to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.export_state()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved digest state to '%s'", path)

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

    @classmethod
    def load_from_file(cls, path: Path) -> DigestStateManager:
        if not path.exists():
            logger.warning("Digest state file not found: '%s'", path)
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        manager = cls.load_from_dict(data)
        logger.info(
            "Loaded digest state from '%s': %d collected, %d ranked",
            path,
            len(manager._collected_items),
            len(manager._ranked_items),
        )
        return manager
