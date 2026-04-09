from __future__ import annotations

import re
import unicodedata

from shared import CollectedItem, logger


class ContentAggregator:

    def aggregate(self, items: list[CollectedItem]) -> list[CollectedItem]:
        seen_urls: dict[str, CollectedItem] = {}

        for item in items:
            if item.url in seen_urls:
                logger.debug("Duplicate URL skipped: '%s'", item.url)
                seen_urls[item.url].metadata.update(item.metadata)
            else:
                seen_urls[item.url] = item

        url_deduped = list(seen_urls.values())

        seen_titles: dict[str, CollectedItem] = {}
        deduplicated: list[CollectedItem] = []
        title_dupes = 0
        for item in url_deduped:
            norm = self._normalize_title(item.title)
            if norm in seen_titles:
                logger.debug(
                    "Duplicate title skipped: '%s' (same as '%s')",
                    item.title[:60],
                    seen_titles[norm].title[:60],
                )
                seen_titles[norm].metadata.update(item.metadata)
                title_dupes += 1
            else:
                seen_titles[norm] = item
                deduplicated.append(item)

        for item in deduplicated:
            item.metadata = self._normalize_metadata(item.metadata)

        logger.info(
            "Aggregated %d items → %d after deduplication (%d url, %d title)",
            len(items),
            len(deduplicated),
            len(items) - len(url_deduped),
            title_dupes,
        )

        return deduplicated

    @staticmethod
    def _normalize_title(title: str) -> str:
        title = unicodedata.normalize("NFKC", title)
        title = re.sub(r"<[^>]+>", "", title)
        title = re.sub(r"[^\w\s]", "", title.lower())
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def _normalize_metadata(metadata: dict) -> dict:
        normalized: dict = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            elif isinstance(value, (list, dict)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized
