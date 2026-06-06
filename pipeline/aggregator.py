from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from shared import CollectedItem, logger, normalize_title

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "ref_url"}


class ContentAggregator:

    def aggregate(self, items: list[CollectedItem]) -> list[CollectedItem]:
        seen_urls: dict[str, CollectedItem] = {}

        for item in items:
            key = self._normalize_url(item.url)
            if key in seen_urls:
                logger.debug("Duplicate URL skipped: '%s'", item.url)
                seen_urls[key].metadata.update(item.metadata)
            else:
                seen_urls[key] = item

        url_deduped = list(seen_urls.values())

        seen_titles: dict[str, CollectedItem] = {}
        deduplicated: list[CollectedItem] = []
        title_dupes = 0
        for item in url_deduped:
            norm = normalize_title(item.title)
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
    def _normalize_url(url: str) -> str:
        # Collapse trivial variants (scheme, host case, trailing slash, tracking
        # params, fragment) so the same article from two sources dedups on URL.
        if not url:
            return url
        try:
            parts = urlsplit(url.strip())
        except ValueError:
            return url
        if not parts.netloc:
            return url.strip()
        host = parts.netloc.lower().removeprefix("www.")
        path = parts.path.rstrip("/") or "/"
        kept = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith(_TRACKING_PARAM_PREFIXES) and k.lower() not in _TRACKING_PARAMS
        ]
        query = urlencode(sorted(kept))
        return urlunsplit(("https", host, path, query, ""))

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
