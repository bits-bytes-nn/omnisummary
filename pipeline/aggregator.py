from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from shared import CollectedItem, logger, normalize_title

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "ref_url"}


def normalize_url(url: str) -> str:
    # Collapse trivial variants (scheme, host case, trailing slash, tracking
    # params, fragment) so the same article from two sources — or the same article
    # seen on different days — dedups on URL.
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


class ContentAggregator:

    def aggregate(self, items: list[CollectedItem], exclude_urls: set[str] | None = None) -> list[CollectedItem]:
        # Drop anything already published on a recent day (cross-day dedup): the caller
        # passes normalized URLs from the published-URL ledger so the same article isn't
        # re-summarized days apart. Excluding here (before ranking) also saves ranker tokens.
        exclude = exclude_urls or set()
        cross_day_skipped = 0
        seen_urls: dict[str, CollectedItem] = {}

        # Drop items missing a url or title up front. They can't be linked or rendered in the
        # digest, and worse, every empty-url item normalizes to the same "" key — so without this
        # they'd dedup against EACH OTHER and silently swallow siblings. Explicit boundary check.
        malformed = 0
        usable: list[CollectedItem] = []
        for item in items:
            if not item.url.strip() or not item.title.strip():
                malformed += 1
                continue
            usable.append(item)
        if malformed:
            logger.warning("Dropped %d item(s) missing a url or title before dedup", malformed)

        for item in usable:
            key = self._normalize_url(item.url)
            # Pinned items (user-specified via --pin-url) bypass cross-day dedup: the user
            # asked for this exact URL today, even if it was published in a recent digest.
            if key in exclude and not item.metadata.get("pinned"):
                cross_day_skipped += 1
                continue
            if key in seen_urls:
                logger.debug("Duplicate URL skipped: '%s'", item.url)
                self._fill_missing_metadata(seen_urls[key], item)
            else:
                seen_urls[key] = item

        if cross_day_skipped:
            logger.info("Skipped %d item(s) already published on a recent day", cross_day_skipped)

        url_deduped = list(seen_urls.values())

        seen_titles: dict[str, CollectedItem] = {}
        deduplicated: list[CollectedItem] = []
        title_dupes = 0
        for item in url_deduped:
            norm = normalize_title(item.title)
            # Pinned items (user-specified via --pin-url) bypass title dedup too — mirroring the
            # URL-dedup bypass above. Otherwise a pin sharing a normalized title with an
            # earlier-inserted story is dropped here, before the ranker's pin-recovery can see it,
            # silently defeating the --pin-url force-inclusion guarantee.
            if norm in seen_titles and not item.metadata.get("pinned"):
                logger.debug(
                    "Duplicate title skipped: '%s' (same as '%s')",
                    item.title[:60],
                    seen_titles[norm].title[:60],
                )
                self._fill_missing_metadata(seen_titles[norm], item)
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
    def _fill_missing_metadata(kept: CollectedItem, dupe: CollectedItem) -> None:
        # Only fill keys the kept item lacks — never overwrite its own origin/engagement
        # metadata (feed_url, subreddit, channel_url, view_count) with a lower-priority dupe's.
        for k, v in dupe.metadata.items():
            kept.metadata.setdefault(k, v)

    _normalize_url = staticmethod(normalize_url)

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
