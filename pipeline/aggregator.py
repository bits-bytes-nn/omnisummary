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
                seen_urls[key] = self._pick_survivor(seen_urls[key], item)
            else:
                seen_urls[key] = item

        if cross_day_skipped:
            logger.info("Skipped %d item(s) already published on a recent day", cross_day_skipped)

        url_deduped = list(seen_urls.values())

        seen_titles: dict[str, CollectedItem] = {}
        deduplicated: list[CollectedItem] = []
        title_dupes = 0
        empty_norm = 0
        for item in url_deduped:
            norm = normalize_title(item.title)
            # A title made only of punctuation/emoji (common for X posts) normalizes to "", and
            # every such item lands in the SAME bucket — so without this they'd dedup against each
            # other and silently swallow unrelated siblings. Same trap as the empty-url guard
            # above; these items have distinct URLs, so pass them through untouched.
            if not norm:
                empty_norm += 1
                deduplicated.append(item)
                continue
            # Pinned items (user-specified via --pin-url) bypass title dedup too — mirroring the
            # URL-dedup bypass above. Otherwise a pin sharing a normalized title with an
            # earlier-inserted story is dropped here, before the ranker's pin-recovery can see it,
            # silently defeating the --pin-url force-inclusion guarantee.
            if norm in seen_titles and not item.metadata.get("pinned"):
                prev = seen_titles[norm]
                logger.debug(
                    "Duplicate title skipped: '%s' (same as '%s')",
                    item.title[:60],
                    prev.title[:60],
                )
                survivor = self._pick_survivor(prev, item)
                if survivor is not prev:
                    # A later duplicate has richer content; swap it in at the survivor's position
                    # so the digest reads the fuller text (order otherwise preserved).
                    seen_titles[norm] = survivor
                    deduplicated[deduplicated.index(prev)] = survivor
                title_dupes += 1
            else:
                seen_titles[norm] = item
                deduplicated.append(item)

        if empty_norm:
            logger.info("Kept %d item(s) whose title normalizes to empty out of title dedup", empty_norm)

        logger.info(
            "Aggregated %d items → %d after deduplication (%d url, %d title)",
            len(items),
            len(deduplicated),
            len(items) - len(url_deduped),
            title_dupes,
        )

        return deduplicated

    @classmethod
    def _pick_survivor(cls, incumbent: CollectedItem, dupe: CollectedItem) -> CollectedItem:
        """Choose which of two duplicates to keep, then back-fill the loser's metadata onto it.

        The ranker and digest read the survivor's `text`, so keep whichever body is richer rather
        than blindly keeping the first-seen item (a thin Reddit .rss link-post would otherwise beat
        the same article's full-text RSS/web entry just because its collector ran first). A pinned
        item always wins; otherwise the meaningfully-longer body wins; ties keep the incumbent so
        collector order still breaks ties deterministically."""
        if dupe.metadata.get("pinned") and not incumbent.metadata.get("pinned"):
            winner, loser = dupe, incumbent
        elif incumbent.metadata.get("pinned"):
            winner, loser = incumbent, dupe
        elif len((dupe.text or "").strip()) > len((incumbent.text or "").strip()):
            winner, loser = dupe, incumbent
        else:
            winner, loser = incumbent, dupe
        cls._fill_missing_metadata(winner, loser)
        return winner

    @staticmethod
    def _fill_missing_metadata(kept: CollectedItem, dupe: CollectedItem) -> None:
        # Only fill keys the kept item lacks — never overwrite its own origin/engagement
        # metadata (feed_url, subreddit, channel_url, view_count) with a lower-priority dupe's.
        for k, v in dupe.metadata.items():
            kept.metadata.setdefault(k, v)

    _normalize_url = staticmethod(normalize_url)
