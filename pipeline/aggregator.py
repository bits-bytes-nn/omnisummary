from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from shared import CollectedItem, SourceType, generate_item_id, logger, normalize_title
from shared.formatting import RETWEET_PREFIX

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "ref_url"}
# Share-link hosts mapped to the canonical one, and params that mark only a share or a position on one
# host: a video shared at 0:30 is still the same story.
_HOST_ALIASES = {
    "twitter.com": "x.com",
    "mobile.twitter.com": "x.com",
    "old.reddit.com": "reddit.com",
    "m.youtube.com": "youtube.com",
}
_YOUTUBE_VIDEO_PATH = re.compile(r"^/(?:shorts|live)/([\w-]+)$")
_X_STATUS_PATH = re.compile(r"^/(?:i/web|[^/]+)/status/(\d+)")
_HOST_TRACKING_PARAMS = {"x.com": {"s", "t"}, "youtube.com": {"si", "feature", "t"}}


def normalize_url(url: str) -> str:
    # Collapse trivial variants (scheme, host case and aliases, trailing slash, tracking
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
    host = _HOST_ALIASES.get(host, host)
    path = parts.path.rstrip("/") or "/"
    params = parse_qsl(parts.query, keep_blank_values=True)
    if host == "youtu.be" and path != "/":
        host, path, params = "youtube.com", "/watch", [("v", path.lstrip("/")), *params]
    if host == "youtube.com" and (video := _YOUTUBE_VIDEO_PATH.match(path)):
        path, params = "/watch", [("v", video.group(1)), *params]
    if host == "x.com" and (status := _X_STATUS_PATH.match(path)):
        # Handles are case-insensitive and /i/status/ID is the same post as /<user>/status/ID.
        path = f"/i/status/{status.group(1)}"
    dropped = _TRACKING_PARAMS | _HOST_TRACKING_PARAMS.get(host, set())
    kept = [
        (k, v) for k, v in params if not k.lower().startswith(_TRACKING_PARAM_PREFIXES) and k.lower() not in dropped
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
        url_dupes = 0
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
            key = normalize_url(item.url)
            # Pinned items (user-specified via --pin-url) bypass cross-day dedup: the user
            # asked for this exact URL today, even if it was published in a recent digest.
            if key in exclude and not item.metadata.get("pinned"):
                cross_day_skipped += 1
                continue
            if key in seen_urls:
                logger.debug("Duplicate URL skipped: '%s'", item.url)
                seen_urls[key] = self._pick_survivor(seen_urls[key], item)
                url_dupes += 1
            else:
                seen_urls[key] = item

        if cross_day_skipped:
            logger.info("Skipped %d item(s) already published on a recent day", cross_day_skipped)

        url_deduped = list(seen_urls.values())

        seen_titles: dict[str, CollectedItem] = {}
        clipped: dict[str, int] = {}
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
            # A pinned duplicate is not lost here: _pick_survivor always keeps the pinned copy. Two pins
            # sharing a title are both kept, since each was asked for by URL.
            norm = _seen_key(norm, item, seen_titles, clipped)
            if norm in seen_titles and not (item.metadata.get("pinned") and seen_titles[norm].metadata.get("pinned")):
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
                if _is_clipped(item):
                    clipped[norm] = len(item.title.rstrip())
                deduplicated.append(item)

        if empty_norm:
            logger.info("Kept %d item(s) whose title normalizes to empty out of title dedup", empty_norm)

        # The ranker keys its scores by item_id, so two different stories sharing one (a bare RSS GUID
        # like "1234" from two feeds) would collapse into one and the other vanish silently.
        seen_ids: set[str] = set()
        for item in deduplicated:
            if item.item_id in seen_ids:
                item.item_id = generate_item_id(normalize_url(item.url))
            seen_ids.add(item.item_id)

        logger.info(
            "Aggregated %d items → %d after deduplication (%d url, %d title)",
            len(items),
            len(deduplicated),
            url_dupes,
            title_dupes,
        )

        return deduplicated

    @classmethod
    def _pick_survivor(cls, incumbent: CollectedItem, dupe: CollectedItem) -> CollectedItem:
        """Choose which of two duplicates to keep, then back-fill the loser's metadata onto it.

        The ranker and digest read the survivor's `text`, so keep whichever body is richer rather
        than blindly keeping the first-seen item (a thin Reddit .rss link-post would otherwise beat
        the same article's full-text RSS/web entry just because its collector ran first). A pinned
        item always wins; otherwise the longer body wins; ties keep the incumbent so
        collector order still breaks ties deterministically."""
        if dupe.metadata.get("pinned") and not incumbent.metadata.get("pinned"):
            winner, loser = dupe, incumbent
        elif incumbent.metadata.get("pinned"):
            winner, loser = incumbent, dupe
        elif bool(RETWEET_PREFIX.match(incumbent.title)) != bool(RETWEET_PREFIX.match(dupe.title)):
            # The post itself over its retweet: the retweet's body is longer only by its "RT name:".
            winner, loser = (dupe, incumbent) if RETWEET_PREFIX.match(incumbent.title) else (incumbent, dupe)
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


def _is_clipped(item: CollectedItem) -> bool:
    """RSSHub cuts a long X post's title at a fixed length and ends it with an ellipsis. Only X: a
    Reddit or blog title ending in "..." is the author's own and can be short ("Introducing...")."""
    return item.source_type == SourceType.X and item.title.rstrip().endswith(("...", "\u2026"))


def _seen_key(norm: str, item: CollectedItem, seen_titles: dict, clipped: dict[str, int]) -> str:
    """The seen title this one duplicates, or `norm` itself. RSSHub clips every long X title at the same
    length, so a retweet, whose "RT name:" prefix spends part of that length, ends earlier than the post
    it retweets and matches only as a prefix of it (both omarsar0 retweets on 2026-10-26 survived). Only
    two titles clipped at that same length are compared so: a short post the author ended with "..."
    ("Big news coming...") is not a clip and must not swallow every title that starts the same way."""
    if norm in seen_titles:
        return norm
    if not _is_clipped(item):
        return norm
    length = len(item.title.rstrip())
    for key, key_length in clipped.items():
        if key_length == length and (key.startswith(norm) or norm.startswith(key)):
            return key
    return norm
