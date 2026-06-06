from __future__ import annotations

import re
import unicodedata
from datetime import date
from urllib.parse import urlparse

from .constants import SourceType
from .models import CollectedItem
from .utils import truncate_text_by_tokens

YOUTUBE_VIEWS_EMOJI = ":arrow_forward:"
RSS_NAME_DELIMITERS = (" - ", " — ")


def agi_countdown_intro(date_str: str, template: str, today: date) -> str:
    """The tongue-in-cheek 'AGI N days away' intro, computed in code (never the LLM) from a fixed
    D-day so it's accurate and ticks down daily. Applied at POST time (not digest generation) so it
    lands on every channel and every run, including reposts from an older stored digest. Returns ""
    when disabled, malformed, or the D-day has passed."""
    if not date_str or not template:
        return ""
    try:
        days = (date.fromisoformat(date_str) - today).days
    except ValueError:
        return ""
    return template.format(days=days) if days > 0 else ""


def normalize_title(title: str) -> str:
    """Normalize a title for dedup/clustering: strip HTML, lowercase, drop punctuation,
    collapse whitespace. Shared by the aggregator (title dedup) and ranker (topic-coherent
    batching) so both agree on what 'the same title' means."""
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def format_collected_item(
    item: CollectedItem,
    *,
    index: int,
    max_tokens: int,
    fields: list[tuple[str, str]],
    text_label: str = "Text",
) -> str:
    """Render a CollectedItem as a labelled `=== Item N ===` block for LLM input.

    `fields` are the leading "Label: value" lines in the caller's chosen order
    (Title, Source, Author, Score, ...); the body text (truncated to `max_tokens`)
    is appended last under `text_label`. Shared so the ranker, digest generator,
    and agent get_detail tool stay in lockstep.
    """
    snippet = truncate_text_by_tokens(item.text, max_tokens)
    lines = [f"=== Item {index} ==="]
    lines.extend(f"{label}: {value}" for label, value in fields)
    lines.append(f"{text_label}:\n{snippet}")
    return "\n".join(lines) + "\n"


def clean_rss_feed_name(feed_title: str, feed_url: str) -> str:
    """Derive a short, human-readable source name from an RSS feed's title/URL.

    Strips the common "Site Name - Section" / "Site Name — Section" suffixes, falling
    back to the feed's hostname (without www./feeds. prefixes) when no title exists.
    """
    name = feed_title.strip()
    for delimiter in RSS_NAME_DELIMITERS:
        name = name.split(delimiter)[0]
    name = name.strip()
    if name:
        return name
    if feed_url:
        return urlparse(feed_url).netloc.removeprefix("www.").removeprefix("feeds.")
    return ""


def resolve_origin_key(item: CollectedItem) -> str | None:
    """Per-origin diversity key (e.g. a single channel/subreddit/feed/account)."""
    meta = item.metadata
    if item.source_type == SourceType.YOUTUBE:
        return meta.get("channel_url")
    if item.source_type == SourceType.REDDIT:
        return meta.get("subreddit")
    if item.source_type == SourceType.RSS:
        return meta.get("feed_url")
    if item.source_type == SourceType.X:
        return item.author
    return None


def format_origin_label(item: CollectedItem) -> str:
    """Plain-text origin label fed to the ranking prompt (no Slack markup)."""
    meta = item.metadata
    if item.source_type == SourceType.REDDIT:
        return f"r/{meta.get('subreddit', '')}" if meta.get("subreddit") else ""
    if item.source_type == SourceType.YOUTUBE:
        return meta.get("channel_url", "")
    if item.source_type == SourceType.X:
        return f"@{item.author}" if item.author else ""
    if item.source_type == SourceType.RSS:
        return meta.get("feed_title", "") or meta.get("feed_url", "")
    return ""
