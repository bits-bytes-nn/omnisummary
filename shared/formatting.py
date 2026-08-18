from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, date, datetime
from urllib.parse import urlparse

from .constants import SourceType
from .logger import logger
from .models import CollectedItem

# A literal Unicode emoji, NOT a Slack `:shortcode:`. Slack renders both, but Threads renders no
# shortcodes — and the Threads renderer strips Slack markup characters, so `:arrow_forward:` lost
# its underscore and published as a bare ":arrowforward:" on 2026-08-18.
YOUTUBE_VIEWS_EMOJI = "▶️"
RSS_NAME_DELIMITERS = (" - ", " — ")


def format_alarm(
    *,
    event: str,
    status: str,
    fields: dict[str, str],
    project: str = "omnisummary",
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """Build a ``(subject, message)`` pair in the project family's unified alarm
    format, shared verbatim across tech-digest/paper-bridge/scholar-lens:

        Subject: [<project>] <event> — <STATUS>

        <event> <STATUS>

        Key:   Value

        — 2026-06-10 04:12:00 UTC

    ``status`` is a short uppercase state (``FAILED``/``ALERT``). ``fields`` is an
    ordered mapping; single-line values render as an aligned ``Key: Value`` block,
    multi-line values render under their own ``Key:`` header. Omit a row by leaving
    it out of the dict.
    """
    ts = (timestamp or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[{project}] {event} — {status}"

    inline = {k: v for k, v in fields.items() if "\n" not in v}
    block = {k: v for k, v in fields.items() if "\n" in v}

    lines = [f"{event} {status}", ""]
    if inline:
        width = max(len(k) for k in inline)
        lines += [f"{k + ':':<{width + 1}} {v}" for k, v in inline.items()]
    for k, v in block.items():
        lines += ["", f"{k}:", v.strip("\n")]
    lines.append("")
    lines.append(f"— {ts} UTC")

    return subject, "\n".join(lines)


def agi_countdown_intro(date_str: str, template: str, today: date, after_template: str = "") -> str:
    """The tongue-in-cheek AGI-countdown intro, computed in code (never the LLM) from a fixed D-day
    so it's accurate and ticks daily. Applied at POST time so it lands on every channel and run.
    Before the D-day: counts DOWN via `template` ({days} = days remaining). On/after the D-day:
    counts UP via `after_template` ({days} = days since), a self-aware nod that the prediction blew
    past. Returns "" when disabled/malformed, or after the D-day if no after_template is set."""
    if not date_str or not template:
        return ""
    try:
        target = date.fromisoformat(date_str)
    except ValueError:
        return ""
    days = (target - today).days
    # The templates are operator-editable config strings. A typo'd placeholder ({day}) or stray
    # brace would otherwise raise KeyError/ValueError mid-generation — AFTER the expensive collect/
    # rank/LLM work — and kill the whole run. Degrade to no intro instead.
    try:
        if days > 0:
            return template.format(days=days)
        if after_template:
            return after_template.format(days=-days)
    except (KeyError, IndexError, ValueError) as e:
        logger.warning("Malformed agi_countdown template (%s); skipping intro", e)
    return ""


COUNTDOWN_SUFFIX_SEPARATOR = "\n\n"


def place_countdown_intro(lead: str, intro: str, position: str = "prefix") -> str:
    """Attach the countdown gag to the lead at the configured end, verbatim and idempotently.

    "suffix" puts it on its own closing line so the lead's FIRST line is the day's actual angle
    (40 consecutive Threads roots opened with the identical countdown sentence). Returns the lead
    unchanged when there is no intro, or when it is already attached at that end."""
    if not intro or not lead:
        return lead
    if position == "suffix":
        tail = intro.strip()
        body = lead.rstrip()
        if not tail or body.endswith(tail):
            return lead
        return f"{body}{COUNTDOWN_SUFFIX_SEPARATOR}{tail}"
    return lead if lead.startswith(intro) else intro + lead


def editorial_lead(lead: str, intro: str) -> str:
    """The lead with the AGI-countdown gag removed from EITHER end, leaving only the editorial
    angle. Callers that reason about the ANGLE (recent-leads novelty, the visual's take) want this
    rather than the raw lead, one end of which is the same fixed daily template every single day.
    Both ends are handled because the gag's position is configurable, and a stored lead can predate
    a change to that setting."""
    if not intro:
        return lead
    if lead.startswith(intro):
        return lead[len(intro) :].lstrip()
    tail = intro.strip()
    body = lead.rstrip()
    if tail and body.endswith(tail):
        return body[: -len(tail)].rstrip()
    return lead


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
    truncate: Callable[[str, int], str],
    text_label: str = "Text",
) -> str:
    """Render a CollectedItem as a labelled `=== Item N ===` block for LLM input.

    `fields` are the leading "Label: value" lines in the caller's chosen order
    (Title, Source, Author, Score, ...); the body text (truncated to `max_tokens` via the
    caller's `truncate` callable — bound to the Bedrock CountTokens-based truncator) is appended
    last under `text_label`. Shared so the ranker, digest generator, and agent stay in lockstep.
    """
    snippet = truncate(item.text, max_tokens)
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
    """Per-origin diversity key (e.g. a single channel/subreddit/feed/account/site).

    Web-search items carry no channel/feed metadata, so their host stands in as the origin —
    without it they returned None and slipped past `max_per_origin` entirely, letting one outlet
    take several of the digest's slots. The host is normalized the same way the RSS feed-name
    helper does it (`netloc` minus a `www.` prefix), deliberately NOT a registrable-domain /
    public-suffix heuristic: no new dependency, and subdomains stay distinct origins.
    """
    meta = item.metadata
    if item.source_type == SourceType.YOUTUBE:
        return meta.get("channel_url")
    if item.source_type == SourceType.REDDIT:
        return meta.get("subreddit")
    if item.source_type == SourceType.RSS:
        return meta.get("feed_url")
    if item.source_type == SourceType.X:
        return item.author
    if item.source_type == SourceType.WEB:
        return urlparse(item.url).netloc.removeprefix("www.") or None
    return None


def format_origin_label(item: CollectedItem) -> str:
    """Plain-text origin label fed to the ranking prompt (no Slack markup).

    Web-search items carried NO origin line, so the ranking prompt was asked to judge "Source
    Authority" for them with the outlet withheld — a press release on a content farm and a report
    from a wire service looked identical. Their host stands in as the origin, derived exactly the
    way resolve_origin_key does it (`netloc` minus a `www.` prefix): no domain/authority table, no
    public-suffix logic, no new dependency.
    """
    meta = item.metadata
    if item.source_type == SourceType.REDDIT:
        return f"r/{meta.get('subreddit', '')}" if meta.get("subreddit") else ""
    if item.source_type == SourceType.YOUTUBE:
        return meta.get("channel_url", "")
    if item.source_type == SourceType.X:
        return f"@{item.author}" if item.author else ""
    if item.source_type == SourceType.RSS:
        return meta.get("feed_title", "") or meta.get("feed_url", "")
    if item.source_type == SourceType.WEB:
        return urlparse(item.url).netloc.removeprefix("www.")
    return ""
