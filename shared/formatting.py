from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import NamedTuple
from urllib.parse import urlparse

from .constants import THREADS_POST_SEPARATOR, SourceType
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
    stage: str = "",
    correlation_id: str = "",
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """Build a ``(subject, message)`` pair in the project family's unified alarm
    format, shared verbatim across tech-digest/paper-bridge/scholar-lens:

        Subject: [<project>/<stage>] <event> — <STATUS>

        <event> <STATUS>

        Key:   Value

        — 2026-06-10 04:12:00 UTC

    ``status`` is a short uppercase state (``FAILED``/``ALERT``). ``fields`` is an
    ordered mapping; single-line values render as an aligned ``Key: Value`` block,
    multi-line values render under their own ``Key:`` header. Omit a row by leaving
    it out of the dict.

    ``project``/``stage`` come from the caller's environment: with the default project and no stage,
    a dev-stage and a prod-stage alert were byte-identical, so a second deployment alerted under the
    wrong name. ``correlation_id`` is appended as the last field — it is set on every handler
    invocation and appeared in no alert, so an operator could not get from the mail to the matching
    JSON log lines.
    """
    ts = (timestamp or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[{project}/{stage}] {event} — {status}" if stage else f"[{project}] {event} — {status}"
    if correlation_id:
        fields = {**fields, "Correlation id": correlation_id}

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


def item_netloc(item: CollectedItem) -> str:
    """The item's host, normalized the way clean_rss_feed_name does it (`netloc` minus a `www.`
    prefix). Deliberately NOT a registrable-domain / public-suffix heuristic: no new dependency, and
    subdomains stay distinct origins.

    This is the LAST-RESORT identity for an item whose source-specific metadata is missing, and it is
    live-fired: DOMAIN_TO_SOURCE relabels a web-search hit or a pinned x.com URL as SourceType.X,
    whose origin is `item.author` — which those items never carry. They therefore had origin key
    None, escaped max_per_origin entirely, were skipped by pin-origin seeding, and reached the
    ranking prompt with no Origin line at all: exactly the gap the WEB branch was added to close."""
    return urlparse(item.url).netloc.removeprefix("www.")


class SourceDescriptor(NamedTuple):
    """Everything the pipeline needs to know about ONE SourceType: the per-origin diversity key, the
    plain-text origin label the ranking prompt reads, and the display tag + metrics the renderers
    show.

    ONE table instead of three if/elif chains over the same five SourceTypes (resolve_origin_key,
    format_origin_label, the digest generator's _source_tag_and_metrics), each of which ended in its
    own silent fall-through default."""

    origin_key: Callable[[CollectedItem], str]
    label: Callable[[CollectedItem], str]
    tag: Callable[[CollectedItem], str]
    metrics: Callable[[CollectedItem], str]


def _no_metrics(item: CollectedItem) -> str:
    """Most sources carry no engagement figures the renderers can show. Reddit is collected through
    the public .rss feed, which drops score/num_comments, so only YouTube has any."""
    return ""


def _youtube_metrics(item: CollectedItem) -> str:
    views = item.metadata.get("view_count")
    return f"{YOUTUBE_VIEWS_EMOJI} {views:,}" if views else ""


def _rss_name(item: CollectedItem) -> str:
    return clean_rss_feed_name(item.metadata.get("feed_title", ""), item.metadata.get("feed_url", ""))


SOURCE_DESCRIPTORS: dict[SourceType, SourceDescriptor] = {
    SourceType.REDDIT: SourceDescriptor(
        origin_key=lambda item: item.metadata.get("subreddit", ""),
        label=lambda item: f"r/{item.metadata['subreddit']}" if item.metadata.get("subreddit") else "",
        tag=lambda item: f"`r/{item.metadata['subreddit']}`" if item.metadata.get("subreddit") else "",
        metrics=_no_metrics,
    ),
    SourceType.RSS: SourceDescriptor(
        origin_key=lambda item: item.metadata.get("feed_url", ""),
        label=lambda item: item.metadata.get("feed_title", "") or item.metadata.get("feed_url", ""),
        tag=lambda item: f"`{name}`" if (name := _rss_name(item)) else "",
        metrics=_no_metrics,
    ),
    SourceType.WEB: SourceDescriptor(
        origin_key=item_netloc,
        label=item_netloc,
        tag=lambda item: f"`{host}`" if (host := item_netloc(item)) else "",
        metrics=_no_metrics,
    ),
    SourceType.X: SourceDescriptor(
        origin_key=lambda item: item.author or "",
        label=lambda item: f"@{item.author}" if item.author else "",
        tag=lambda item: f"`@{item.author}`" if item.author else "",
        metrics=_no_metrics,
    ),
    SourceType.YOUTUBE: SourceDescriptor(
        origin_key=lambda item: item.metadata.get("channel_url", ""),
        label=lambda item: item.metadata.get("channel_url", ""),
        # The CHANNEL, like every other source's tag names the specific origin (a subreddit, an
        # account, a publication, a host). This was the literal "YouTube" — the platform — so the
        # reader learned nothing, and the editor compensated by appending the speaker to the item
        # TITLE instead. `author` is the channelTitle the collector already stores; the RSS-fallback
        # path has no author, so the platform name remains the floor.
        tag=lambda item: f"`{item.author}`" if item.author else "`YouTube`",
        metrics=_youtube_metrics,
    ),
}

# A SourceType with no descriptor would silently take every fall-through default at once: no origin
# key (so no max_per_origin, no pin seeding), no Origin line in the ranking prompt, and no source tag
# in the digest. Same shape as PipelineConfig's source_slots validator — fail at import, not at 19:00.
assert set(SOURCE_DESCRIPTORS) == set(
    SourceType
), f"SOURCE_DESCRIPTORS is missing an entry for {sorted(s.value for s in SourceType if s not in SOURCE_DESCRIPTORS)}"


def resolve_origin_key(item: CollectedItem) -> str | None:
    """Per-origin diversity key (a single channel/subreddit/feed/account/site), falling back to the
    item's host when the source's own metadata is absent.

    The fallback is what makes the key TOTAL: an origin-less item slipped past `max_per_origin`
    entirely, letting one outlet — or one author-less scrape relabelled as SourceType.X — take
    several of the digest's slots."""
    return SOURCE_DESCRIPTORS[item.source_type].origin_key(item) or item_netloc(item) or None


def format_origin_label(item: CollectedItem) -> str:
    """Plain-text origin label fed to the ranking prompt (no Slack markup), falling back to the host.

    An item with no origin line asked the prompt to judge "Source Authority" with the outlet
    withheld — a press release on a content farm and a wire-service report looked identical."""
    return SOURCE_DESCRIPTORS[item.source_type].label(item) or item_netloc(item)


def source_tag_and_metrics(item: CollectedItem) -> tuple[str, str]:
    """(source_tag, metrics) for an item: a backtick-wrapped source label and a ' · '-joined emoji
    metric string. Code owns this — the LLM never writes source markup. The tag falls back to the
    host for the same reason the origin key does: an item whose source metadata is missing still has
    to tell the reader where it came from."""
    descriptor = SOURCE_DESCRIPTORS[item.source_type]
    tag = descriptor.tag(item)
    if not tag and (host := item_netloc(item)):
        tag = f"`{host}`"
    return tag, descriptor.metrics(item)


# Sentence-ending boundaries: Korean '다.' plus the usual terminators. Splitting AFTER a boundary
# (never on a fixed stride) is what lets a post be trimmed at a clean sentence.
_SENTENCE_END = ("다.", "다!", "다?", ". ", "。", "! ", "? ", "…")
# A bare http(s) URL run. Public so the renderers can protect/extract citation URLs with the same
# pattern strip_slack_mrkdwn uses, instead of re-declaring it.
URL_RE = re.compile(r"https?://\S+")
# Punctuation a URL can pick up from the prose around it (a closing paren, a sentence period, the
# '>' of a Slack link). Trimmed from the right by both the URL extractor and the citation normalizer.
_URL_TRAILING_PUNCT = "').,;:\"]>"


def extract_urls(text: str) -> list[str]:
    """Every http(s) URL in `text`, in order of appearance, with Slack mrkdwn unwrapped first and
    trailing prose punctuation trimmed.

    URL_RE's `\\S+` does not stop at '|', so a well-formed Slack link `<https://ex.com/a|Label>`
    matches as `https://ex.com/a|Label` and can never equal the bare URL a tool returned — which
    made the citation guard refuse every correctly formatted Slack report. strip_slack_mrkdwn is
    the converter that already knows this markup, so extraction goes through it rather than
    growing a second URL pattern that has to be kept in sync with it."""
    return [url.strip().rstrip(_URL_TRAILING_PUNCT) for url in URL_RE.findall(strip_slack_mrkdwn(text))]


def normalize_citation_url(url: str) -> str:
    """Loose identity for comparing a URL a report CITES against one a tool actually returned.

    Scheme and a `www.` prefix are dropped, trailing sentence punctuation and a trailing slash are
    trimmed, and the fragment is discarded. Deliberately LENIENT: the comparison exists to refuse a
    FABRICATED citation, so it must never reject a real one the model rewrote from http to https or
    quoted inside parentheses. The query string is kept — it is what distinguishes one video or
    search result from another."""
    trimmed = url.strip().rstrip(_URL_TRAILING_PUNCT)
    parsed = urlparse(trimmed if "//" in trimmed else f"//{trimmed}")
    host = parsed.netloc.removeprefix("www.").lower()
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without losing characters, breaking only AFTER a
    sentence-ending boundary (Korean '다.' / '?' / '!' or '. '). Whitespace-only tails
    are dropped. Used so a post is trimmed at a clean sentence, never mid-word.

    Channel-agnostic on purpose: the pipeline's prose budget and the Threads renderer both need it,
    and the pipeline used to reach into the renderer's private helper to get it."""
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        matched = next((e for e in _SENTENCE_END if text.startswith(e, i)), None)
        if matched:
            end = i + len(matched)
            out.append(text[start:end].strip())
            start = end
            i = end
        else:
            i += 1
    if text[start:].strip():
        out.append(text[start:].strip())
    return [s for s in out if s]


def truncate_at_word(text: str, max_len: int) -> str:
    """Trim text to <=max_len on a whitespace boundary (never mid-word); if there's no space
    in range, fall back to a hard character cut. Used only when prose has no sentence boundary."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    window = text[:max_len]
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip()


def strip_slack_mrkdwn(text: str) -> str:
    """Convert Slack mrkdwn to plain text (for channels that render no markup, and for measuring
    how long a string really is): turn <url|label> into 'label (url)', drop *bold*/_italic_/`code`
    markers, and remove leading bullet/heading glyphs. URLs are protected from the marker strip —
    they legitimately contain '_'/'*' (arxiv, github, query params), so stripping those characters
    globally would silently break the links. Whitespace structure is preserved."""
    text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2 (\1)", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Stash URLs so the [*_`] strip below can't corrupt them, then restore verbatim.
    urls: list[str] = []

    def _stash(match: re.Match) -> str:
        urls.append(match.group(0))
        return f"\x00{len(urls) - 1}\x00"

    text = URL_RE.sub(_stash, text)
    # Strip leading bullet/heading glyphs FIRST (so a "* 항목" or "- 항목" bullet is removed as a
    # unit), THEN drop inline *bold*/_italic_/`code` markers.
    out_lines = [re.sub(r"^\s*(?:[-*•]\s+|#{1,6}\s+)", "", line) for line in text.split("\n")]
    text = re.sub(r"[*_`]", "", "\n".join(out_lines))
    return re.sub(r"\x00(\d+)\x00", lambda m: urls[int(m.group(1))], text)


def threads_meta_line(source_tag: str, metrics: str) -> str:
    """One item's provenance line exactly as a Threads post shows it: the same composition Slack's
    context block uses, with the markup stripped (`source_tag` is stored backtick-wrapped for Slack
    mrkdwn and Threads renders none, so the backticks would show up literally).

    Shared by the renderer that writes the line and the pipeline that budgets for its length, so the
    budget can never be derived from a string the post does not actually carry."""
    return strip_slack_mrkdwn(" · ".join(p for p in (source_tag, metrics) if p)).strip()


def threads_item_overhead_chars(meta: str, url: str) -> int:
    """Characters ONE item's Threads post spends on the parts CODE owns: the source line, the URL,
    and the blank-line separators between title / source / body / implication / URL. Everything left
    over is what the editor may write (title + body + implication), so the prose budget it is told
    about is derived from this — not from a hand-estimated "~120 chars in practice".

    The source line rides in parentheses ON the title line, so it costs its own length plus the
    " (" and ")" around it but adds NO separator — the post is `title (meta)` | body | implication |
    url. The implication is its own block, hence one separator more than there are separate
    code-owned blocks (title | body | implication is 2 separators even with no meta and no URL). An
    item with no implication is charged that separator anyway — a slightly smaller budget, never a
    too-large one.

    Lives beside the other post-length primitives (not in the renderer) so the pipeline can derive
    the editor's budget without importing an output channel."""
    blocks = [p for p in (url.strip(),) if p]
    inline_meta = len(meta.strip()) + len(" ()") if meta.strip() else 0
    return inline_meta + sum(len(p) for p in blocks) + len(THREADS_POST_SEPARATOR) * (len(blocks) + 2)
