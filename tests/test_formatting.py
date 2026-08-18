from shared import format_collected_item
from shared.constants import SourceType
from shared.formatting import format_origin_label, resolve_origin_key
from shared.models import CollectedItem


def _item(text: str = "body text") -> CollectedItem:
    return CollectedItem(
        item_id="id1",
        source_type=SourceType.RSS,
        title="A Title",
        url="http://example.com",
        text=text,
        author="Alice",
    )


# The caller supplies the truncator (bound to the Bedrock CountTokens-based one in prod);
# tests use a word-cap stand-in so they stay offline.
def _truncate(text: str, max_tokens: int) -> str:
    words = text.split()
    return " ".join(words[:max_tokens]) if len(words) > max_tokens else text


class TestFormatCollectedItem:
    def test_renders_header_fields_and_text(self):
        out = format_collected_item(
            _item(),
            index=2,
            max_tokens=1000,
            fields=[("Title", "A Title"), ("Source", "rss")],
            truncate=_truncate,
        )
        assert out.startswith("=== Item 2 ===\n")
        assert "Title: A Title\n" in out
        assert "Source: rss\n" in out
        assert out.endswith("Text:\nbody text\n")

    def test_custom_text_label(self):
        out = format_collected_item(
            _item(), index=1, max_tokens=1000, fields=[], truncate=_truncate, text_label="Content"
        )
        assert "Content:\nbody text" in out

    def test_truncates_body_with_supplied_callable(self):
        out = format_collected_item(_item("word " * 500), index=1, max_tokens=10, fields=[], truncate=_truncate)
        body = out.split("Text:\n", 1)[1]
        assert len(body.split()) <= 10


def _sourced(source: SourceType, *, url: str = "http://example.com/a", **fields) -> CollectedItem:
    return CollectedItem(
        item_id="id",
        source_type=source,
        title="t",
        url=url,
        author=fields.pop("author", None),
        metadata=fields,
    )


class TestResolveOriginKey:
    def test_web_origin_is_the_host(self):
        # Web items carry no channel/feed metadata; without a host key they slipped past the
        # per-origin diversity cap entirely.
        assert resolve_origin_key(_sourced(SourceType.WEB, url="https://techcrunch.com/a/b")) == "techcrunch.com"

    def test_web_origin_drops_www_prefix(self):
        # Same normalization the RSS feed-name helper uses, so www/non-www are ONE origin.
        assert resolve_origin_key(_sourced(SourceType.WEB, url="https://www.wired.com/x")) == "wired.com"

    def test_web_subdomains_stay_distinct(self):
        # Deliberately NOT a registrable-domain (public-suffix) heuristic.
        assert resolve_origin_key(_sourced(SourceType.WEB, url="https://ai.googleblog.com/p")) == "ai.googleblog.com"
        assert resolve_origin_key(_sourced(SourceType.WEB, url="https://blog.google/p")) == "blog.google"

    def test_web_origin_none_without_host(self):
        assert resolve_origin_key(_sourced(SourceType.WEB, url="notaurl")) is None

    def test_other_sources_unchanged(self):
        assert resolve_origin_key(_sourced(SourceType.YOUTUBE, channel_url="c")) == "c"
        assert resolve_origin_key(_sourced(SourceType.REDDIT, subreddit="LocalLLaMA")) == "LocalLLaMA"
        assert resolve_origin_key(_sourced(SourceType.RSS, feed_url="f")) == "f"
        assert resolve_origin_key(_sourced(SourceType.X, author="karpathy")) == "karpathy"
        assert resolve_origin_key(_sourced(SourceType.RSS)) is None  # no feed metadata -> no origin


class TestFormatOriginLabel:
    """The ranking prompt judges "Source Authority" from this line. Web-search items had none, so
    the outlet was withheld from the one criterion that needs it."""

    def test_web_label_is_the_host(self):
        assert format_origin_label(_sourced(SourceType.WEB, url="https://techcrunch.com/a/b")) == "techcrunch.com"

    def test_web_label_drops_www_and_keeps_subdomains(self):
        # Mirrors resolve_origin_key exactly: netloc minus 'www.', no public-suffix folding.
        assert format_origin_label(_sourced(SourceType.WEB, url="https://www.wired.com/x")) == "wired.com"
        assert format_origin_label(_sourced(SourceType.WEB, url="https://ai.googleblog.com/p")) == "ai.googleblog.com"

    def test_web_label_empty_without_a_host(self):
        assert format_origin_label(_sourced(SourceType.WEB, url="notaurl")) == ""

    def test_other_sources_unchanged(self):
        assert format_origin_label(_sourced(SourceType.REDDIT, subreddit="LocalLLaMA")) == "r/LocalLLaMA"
        assert format_origin_label(_sourced(SourceType.YOUTUBE, channel_url="c")) == "c"
        assert format_origin_label(_sourced(SourceType.X, author="karpathy")) == "@karpathy"
        assert format_origin_label(_sourced(SourceType.RSS, feed_title="The Verge")) == "The Verge"
