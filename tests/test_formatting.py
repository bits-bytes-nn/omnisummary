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

    def test_missing_source_metadata_falls_back_to_the_host(self):
        # An origin-less item escaped max_per_origin entirely, was skipped by pin-origin seeding, and
        # reached the ranking prompt with no Origin line. Live-fired: DOMAIN_TO_SOURCE relabels a
        # web-search hit or a pinned x.com URL as SourceType.X, whose origin is item.author — which
        # those items never carry.
        assert resolve_origin_key(_sourced(SourceType.X, url="https://x.com/a/status/1")) == "x.com"
        assert resolve_origin_key(_sourced(SourceType.RSS, url="https://blog.example/p")) == "blog.example"

    def test_no_origin_only_when_there_is_no_host_either(self):
        assert resolve_origin_key(_sourced(SourceType.RSS, url="notaurl")) is None


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

    def test_a_missing_label_falls_back_to_the_host(self):
        # Otherwise the prompt judges "Source Authority" with the outlet withheld.
        assert format_origin_label(_sourced(SourceType.X, url="https://x.com/a/status/1")) == "x.com"


class TestSourceDescriptorRegistry:
    """One table keyed by SourceType replaces three if/elif chains over the same five types
    (resolve_origin_key, format_origin_label, the digest generator's tag/metrics), each of which ended
    in its own silent fall-through default — so only one of them was ever fixed at a time."""

    def test_every_source_type_has_a_descriptor(self):
        from shared.formatting import SOURCE_DESCRIPTORS

        assert set(SOURCE_DESCRIPTORS) == set(SourceType)

    def test_the_tag_falls_back_to_the_host(self):
        from shared.formatting import source_tag_and_metrics

        tag, metrics = source_tag_and_metrics(_sourced(SourceType.X, url="https://x.com/a/status/1"))
        assert tag == "`x.com`"
        assert metrics == ""

    def test_only_youtube_carries_metrics(self):
        from shared.formatting import source_tag_and_metrics

        _, views = source_tag_and_metrics(_sourced(SourceType.YOUTUBE, view_count=12345))
        assert "12,345" in views
        # Reddit comes through the public .rss feed, which drops score/num_comments.
        assert source_tag_and_metrics(_sourced(SourceType.REDDIT, subreddit="LocalLLaMA"))[1] == ""


class TestPublicPackageBoundaries:
    """Two boundary violations kept biting refactors: `shared.research` and `shared` exported
    UNDERSCORE names as their cross-package contract, and the pipeline's prose budget imported an
    output channel's private helpers. Both are now public names in shared/."""

    def test_shared_exports_no_underscore_names(self):
        import shared
        import shared.research

        assert not [name for name in shared.__all__ if name.startswith("_")]
        assert not [name for name in shared.research.__all__ if name.startswith("_")]

    def test_the_digest_generator_does_not_import_from_the_output_layer(self):
        # Its Threads-aware prose budget used to depend on output.renderers' PRIVATE helpers, so
        # renaming a renderer internal broke the pipeline. (daily_visual legitimately imports the
        # delivery handlers — it is the publish path; the generator only computes a budget.)
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "pipeline" / "digest_generator.py").read_text()
        assert "from output" not in source and "import output" not in source

    def test_the_threads_post_cap_has_one_definition(self):
        from output import renderers, threads_handler
        from pipeline import digest_generator
        from shared.constants import THREADS_MAX_POST_CHARS

        assert renderers.THREADS_MAX_POST_CHARS is THREADS_MAX_POST_CHARS
        assert threads_handler.THREADS_MAX_POST_CHARS is THREADS_MAX_POST_CHARS
        assert digest_generator.THREADS_MAX_POST_CHARS is THREADS_MAX_POST_CHARS


class TestExtractUrls:
    """URL_RE's `\\S+` does not stop at '|', so the raw pattern read a Slack `<url|label>` link as
    `url|label` and the citation guard refused every well-formed Slack report. extract_urls unwraps
    the markup first, through the converter that already knows it."""

    def test_a_slack_link_with_a_label_yields_the_bare_url(self):
        from shared.formatting import URL_RE, extract_urls

        report = "근거: <https://real.example/a|Real Example>"
        assert extract_urls(report) == ["https://real.example/a"]
        assert URL_RE.findall(report) == ["https://real.example/a|Real"]

    def test_a_multi_word_label_does_not_leak_into_the_url(self):
        from shared.formatting import extract_urls

        assert extract_urls("<https://ex.example/p?q=1|한 줄 라벨>") == ["https://ex.example/p?q=1"]

    def test_an_angle_wrapped_url_yields_the_bare_url(self):
        from shared.formatting import extract_urls

        assert extract_urls("출처 <https://ex.example/p>") == ["https://ex.example/p"]

    def test_prose_punctuation_is_trimmed_and_order_preserved(self):
        from shared.formatting import extract_urls

        text = "본문 (https://a.example/1). 그리고 https://b.example/2, 끝."
        assert extract_urls(text) == ["https://a.example/1", "https://b.example/2"]

    def test_a_slack_link_and_the_bare_url_normalize_to_the_same_identity(self):
        from shared.formatting import extract_urls, normalize_citation_url

        surfaced = normalize_citation_url("http://www.real.example/a/")
        (cited,) = extract_urls("<https://real.example/a|Real Example>")
        assert normalize_citation_url(cited) == surfaced


class TestFormatAlarm:
    """The subject is what an operator sees first, and it used to carry a hardcoded project with no
    stage — so a dev alert and a prod alert were byte-identical."""

    @staticmethod
    def _alarm(**kwargs):
        from shared.formatting import format_alarm

        base = {"event": "Source Health", "status": "ALERT", "fields": {"Failed sources": "reddit"}}
        base.update(kwargs)
        return format_alarm(**base)

    def test_subject_defaults_to_the_family_format(self):
        subject, _ = self._alarm()
        assert subject == "[omnisummary] Source Health — ALERT"

    def test_stage_is_named_in_the_subject(self):
        subject, _ = self._alarm(project="omnisummary", stage="prod")
        assert subject == "[omnisummary/prod] Source Health — ALERT"

    def test_correlation_id_is_the_last_field(self):
        _, message = self._alarm(correlation_id="abc123def456")
        lines = [line for line in message.splitlines() if line.strip()]
        assert lines[-2].startswith("Correlation id:")  # the timestamp footer is last
        assert "abc123def456" in message

    def test_no_correlation_row_when_there_is_no_id(self):
        _, message = self._alarm()
        assert "Correlation id" not in message

    def test_multi_line_fields_still_render_as_blocks_after_the_id(self):
        _, message = self._alarm(fields={"Report": "line1\nline2"}, correlation_id="cid")
        assert "Report:" in message and "line1" in message
        assert "Correlation id: cid" in message
