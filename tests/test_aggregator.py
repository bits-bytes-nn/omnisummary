from pipeline.aggregator import ContentAggregator
from shared.constants import SourceType
from shared.models import CollectedItem


def _item(item_id: str = "", url: str = "http://example.com", title: str = "T", **kwargs) -> CollectedItem:
    return CollectedItem(item_id=item_id, source_type=SourceType.REDDIT, title=title, url=url, **kwargs)


class TestContentAggregator:
    def test_preserves_original_item_id(self):
        items = [_item(item_id="reddit_abc123", url="http://a.com")]
        result = ContentAggregator().aggregate(items)
        assert result[0].item_id == "reddit_abc123"

    def test_deduplicates_by_url(self):
        items = [
            _item(item_id="id1", url="http://a.com", title="First"),
            _item(item_id="id2", url="http://a.com", title="Duplicate"),
            _item(item_id="id3", url="http://b.com", title="Second"),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 2
        urls = {item.url for item in result}
        assert urls == {"http://a.com", "http://b.com"}

    def test_excludes_recently_published_urls(self):
        # Cross-day dedup: an article already published on a recent day is dropped, even
        # across http/https + trailing-slash variants (exclude set holds normalized URLs).
        from pipeline.aggregator import normalize_url

        items = [
            _item(item_id="id1", url="https://a.com/post/", title="Already published"),
            _item(item_id="id2", url="http://b.com", title="Fresh"),
        ]
        exclude = {normalize_url("http://a.com/post")}
        result = ContentAggregator().aggregate(items, exclude_urls=exclude)
        assert {it.url for it in result} == {"http://b.com"}

    def test_pinned_item_bypasses_cross_day_dedup(self):
        # A user-pinned URL must survive even if it's in the recently-published exclude set.
        from pipeline.aggregator import normalize_url

        items = [
            _item(item_id="id1", url="https://a.com/post/", title="Pinned", metadata={"pinned": True}),
            _item(item_id="id2", url="http://b.com", title="Fresh"),
        ]
        exclude = {normalize_url("http://a.com/post")}
        result = ContentAggregator().aggregate(items, exclude_urls=exclude)
        assert {it.url for it in result} == {"https://a.com/post/", "http://b.com"}

    def test_dedup_normalizes_url_variants(self):
        # trailing slash, scheme, www, tracking params, fragment, query order
        # should all collapse to one item.
        items = [
            _item(item_id="id1", url="https://www.a.com/post", title="A1"),
            _item(item_id="id2", url="http://a.com/post/", title="A2"),
            _item(item_id="id3", url="https://a.com/post?utm_source=x&fbclid=y", title="A3"),
            _item(item_id="id4", url="https://a.com/post#section", title="A4"),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1
        assert result[0].item_id == "id1"  # first wins

    def test_dedup_preserves_meaningful_query_params(self):
        items = [
            _item(item_id="id1", url="https://a.com/watch?v=abc", title="V1"),
            _item(item_id="id2", url="https://a.com/watch?v=def", title="V2"),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 2

    def test_merges_metadata_on_duplicate(self):
        items = [
            _item(url="http://a.com", metadata={"key1": "val1"}),
            _item(url="http://a.com", metadata={"key2": "val2"}),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1
        assert result[0].metadata["key1"] == "val1"
        assert result[0].metadata["key2"] == "val2"

    def test_normalizes_metadata(self):
        from datetime import datetime

        items = [_item(url="http://a.com", metadata={"dt": datetime(2024, 1, 1)})]
        result = ContentAggregator().aggregate(items)
        assert isinstance(result[0].metadata["dt"], str)

    def test_empty_input(self):
        result = ContentAggregator().aggregate([])
        assert result == []

    def test_multiple_unique_items(self):
        items = [_item(item_id=f"id{i}", url=f"http://{i}.com", title=f"Title {i}") for i in range(5)]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 5

    def test_drops_items_missing_url_or_title(self):
        # Empty-url items all normalize to the same "" key and would dedup against each other,
        # silently swallowing siblings; empty-title items can't render. Both are dropped up front.
        items = [
            _item(item_id="ok", url="http://real.com", title="Real"),
            _item(item_id="nourl", url="", title="Has title but no URL"),
            _item(item_id="notitle", url="http://x.com", title="   "),
            _item(item_id="ok2", url="http://other.com", title="Another"),
        ]
        result = ContentAggregator().aggregate(items)
        urls = {i.url for i in result}
        assert urls == {"http://real.com", "http://other.com"}

    def test_deduplicates_by_title(self):
        items = [
            _item(url="http://a.com", title="LiteLLM 공급망 공격에 대한 분 단위 대응 기록"),
            _item(url="http://b.com", title="LiteLLM 공급망 공격에 대한 분 단위 대응 기록"),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1

    def test_title_dedup_case_insensitive(self):
        items = [
            _item(url="http://a.com", title="Hello World"),
            _item(url="http://b.com", title="hello world"),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1

    def test_title_dedup_ignores_punctuation(self):
        items = [
            _item(url="http://a.com", title="What's New in AI?"),
            _item(url="http://b.com", title="Whats New in AI"),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1

    def test_pinned_item_bypasses_title_dedup(self):
        # A pinned item sharing a normalized title with an earlier story must NOT be dropped by
        # title dedup — otherwise the --pin-url guarantee dies before the ranker's pin-recovery
        # (which only sees the post-aggregation list) can restore it.
        items = [
            _item(url="http://a.com", title="Same Title"),
            _item(url="http://b.com", title="Same Title", metadata={"pinned": True}),
        ]
        result = ContentAggregator().aggregate(items)
        urls = {it.url for it in result}
        assert "http://b.com" in urls  # pinned survived title dedup

    def test_title_dedup_keeps_survivor_metadata_fills_only_missing(self):
        # The kept (first) item's own metadata must NOT be overwritten by a later duplicate;
        # the duplicate only fills keys the survivor lacks.
        items = [
            _item(url="http://a.com", title="Same Title", metadata={"src": "rss"}),
            _item(url="http://b.com", title="Same Title", metadata={"src": "web", "extra": "x"}),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1
        assert result[0].metadata["src"] == "rss"  # survivor's own value preserved
        assert result[0].metadata["extra"] == "x"  # missing key filled from the duplicate
