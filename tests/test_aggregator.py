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

    def test_title_dedup_merges_metadata(self):
        items = [
            _item(url="http://a.com", title="Same Title", metadata={"src": "rss"}),
            _item(url="http://b.com", title="Same Title", metadata={"src": "web"}),
        ]
        result = ContentAggregator().aggregate(items)
        assert len(result) == 1
        assert result[0].metadata["src"] == "web"
