from pipeline.digest_generator import DigestGenerator
from shared.constants import SourceType
from shared.models import CollectedItem


def _item(source_type=SourceType.REDDIT, metadata=None, author=None):
    return CollectedItem(
        item_id="test",
        source_type=source_type,
        title="Test",
        url="http://test.com",
        metadata=metadata or {},
        author=author,
    )


class TestFormatSourceDetail:
    def test_reddit(self):
        # .rss feed carries no score/num_comments — only the subreddit tag is rendered.
        item = _item(SourceType.REDDIT, metadata={"subreddit": "LocalLLaMA"})
        result = DigestGenerator._format_source_detail(item)
        assert result == "`r/LocalLLaMA`"

    def test_youtube(self):
        item = _item(SourceType.YOUTUBE, metadata={"view_count": 12345})
        result = DigestGenerator._format_source_detail(item)
        assert "`YouTube`" in result
        assert ":arrow_forward: 12,345" in result

    def test_x_with_author(self):
        item = _item(SourceType.X, author="karpathy")
        result = DigestGenerator._format_source_detail(item)
        assert "`@karpathy`" in result

    def test_rss_with_feed_title(self):
        item = _item(SourceType.RSS, metadata={"feed_title": "GeekNews - 개발/기술/스타트업 뉴스 서비스"})
        result = DigestGenerator._format_source_detail(item)
        assert "`GeekNews`" in result
        assert "개발" not in result

    def test_web(self):
        item = CollectedItem(
            item_id="test",
            source_type=SourceType.WEB,
            title="Test",
            url="http://arxiv.org/abs/1234",
        )
        result = DigestGenerator._format_source_detail(item)
        assert "`arxiv.org`" in result

    def test_reddit_no_engagement(self):
        item = _item(SourceType.REDDIT, metadata={"subreddit": "MachineLearning"})
        result = DigestGenerator._format_source_detail(item)
        assert "`r/MachineLearning`" in result
        assert ":thumbsup:" not in result
