import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipeline.digest_generator import DigestGenerator
from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem


def _item(source_type=SourceType.REDDIT, metadata=None, author=None):
    return CollectedItem(
        item_id="test",
        source_type=source_type,
        title="Test",
        url="http://test.com",
        metadata=metadata or {},
        author=author,
    )


def _generator(check_output: str):
    config = PipelineConfig()
    factory = MagicMock()
    factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=check_output))
    return DigestGenerator(config, factory)


def _ranked():
    return [
        RankedItem(
            item=CollectedItem(item_id="a", source_type=SourceType.RSS, title="T", url="u", text="body"), score=0.8
        )
    ]


class TestGroundingCheck:
    @pytest.mark.asyncio
    async def test_revises_unsupported_claim(self):
        out = json.dumps(
            {
                "violations": [{"claim": "$7B", "issue": "not in source", "fix": "attributed"}],
                "corrected_digest": "보도에 따르면 대규모 투자.",
            }
        )
        result = await _generator(out)._verify_grounding("정확히 $7B 투자.", _ranked())
        assert "보도에 따르면" in result

    @pytest.mark.asyncio
    async def test_no_violation_keeps_original(self):
        original = "근거 있는 문장."
        out = json.dumps({"violations": [], "corrected_digest": "should be ignored"})
        result = await _generator(out)._verify_grounding(original, _ranked())
        assert result == original

    @pytest.mark.asyncio
    async def test_malformed_check_keeps_original(self):
        original = "원본 다이제스트."
        result = await _generator("not json")._verify_grounding(original, _ranked())
        assert result == original


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
