import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipeline.digest_generator import DigestGenerator
from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem, DigestContent, DigestItem, RankedItem


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


def _content(lead="정확히 $7B 투자.", body="본문.", implication="시사점."):
    return DigestContent(
        lead=lead,
        headline_index=1,
        items=[DigestItem(title="T", url="u", body=body, implication=implication)],
    )


class TestAgiCountdown:
    def test_computes_day_count(self):
        from datetime import date

        from shared import agi_countdown_intro

        assert (
            agi_countdown_intro("2029-01-01", "AGI 등장 {days}일 전이다. ", date(2026, 1, 1))
            == "AGI 등장 1096일 전이다. "
        )

    def test_counts_up_after_d_day(self):
        from datetime import date

        from shared import agi_countdown_intro

        before, after = "D-{days}", "D+{days}"
        assert agi_countdown_intro("2029-01-01", before, date(2029, 1, 1), after) == "D+0"  # D-day
        assert agi_countdown_intro("2029-01-01", before, date(2029, 1, 11), after) == "D+10"  # after
        assert agi_countdown_intro("2029-01-01", before, date(2028, 12, 22), after) == "D-10"  # before

    def test_empty_past_date_or_disabled(self):
        from datetime import date

        from shared import agi_countdown_intro

        # past D-day with NO after_template → empty (countup opt-in)
        assert agi_countdown_intro("2029-01-01", "x{days}", date(2030, 1, 1)) == ""
        assert agi_countdown_intro("", "x{days}", date(2026, 1, 1)) == ""  # disabled
        assert agi_countdown_intro("not-a-date", "x{days}", date(2026, 1, 1)) == ""  # malformed


class TestGroundingCheck:
    @pytest.mark.asyncio
    async def test_revises_unsupported_claim(self):
        out = json.dumps(
            {
                "violations": [{"claim": "$7B", "issue": "not in source", "fix": "attributed"}],
                "corrected_digest": "LEAD: 보도에 따르면 대규모 투자.\nITEM 0 BODY: 본문.\nITEM 0 IMPLICATION: 시사점.",
            }
        )
        result = await _generator(out)._verify_grounding(_content(), _ranked())
        assert "보도에 따르면" in result.lead

    @pytest.mark.asyncio
    async def test_no_violation_keeps_original(self):
        content = _content(lead="근거 있는 문장.")
        out = json.dumps({"violations": [], "corrected_digest": "should be ignored"})
        result = await _generator(out)._verify_grounding(content, _ranked())
        assert result.lead == "근거 있는 문장."

    @pytest.mark.asyncio
    async def test_malformed_check_keeps_original(self):
        content = _content(lead="원본 다이제스트.")
        result = await _generator("not json")._verify_grounding(content, _ranked())
        assert result.lead == "원본 다이제스트."


def _source_detail(item) -> str:
    tag, metrics = DigestGenerator._source_tag_and_metrics(item)
    return " · ".join(p for p in (tag, metrics) if p)


class TestFormatSourceDetail:
    def test_reddit(self):
        # .rss feed carries no score/num_comments — only the subreddit tag is rendered.
        item = _item(SourceType.REDDIT, metadata={"subreddit": "LocalLLaMA"})
        assert _source_detail(item) == "`r/LocalLLaMA`"

    def test_youtube(self):
        item = _item(SourceType.YOUTUBE, metadata={"view_count": 12345})
        result = _source_detail(item)
        assert "`YouTube`" in result
        assert ":arrow_forward: 12,345" in result

    def test_x_with_author(self):
        item = _item(SourceType.X, author="karpathy")
        assert "`@karpathy`" in _source_detail(item)

    def test_rss_with_feed_title(self):
        item = _item(SourceType.RSS, metadata={"feed_title": "GeekNews - 개발/기술/스타트업 뉴스 서비스"})
        result = _source_detail(item)
        assert "`GeekNews`" in result
        assert "개발" not in result

    def test_web(self):
        item = CollectedItem(
            item_id="test",
            source_type=SourceType.WEB,
            title="Test",
            url="http://arxiv.org/abs/1234",
        )
        assert "`arxiv.org`" in _source_detail(item)

    def test_reddit_no_engagement(self):
        item = _item(SourceType.REDDIT, metadata={"subreddit": "MachineLearning"})
        result = _source_detail(item)
        assert "`r/MachineLearning`" in result
        assert ":thumbsup:" not in result
