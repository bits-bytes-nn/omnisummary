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


class TestParseContent:
    def test_parses_items_with_raw_control_chars(self):
        # Reproduces the 2026-07-11 prod failure: Sonnet 5 emitted unescaped newlines/tabs
        # inside string values, which strict json.loads rejected → 0-item fallback → Threads
        # got only the lead. The lenient parse must recover all items.
        raw = (
            '{"lead": "리드 문장.", "headline_index": 1, "items": ['
            '{"title": "T1", "url": "u1", "body": "본문 첫 줄.\n둘째 줄.", "implication": "시사점\t들여쓰기."},'
            '{"title": "T2", "url": "u2", "body": "다른 본문.", "implication": "또 다른 시사점."}'
            "]}"
        )
        content = _generator("")._parse_content(raw)
        assert len(content.items) == 2
        assert content.items[0].body == "본문 첫 줄.\n둘째 줄."
        assert content.items[0].implication == "시사점\t들여쓰기."
        assert content.headline_index == 1

    def test_malformed_json_falls_back_to_minimal(self):
        content = _generator("")._parse_content("totally not json")
        assert content.items == []
        assert content.lead == "totally not json"

    def test_one_malformed_item_does_not_collapse_whole_digest(self):
        # Valid JSON, valid lead, but the 2nd item is missing its required `url`. Item-level
        # validation must drop only that item and keep the other three — NOT fall back to a
        # 0-item digest (the same silent-empty failure class as the control-char bug).
        raw = json.dumps(
            {
                "lead": "리드 문장.",
                "items": [
                    {"title": "T0", "url": "u0", "body": "b0"},
                    {"title": "T1", "body": "b1"},  # missing url → invalid
                    {"title": "T2", "url": "u2", "body": "b2"},
                    {"title": "T3", "url": "u3", "body": "b3"},
                ],
            }
        )
        content = _generator("")._parse_content(raw)
        assert [it.url for it in content.items] == ["u0", "u2", "u3"]
        assert content.lead == "리드 문장."

    def test_missing_lead_falls_back_to_minimal(self):
        # Valid JSON with items but no usable lead → deterministic minimal fallback, not a crash.
        raw = json.dumps({"items": [{"title": "T0", "url": "u0", "body": "b0"}]})
        content = _generator("")._parse_content(raw)
        assert content.items == []


class TestTargetCountTrim:
    @pytest.mark.asyncio
    async def test_overemitted_items_trimmed_to_target(self):
        from datetime import date

        # The LLM ignores "EXACTLY target_count" and emits 5 items; with top_n=3 the digest must
        # trim deterministically to 3 (headline retained) rather than trusting prompt compliance.
        emitted = {
            "lead": "리드.",
            "headline_index": 1,
            "items": [{"title": f"T{i}", "url": f"u{i}", "body": "본문.", "implication": "시사점."} for i in range(5)],
        }
        config = PipelineConfig(enable_grounding_check=False, top_n=3)
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=json.dumps(emitted)))
        gen = DigestGenerator(config, factory)
        ranked = [
            RankedItem(
                item=CollectedItem(item_id=f"i{i}", source_type=SourceType.RSS, title=f"T{i}", url=f"u{i}"), score=0.8
            )
            for i in range(5)
        ]
        result = await gen.generate(ranked, [r.item for r in ranked], today=date(2030, 1, 1))
        assert len(result.content.items) == 3  # trimmed to top_n

    @pytest.mark.asyncio
    async def test_pinned_item_survives_trim(self):
        from datetime import date

        # The editor emits the pinned item LAST (u4); top_n=3 would normally trim it out, but a
        # pinned URL must survive the trim.
        emitted = {
            "lead": "리드.",
            "headline_index": 1,
            "items": [{"title": f"T{i}", "url": f"u{i}", "body": "본문.", "implication": "시사점."} for i in range(5)],
        }
        config = PipelineConfig(enable_grounding_check=False, top_n=3)
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=json.dumps(emitted)))
        gen = DigestGenerator(config, factory)
        ranked = [
            RankedItem(
                item=CollectedItem(
                    item_id=f"i{i}",
                    source_type=SourceType.RSS,
                    title=f"T{i}",
                    url=f"u{i}",
                    metadata={"pinned": True} if i == 4 else {},
                ),
                score=0.8,
            )
            for i in range(5)
        ]
        result = await gen.generate(ranked, [r.item for r in ranked], today=date(2030, 1, 1))
        urls = [it.url for it in result.content.items]
        assert len(urls) == 3
        assert "u4" in urls  # pinned survived
        assert urls[0] == "u0"  # headline preserved at front

    def test_trim_keeping_pinned_no_pins_is_plain_slice(self):
        ranked = [
            RankedItem(
                item=CollectedItem(item_id=f"i{i}", source_type=SourceType.RSS, title="T", url=f"u{i}"), score=0.5
            )
            for i in range(5)
        ]
        items = [DigestItem(title="T", url=f"u{i}", body="b") for i in range(5)]
        kept = DigestGenerator._trim_keeping_pinned(items, 3, ranked)
        assert [it.url for it in kept] == ["u0", "u1", "u2"]

    def test_headline_survives_when_pins_fill_all_slots(self):
        # Non-pinned headline (u0) + 5 pinned (u1..u5), target=5. The headline MUST be kept —
        # the lead prose and daily visual are about items[0]; dropping it desyncs them. One pin
        # is squeezed out instead (pins already exceed the remaining slots).
        ranked = [
            RankedItem(
                item=CollectedItem(
                    item_id=f"i{i}",
                    source_type=SourceType.RSS,
                    title="T",
                    url=f"u{i}",
                    metadata={} if i == 0 else {"pinned": True},
                ),
                score=0.5,
            )
            for i in range(6)
        ]
        items = [DigestItem(title="T", url=f"u{i}", body="b") for i in range(6)]
        kept = DigestGenerator._trim_keeping_pinned(items, 5, ranked)
        urls = [it.url for it in kept]
        assert len(urls) == 5
        assert urls[0] == "u0"  # headline preserved at the front


class TestFormatRecentLeads:
    def test_bullets_recent_leads(self):
        from pipeline.digest_generator import _format_recent_leads

        out = _format_recent_leads(["어제 리드.", "그제 리드."])
        assert out == "- 어제 리드.\n- 그제 리드."

    def test_empty_when_none(self):
        from pipeline.digest_generator import _format_recent_leads

        assert "No recent digests" in _format_recent_leads([])
        assert "No recent digests" in _format_recent_leads(["", "  "])


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
