import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from application.digest_generator import DigestContentError, DigestGenerator
from domain.models import CollectedItem, DigestContent, DigestItem, RankedItem
from shared.config import PipelineConfig
from shared.constants import SourceType


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


def _sources():
    """What the grounding check is given: the source items that actually SHIPPED, as
    _fill_source_metadata returns them."""
    return [r.item for r in _ranked()]


def _content(lead="정확히 $7B 투자.", body="본문.", implication="시사점."):
    return DigestContent(
        lead=lead,
        items=[DigestItem(title="T", url="u", body=body, implication=implication)],
    )


class TestAgiCountdown:
    def test_computes_day_count(self):
        from datetime import date

        from domain.formatting import agi_countdown_intro

        assert (
            agi_countdown_intro("2029-01-01", "AGI 등장 {days}일 전이다. ", date(2026, 1, 1))
            == "AGI 등장 1096일 전이다. "
        )

    def test_counts_up_after_d_day(self):
        from datetime import date

        from domain.formatting import agi_countdown_intro

        before, after = "D-{days}", "D+{days}"
        assert agi_countdown_intro("2029-01-01", before, date(2029, 1, 1), after) == "D+0"  # D-day
        assert agi_countdown_intro("2029-01-01", before, date(2029, 1, 11), after) == "D+10"  # after
        assert agi_countdown_intro("2029-01-01", before, date(2028, 12, 22), after) == "D-10"  # before

    def test_empty_past_date_or_disabled(self):
        from datetime import date

        from domain.formatting import agi_countdown_intro

        # past D-day with NO after_template → empty (countup opt-in)
        assert agi_countdown_intro("2029-01-01", "x{days}", date(2030, 1, 1)) == ""
        assert agi_countdown_intro("", "x{days}", date(2026, 1, 1)) == ""  # disabled
        assert agi_countdown_intro("not-a-date", "x{days}", date(2026, 1, 1)) == ""  # malformed

    def test_malformed_template_does_not_crash(self):
        from datetime import date

        from domain.formatting import agi_countdown_intro

        # An operator typo in the config template must degrade to no intro, never raise mid-run
        # (it's applied AFTER the expensive collect/rank/LLM work).
        assert agi_countdown_intro("2029-01-01", "AGI {day}일 전", date(2026, 1, 1)) == ""  # wrong placeholder
        assert agi_countdown_intro("2029-01-01", "AGI {days", date(2026, 1, 1)) == ""  # stray brace
        # a malformed after_template on/after D-day also degrades cleanly
        assert agi_countdown_intro("2029-01-01", "D-{days}", date(2029, 1, 2), "D+{nope}") == ""


class TestGroundingCheck:
    @pytest.mark.asyncio
    async def test_revises_unsupported_claim(self):
        out = json.dumps(
            {
                "violations": [{"claim": "$7B", "issue": "not in source", "fix": "dropped the figure"}],
                "corrected_digest": "LEAD: 대규모 투자.\nITEM 0 BODY: 본문.\nITEM 0 IMPLICATION: 시사점.",
            }
        )
        result = await _generator(out)._verify_grounding(_content(), _sources())
        assert result.lead.startswith("대규모 투자")

    @pytest.mark.asyncio
    async def test_no_violation_keeps_original(self):
        content = _content(lead="근거 있는 문장.")
        out = json.dumps({"violations": [], "corrected_digest": "should be ignored"})
        result = await _generator(out)._verify_grounding(content, _sources())
        assert result.lead == "근거 있는 문장."

    @pytest.mark.asyncio
    async def test_malformed_check_keeps_original(self):
        content = _content(lead="원본 다이제스트.")
        result = await _generator("not json")._verify_grounding(content, _sources())
        assert result.lead == "원본 다이제스트."


class TestParseContent:
    def test_parses_items_with_raw_control_chars(self):
        # Reproduces the 2026-07-11 prod failure: Sonnet 5 emitted unescaped newlines/tabs
        # inside string values, which strict json.loads rejected → 0-item fallback → Threads
        # got only the lead. The lenient parse must recover all items.
        raw = (
            '{"lead": "리드 문장.", "items": ['
            '{"title": "T1", "url": "u1", "body": "본문 첫 줄.\n둘째 줄.", "implication": "시사점\t들여쓰기."},'
            '{"title": "T2", "url": "u2", "body": "다른 본문.", "implication": "또 다른 시사점."}'
            "]}"
        )
        content = _generator("")._parse_content(raw)
        assert len(content.items) == 2
        assert content.items[0].body == "본문 첫 줄.\n둘째 줄."
        assert content.items[0].implication == "시사점\t들여쓰기."

    def test_parses_items_first_lead_last_key_order(self):
        # The prompt now asks for `items` BEFORE `lead`: the lead is written after the stories
        # exist, so it comments on them instead of re-narrating the headline (measured: the
        # lead/headline-reply word overlap fell from ~0.34 to ~0.11 across five sampled days).
        # Parsing must not depend on key order.
        raw = '{"items": [{"title": "T0", "url": "u0", "body": "b0", "implication": "i0"}], ' '"lead": "오늘의 논평."}'
        content = _generator("")._parse_content(raw)
        assert content.lead == "오늘의 논평."
        assert [it.url for it in content.items] == ["u0"]

    def test_malformed_json_raises_instead_of_shipping_an_empty_digest(self):
        # Regression (2026-08-13, 2026-08-17): the editor emitted a stray `]` after the lead
        # string, the old fallback returned lead=raw/items=[], and the day shipped a story-less
        # post whose "lead" was the raw fenced JSON. A parse failure must raise so the caller
        # re-asks — a digest with no stories is never a valid result.
        with pytest.raises(DigestContentError):
            _generator("")._parse_content("totally not json")

    def test_stray_bracket_after_lead_raises(self):
        # The exact production emission: a valid lead string closed with an extra `]`.
        raw = '{"lead": "오늘의 리드."], "items": [{"title": "T", "url": "u", "body": "b"}]}'
        with pytest.raises(DigestContentError):
            _generator("")._parse_content(raw)

    def test_valid_json_with_zero_items_raises(self):
        # Well-formed JSON is not enough: an empty items array is the same broken outcome.
        raw = json.dumps({"lead": "리드 문장.", "items": []})
        with pytest.raises(DigestContentError):
            _generator("")._parse_content(raw)

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

    def test_malformed_headline_item_raises(self):
        # If items[0] (the headline the lead + visual are about) fails validation, keeping the
        # rest would leave the lead/visual describing a dropped story. Raise and re-ask instead
        # of shipping a headline/lead/visual mismatch.
        raw = json.dumps(
            {
                "lead": "The headline is about the GPT-6 launch.",
                "items": [
                    {"title": "GPT-6 launches", "body": "no url"},  # headline invalid
                    {"title": "T1", "url": "u1", "body": "b1"},
                ],
            }
        )
        with pytest.raises(DigestContentError):
            _generator("")._parse_content(raw)

    def test_missing_lead_raises(self):
        # Valid JSON with items but no usable lead → re-askable failure, not a lead-less digest.
        raw = json.dumps({"items": [{"title": "T0", "url": "u0", "body": "b0"}]})
        with pytest.raises(DigestContentError):
            _generator("")._parse_content(raw)


class TestProseBudgetReachesTheEditor:
    """The renderer enforces the 500-char Threads cap by dropping trailing body sentences, but the
    editor was never told a budget: 5 of 95 sampled items lost their closing sentence (median 106
    chars — usually the concrete figures), and adding the source line pushed that to 8 of 95."""

    def test_the_budget_clause_points_at_the_items_own_number(self):
        # One number for the whole digest charged every short-URL item for the longest URL in the
        # pool, so the clause defers to the PROSE BUDGET stated with each candidate.
        from application.digest_generator import _PROSE_BUDGET_RULE

        assert "PROSE BUDGET" in _PROSE_BUDGET_RULE
        assert "implication" in _PROSE_BUDGET_RULE

    @pytest.mark.asyncio
    async def test_budget_is_passed_into_the_prompt(self):
        from datetime import date

        emitted = {
            "items": [{"title": "T", "url": "u", "body": "본문.", "implication": "시사점."}],
            "lead": "리드.",
        }
        seen: list[str] = []

        def _reply(prompt_value):
            seen.append(str(prompt_value))
            return AIMessage(content=json.dumps(emitted))

        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(_reply)
        config = PipelineConfig(enable_grounding_check=False)
        gen = DigestGenerator(config, factory)
        ranked = [
            RankedItem(item=CollectedItem(item_id="i", source_type=SourceType.RSS, title="T", url="u"), score=0.8)
        ]
        await gen.generate(ranked, today=date(2030, 1, 1))
        budget = gen._item_prose_budget(ranked[0].item)
        assert any(f"{budget} characters" in s for s in seen)


class TestDerivedBudgets:
    """The per-item number is computed from the parts CODE owns (URL + source line + separators)
    and now covers the TITLE too, because the editor authors it."""

    @staticmethod
    def _gen(**overrides) -> DigestGenerator:
        factory = MagicMock()
        factory.get_model.return_value = MagicMock()
        return DigestGenerator(PipelineConfig(**overrides), factory)

    @staticmethod
    def _web(url: str) -> RankedItem:
        return RankedItem(item=CollectedItem(item_id=url, source_type=SourceType.WEB, title="T", url=url), score=0.8)

    def test_budget_shrinks_with_the_real_fixed_parts(self):
        from shared.constants import THREADS_MAX_POST_CHARS

        gen = self._gen()
        short_url, long_url = "https://e.com/a", "https://e.com/" + "p" * 120
        short = gen._item_prose_budget(self._web(short_url).item)
        long_ = gen._item_prose_budget(self._web(long_url).item)
        assert short < THREADS_MAX_POST_CHARS
        assert long_ == short - (len(long_url) - len(short_url))

    def test_each_candidate_keeps_its_own_budget(self):
        # The budget used to be the worst case across the whole pool, so a short-URL item was charged
        # for the longest URL in it and lost its closing sentence to a cap it never came near.
        gen = self._gen()
        short, long_ = self._web("https://e.com/a"), self._web("https://e.com/" + "p" * 120)
        assert gen._item_prose_budget(short.item) > gen._item_prose_budget(long_.item)

    def test_every_candidates_budget_reaches_the_editor(self):
        gen = self._gen()
        items = [self._web("https://e.com/a"), self._web("https://e.com/" + "p" * 120)]
        rendered = gen._format_ranked_items(items)
        for ranked in items:
            assert f"{gen._item_prose_budget(ranked.item)} characters" in rendered

    def test_lead_budget_reserves_the_code_owned_countdown(self):
        from shared.constants import THREADS_MAX_POST_CHARS

        gen = self._gen()
        intro = "AGI 등장 100일 전이다. "
        assert gen._lead_budget(intro) == THREADS_MAX_POST_CHARS - len(intro)
        assert gen._lead_budget("") == THREADS_MAX_POST_CHARS

    def test_lead_budget_reaches_the_prompt(self):
        import asyncio
        from datetime import date

        emitted = {"items": [{"title": "T", "url": "u", "body": "본문."}], "lead": "리드."}
        seen: list[str] = []
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(
            lambda p: (seen.append(str(p)), AIMessage(content=json.dumps(emitted)))[1]
        )
        config = PipelineConfig(enable_grounding_check=False, agi_countdown_date="")
        gen = DigestGenerator(config, factory)
        ranked = [
            RankedItem(item=CollectedItem(item_id="i", source_type=SourceType.RSS, title="T", url="u"), score=0.8)
        ]
        asyncio.run(gen.generate(ranked, today=date(2030, 1, 1)))
        assert any("under 500 characters" in s for s in seen)


class TestReAskOnUnparseableEmission:
    """Regression (2026-08-13, 2026-08-17): the editor emitted malformed JSON once, the digest
    silently became lead=raw/items=[], and both days published a story-less post. generate() must
    re-ask, and must fail the run outright rather than return a digest with no stories."""

    @staticmethod
    def _emitted(n: int = 2) -> str:
        return json.dumps(
            {
                "lead": "리드.",
                "items": [
                    {"title": f"T{i}", "url": f"u{i}", "body": "본문.", "implication": "시사점."} for i in range(n)
                ],
            }
        )

    @staticmethod
    def _ranked(n: int = 2):
        return [
            RankedItem(
                item=CollectedItem(item_id=f"i{i}", source_type=SourceType.RSS, title=f"T{i}", url=f"u{i}"), score=0.8
            )
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_stray_bracket_emission_is_re_asked_and_recovers(self, monkeypatch):
        from datetime import date

        monkeypatch.setattr("shared.utils.asyncio.sleep", AsyncMock())
        # First emission carries the production defect (a stray `]` closing the lead); the re-ask
        # returns valid JSON. The digest must come back whole, not empty.
        outputs = ['{"lead": "리드."], "items": []}', self._emitted()]
        calls = {"n": 0}

        def _reply(_):
            calls["n"] += 1
            return AIMessage(content=outputs[min(calls["n"] - 1, len(outputs) - 1)])

        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(_reply)
        gen = DigestGenerator(PipelineConfig(enable_grounding_check=False), factory)
        ranked = self._ranked()

        result = await gen.generate(ranked, today=date(2030, 1, 1))
        assert calls["n"] == 2  # asked again instead of degrading
        assert len(result.content.items) == 2

    @pytest.mark.asyncio
    async def test_persistently_unparseable_emission_raises(self, monkeypatch):
        from datetime import date

        monkeypatch.setattr("shared.utils.asyncio.sleep", AsyncMock())
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content="```json\n{oops"))
        gen = DigestGenerator(PipelineConfig(enable_grounding_check=False, digest_max_retries=2), factory)
        ranked = self._ranked()

        # A failed run is the correct outcome: the Lambda reports the error (alarm fires) and no
        # broken digest reaches AgentCore Memory or Threads.
        with pytest.raises(DigestContentError):
            await gen.generate(ranked, today=date(2030, 1, 1))


class TestTargetCountTrim:
    @pytest.mark.asyncio
    async def test_overemitted_items_trimmed_to_target(self):
        from datetime import date

        # The LLM ignores "EXACTLY target_count" and emits 5 items; with top_n=3 the digest must
        # trim deterministically to 3 (headline retained) rather than trusting prompt compliance.
        emitted = {
            "lead": "리드.",
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
        result = await gen.generate(ranked, today=date(2030, 1, 1))
        assert len(result.content.items) == 3  # trimmed to top_n

    @pytest.mark.asyncio
    async def test_an_unmatched_or_repeated_item_does_not_take_a_valid_storys_slot(self):
        from datetime import date

        # Five emitted, top_n=3: #2 has an invented URL and #3 repeats #1. Matching runs BEFORE the
        # trim, so the valid #4 ships instead of the digest coming out one story short.
        urls = [
            "https://e.com/0",
            "https://e.com/1",
            "https://invented.example/x",
            "https://e.com/1/",
            "https://e.com/4",
        ]
        emitted = {
            "lead": "리드.",
            "items": [
                {"title": f"T{i}", "url": u, "body": "본문.", "implication": "시사점."} for i, u in enumerate(urls)
            ],
        }
        config = PipelineConfig(enable_grounding_check=False, top_n=3)
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=json.dumps(emitted)))
        gen = DigestGenerator(config, factory)
        ranked = [
            RankedItem(
                item=CollectedItem(
                    item_id=f"i{i}", source_type=SourceType.RSS, title=f"T{i}", url=f"https://e.com/{i}"
                ),
                score=0.8,
            )
            for i in (0, 1, 4)
        ]
        result = await gen.generate(ranked, today=date(2030, 1, 1))
        assert [it.url for it in result.content.items] == ["https://e.com/0", "https://e.com/1", "https://e.com/4"]

    @pytest.mark.asyncio
    async def test_pins_exceeding_top_n_all_survive_with_headline(self):
        from datetime import date

        # User pins 4 URLs but top_n=3. Both the non-pinned headline (u0) AND every pin (u1..u4)
        # must survive — the target is raised to fit them rather than dropping a pin or the headline.
        emitted = {
            "lead": "리드.",
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
                    metadata={"pinned": True} if i >= 1 else {},  # u1..u4 pinned, u0 headline
                ),
                score=0.8,
            )
            for i in range(5)
        ]
        result = await gen.generate(ranked, today=date(2030, 1, 1))
        urls = [it.url for it in result.content.items]
        assert urls[0] == "u0"  # headline preserved at front
        for pin in ("u1", "u2", "u3", "u4"):
            assert pin in urls  # every pin survived

    @pytest.mark.asyncio
    async def test_pinned_item_survives_trim(self):
        from datetime import date

        # The editor emits the pinned item LAST (u4); top_n=3 would normally trim it out, but a
        # pinned URL must survive the trim.
        emitted = {
            "lead": "리드.",
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
        result = await gen.generate(ranked, today=date(2030, 1, 1))
        urls = [it.url for it in result.content.items]
        assert len(urls) == 3
        assert "u4" in urls  # pinned survived
        assert urls[0] == "u0"  # headline preserved at front

    @pytest.mark.asyncio
    async def test_a_pinned_story_the_editor_left_out_is_logged(self):
        from datetime import date

        emitted = {
            "lead": "리드.",
            "items": [{"title": f"T{i}", "url": f"u{i}", "body": "본문.", "implication": "시사점."} for i in range(2)],
        }
        config = PipelineConfig(enable_grounding_check=False, top_n=2)
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=json.dumps(emitted)))
        ranked = [
            RankedItem(
                item=CollectedItem(
                    item_id=f"i{i}",
                    source_type=SourceType.RSS,
                    title=f"T{i}",
                    url=f"u{i}",
                    metadata={"pinned": True} if i == 2 else {},
                ),
                score=0.8,
            )
            for i in range(3)
        ]
        with patch("application.digest_generator.logger") as log:
            await DigestGenerator(config, factory).generate(ranked, today=date(2030, 1, 1))
        assert any("Pinned URL(s) missing" in str(c.args[0]) for c in log.error.call_args_list)

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
        from application.digest_generator import _format_recent_leads

        out = _format_recent_leads(["어제 리드.", "그제 리드."])
        assert out == "- 어제 리드.\n- 그제 리드."

    def test_empty_when_none(self):
        from application.digest_generator import _format_recent_leads

        assert "No recent digests" in _format_recent_leads([])
        assert "No recent digests" in _format_recent_leads(["", "  "])

    def test_only_the_thesis_sentence_of_each_lead_is_shown(self):
        # The lead names its story first and lands its thesis last; the thesis is what must not
        # repeat, and the first sentence never carried one.
        from application.digest_generator import _format_recent_leads

        stored = "오픈AI가 또 모델을 냈다. 두 번째 문장이다. 결국 이득은 플랫폼이 챙긴다."
        out = _format_recent_leads([stored])
        assert out == "- 결국 이득은 플랫폼이 챙긴다."

    def test_long_leads_are_truncated_in_code_not_by_a_prompt_rule(self):
        # Five full leads crowded the prompt with prose the editor is not being asked to compare.
        from application.digest_generator import RECENT_LEAD_PREVIEW_CHARS, _format_recent_leads

        out = _format_recent_leads(["가" * (RECENT_LEAD_PREVIEW_CHARS + 50)])
        assert out.startswith("- " + "가" * RECENT_LEAD_PREVIEW_CHARS)
        assert out.endswith("…")
        assert len(out) < RECENT_LEAD_PREVIEW_CHARS + 50


class TestFormatRecentTitles:
    """The editor is shown what the LAST digest ran, so today isn't a re-run of it. Information
    only — the URL ledger stays the mechanism that suppresses a repeat."""

    def test_bullets_the_titles(self):
        from application.digest_generator import format_recent_titles

        assert format_recent_titles(["첫 스토리", "둘째 스토리"]) == "- 첫 스토리\n- 둘째 스토리"

    def test_states_when_there_is_no_recent_digest(self):
        from application.digest_generator import format_recent_titles

        assert "No recent digest" in format_recent_titles([])
        assert "No recent digest" in format_recent_titles(None)


def _source_detail(item) -> str:
    tag, metrics = DigestGenerator._source_tag_and_metrics(item)
    return " · ".join(p for p in (tag, metrics) if p)


class TestFormatSourceDetail:
    def test_reddit(self):
        # .rss feed carries no score/num_comments — only the subreddit tag is rendered.
        item = _item(SourceType.REDDIT, metadata={"subreddit": "LocalLLaMA"})
        assert _source_detail(item) == "r/LocalLLaMA"

    def test_youtube(self):
        item = _item(SourceType.YOUTUBE, metadata={"view_count": 12345})
        result = _source_detail(item)
        assert "YouTube" in result
        # A literal Unicode emoji, not a `:shortcode:` — Threads renders no shortcodes, and the
        # markup strip removes underscores, which published a bare ":arrowforward:".
        assert "조회 12,345" in result
        assert ":" not in result.split("YouTube")[-1]

    def test_x_with_author(self):
        item = _item(SourceType.X, author="karpathy")
        assert "@karpathy" in _source_detail(item)

    def test_rss_with_feed_title(self):
        item = _item(SourceType.RSS, metadata={"feed_title": "GeekNews - 개발/기술/스타트업 뉴스 서비스"})
        result = _source_detail(item)
        assert "GeekNews" in result
        assert "개발" not in result

    def test_web(self):
        item = CollectedItem(
            item_id="test",
            source_type=SourceType.WEB,
            title="Test",
            url="http://arxiv.org/abs/1234",
        )
        assert "arxiv.org" in _source_detail(item)

    def test_reddit_no_engagement(self):
        item = _item(SourceType.REDDIT, metadata={"subreddit": "MachineLearning"})
        result = _source_detail(item)
        assert "r/MachineLearning" in result
        assert ":thumbsup:" not in result


class TestCountdownIntro:
    """The gag opens the lead verbatim; the strip helper removes it so recent-leads novelty and the
    visual's editorial take never compare boilerplate."""

    def test_prefix_places_the_intro_first(self):
        from domain.formatting import place_countdown_intro

        assert (
            place_countdown_intro("오늘의 각도다.", "AGI 등장 870일 전이다. ")
            == "AGI 등장 870일 전이다. 오늘의 각도다."
        )

    def test_idempotent(self):
        from domain.formatting import place_countdown_intro

        intro = "AGI 등장 870일 전이다. "
        prefixed = place_countdown_intro("각도다.", intro)
        assert place_countdown_intro(prefixed, intro) == prefixed

    def test_no_intro_or_no_lead_changes_nothing(self):
        from domain.formatting import place_countdown_intro

        assert place_countdown_intro("각도다.", "") == "각도다."
        assert place_countdown_intro("", "AGI 등장 870일 전이다. ") == ""

    def test_editorial_lead_strips_the_gag(self):
        from domain.formatting import editorial_lead

        intro = "AGI 등장 870일 전이다. "
        assert editorial_lead("AGI 등장 870일 전이다. 각도다.", intro) == "각도다."
        assert editorial_lead("각도다.", intro) == "각도다."  # never attached


class TestSourceMatchingByNormalizedUrl:
    """The editor echoes each story's URL back, and the source tag was matched by EXACT string: one
    trailing slash, an http→https flip or a dropped utm param and the story shipped with no
    provenance line at all on Threads."""

    def _generator_with(self, ranked_url: str):
        gen = _generator("{}")
        return gen, [
            RankedItem(
                item=CollectedItem(
                    item_id="a",
                    source_type=SourceType.RSS,
                    title="T",
                    url=ranked_url,
                    metadata={"feed_title": "Interconnects"},
                ),
                score=0.9,
            )
        ]

    def test_trailing_slash_and_scheme_variants_still_match(self):
        gen, ranked = self._generator_with("http://www.interconnects.ai/p/x/")
        content = DigestContent(lead="l", items=[DigestItem(title="T", url="https://interconnects.ai/p/x", body="b")])
        gen._fill_source_metadata(content, ranked)
        assert content.items[0].source_tag == "Interconnects"

    def test_identical_urls_are_unaffected(self):
        gen, ranked = self._generator_with("https://interconnects.ai/p/x")
        content = DigestContent(lead="l", items=[DigestItem(title="T", url="https://interconnects.ai/p/x", body="b")])
        gen._fill_source_metadata(content, ranked)
        assert content.items[0].source_tag == "Interconnects"

    def test_the_match_map_is_returned_for_the_grounding_check(self):
        gen, ranked = self._generator_with("https://interconnects.ai/p/x")
        content = DigestContent(lead="l", items=[DigestItem(title="T", url="https://interconnects.ai/p/x", body="b")])
        assert gen._fill_source_metadata(content, ranked) == [ranked[0].item]


class TestUnmatchedItemsNeverShip:
    """An item matching no ranked candidate is a story the editor invented (or whose URL it mangled
    beyond normalization). It used to be tagged with its own host, shipped to the reader, and written
    into the published-URL ledger — which then suppressed the REAL article for the whole TTL window."""

    @staticmethod
    def _ranked_two():
        return [
            RankedItem(
                item=CollectedItem(
                    item_id=str(n),
                    source_type=SourceType.RSS,
                    title="T",
                    url=f"https://interconnects.ai/p/{n}",
                    metadata={"feed_title": "Interconnects"},
                ),
                score=0.9,
            )
            for n in (1, 2)
        ]

    def test_an_unmatched_non_headline_item_is_dropped(self):
        gen = _generator("{}")
        ranked = self._ranked_two()
        content = DigestContent(
            lead="l",
            items=[
                DigestItem(title="T1", url="https://interconnects.ai/p/1", body="b"),
                DigestItem(title="Invented", url="https://www.newsite.com/a/b", body="b"),
            ],
        )
        sources = gen._fill_source_metadata(content, ranked)
        assert [it.url for it in content.items] == ["https://interconnects.ai/p/1"]
        assert [s.item_id for s in sources] == ["1"]

    def test_an_unmatched_headline_rejects_the_whole_emission(self):
        # The lead and the daily visual are both written about items[0], so there is nothing to
        # salvage — DigestContentError is what makes generate()'s retry re-ask.
        gen = _generator("{}")
        content = DigestContent(
            lead="l",
            items=[
                DigestItem(title="Invented", url="https://www.newsite.com/a/b", body="b"),
                DigestItem(title="T1", url="https://interconnects.ai/p/1", body="b"),
            ],
        )
        with pytest.raises(DigestContentError, match="Headline item"):
            gen._fill_source_metadata(content, self._ranked_two())

    @pytest.mark.asyncio
    async def test_generate_re_asks_and_recovers_from_an_invented_headline(self):
        first = json.dumps(
            {"lead": "리드.", "items": [{"title": "Invented", "url": "https://nowhere.example/x", "body": "b"}]}
        )
        second = json.dumps({"lead": "리드.", "items": [{"title": "T", "url": "u", "body": "b"}]})
        outputs = [first, second]
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=outputs.pop(0)))
        gen = DigestGenerator(
            PipelineConfig(enable_grounding_check=False, digest_max_retries=2, digest_retry_backoff_sec=0), factory
        )
        result = await gen.generate(_ranked(), today=date(2030, 1, 1))
        assert result.content is not None
        assert [it.url for it in result.content.items] == ["u"]
        assert outputs == []  # the first emission was re-asked, not shipped

    @pytest.mark.asyncio
    async def test_grounding_sees_only_the_sources_that_shipped(self):
        # Joining ALL ranked candidates meant a specific claim whose only support was a candidate the
        # editor DROPPED read as grounded — a false negative in the one pass that exists to catch
        # invented specifics — and carried the pool's surplus input tokens.
        shipped = RankedItem(
            item=CollectedItem(item_id="a", source_type=SourceType.RSS, title="Shipped", url="u1", text="kept body"),
            score=0.9,
        )
        dropped = RankedItem(
            item=CollectedItem(item_id="b", source_type=SourceType.RSS, title="Dropped", url="u2", text="unused body"),
            score=0.8,
        )
        payload = json.dumps({"lead": "리드.", "items": [{"title": "T", "url": "u1", "body": "b"}]})
        factory = MagicMock()
        seen: list[str] = []

        def _respond(prompt_value):
            seen.append(str(prompt_value))
            return AIMessage(content=payload if len(seen) == 1 else json.dumps({"violations": [], "corrected": ""}))

        factory.get_model.return_value = RunnableLambda(_respond)
        factory.truncate_to_tokens.side_effect = lambda text, max_tokens: text
        gen = DigestGenerator(PipelineConfig(enable_grounding_check=True, top_n=1), factory)
        await gen.generate([shipped, dropped], today=date(2030, 1, 1))
        assert "kept body" in seen[1]
        assert "unused body" not in seen[1]


class TestDroppedStoryIsAnError:
    """A digest that lost a story to item-level validation still looks completely normal
    downstream, so the log line is the only trace. Dropping an emitted item is an ERROR; ending up
    with fewer stories because the editor MERGED same-event items is not (that stays a warning in
    the caller, which knows the target)."""

    def test_dropped_item_logs_an_error(self):
        raw = json.dumps(
            {
                "lead": "리드 문장.",
                "items": [
                    {"title": "T0", "url": "u0", "body": "b0"},
                    {"title": "T1", "body": "b1"},  # missing url → dropped
                ],
            }
        )
        with patch("application.digest_generator.logger.error") as err:
            content = _generator("")._parse_content(raw)
        assert len(content.items) == 1
        assert err.called

    def test_a_complete_emission_logs_no_error(self):
        raw = json.dumps({"lead": "리드 문장.", "items": [{"title": "T0", "url": "u0", "body": "b0"}]})
        with patch("application.digest_generator.logger.error") as err:
            _generator("")._parse_content(raw)
        assert not err.called


class TestGroundingStageAttribution:
    def test_the_grounding_pass_is_billed_under_its_own_stage(self):
        # A second ~50k-token Sonnet call billed as "digest" is indistinguishable from the generation
        # itself, which defeats the only spend-attribution mechanism the repo has.
        factory = MagicMock()
        factory.get_model.return_value = MagicMock()
        DigestGenerator(PipelineConfig(), factory)
        stages = [call.kwargs["stage"] for call in factory.get_model.call_args_list]
        assert stages == ["digest", "grounding"]


def test_grounding_corrections_land_on_the_right_fields_and_leave_the_input_alone():
    from application.digest_generator import _apply_grounding_payload
    from domain.models import DigestContent, DigestItem

    original = DigestContent(
        lead="원래 리드.",
        items=[
            DigestItem(title="A", url="u0", body="본문 0.", implication="시사점 0."),
            DigestItem(title="B", url="u1", body="본문 1.", implication="시사점 1."),
        ],
    )
    corrected = "\n".join(
        [
            "LEAD: 고친 리드.",
            "ITEM 0 BODY: 본문 0.",
            "ITEM 1 BODY: 고친 본문 1.",
            "둘째 줄.",
            "ITEM 1 IMPLICATION: 고친 시사점 1.",
            "ITEM 0 IMPLICATION:",
        ]
    )
    out = _apply_grounding_payload(original, corrected)
    assert out.lead == "고친 리드."
    assert out.items[1].body == "고친 본문 1.\n둘째 줄."
    assert out.items[1].implication == "고친 시사점 1."
    assert out.items[0].implication == "시사점 0."  # an empty marker keeps the original
    assert original.items[1].body == "본문 1."  # the input is not mutated


def test_grounding_corrections_with_labels_never_sent_are_not_applied():
    # A model that renumbered ITEM 0..1 as 1..2 would have written item 0's text onto item 1.
    from application.digest_generator import _apply_grounding_payload
    from domain.models import DigestContent, DigestItem

    original = DigestContent(
        lead="리드.",
        items=[
            DigestItem(title="A", url="u0", body="본문 0.", implication="시사점 0."),
            DigestItem(title="B", url="u1", body="본문 1.", implication="시사점 1."),
        ],
    )
    corrected = "LEAD: 리드.\nITEM 1 BODY: 고친 본문 0.\nITEM 2 BODY: 본문 1."
    assert _apply_grounding_payload(original, corrected) is original


def test_grounding_checks_and_corrects_the_editors_titles():
    from application.digest_generator import _apply_grounding_payload, _grounding_payload
    from domain.models import DigestContent, DigestItem

    original = DigestContent(lead="리드.", items=[DigestItem(title="72% 향상", url="u0", body="본문.")])
    assert "ITEM 0 TITLE: 72% 향상" in _grounding_payload(original)
    out = _apply_grounding_payload(original, "LEAD: 리드.\nITEM 0 TITLE: 27% 향상\nITEM 0 BODY: 본문.")
    assert out.items[0].title == "27% 향상"
