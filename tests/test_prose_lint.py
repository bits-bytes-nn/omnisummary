import pytest

from shared.prose_lint import (
    ItemProse,
    figures,
    lead_figure_repeats,
    lead_specificity_hits,
    lint_digest_prose,
    specifics,
)


def item(body: str = "", implication: str = "", *, title: str = "", budget: int = 0) -> ItemProse:
    return ItemProse(title=title, body=body, implication=implication, budget=budget)


class TestCommaAfterAFinishedPredicate:
    """KOREAN_STYLE_RULES bans this form BY NAME and PipelineConfig spells out the exact shape with
    examples, yet digest_2026-07-12 items[3].implication shipped '못 쓴다,'. A prompt rule that has
    already failed is not fixed by writing it again."""

    def test_the_shipped_defect_is_caught(self):
        hits = lint_digest_prose("리드다.", [item("본문이다.", "그래서 못 쓴다, 그게 문제다.")])
        assert len(hits) == 1
        assert "items[0].implication" in hits[0]
        assert "comma after a finished predicate" in hits[0]

    def test_the_documented_examples_are_caught(self):
        for prose in ("성립한다, 그 순간이 오지 않으면 끝이다.", "이제 토큰의 시대다, 토큰이 곧 돈이다."):
            assert lint_digest_prose(prose, [])

    def test_a_comma_inside_one_sentence_is_fine(self):
        assert lint_digest_prose("빠르고, 정확하고, 값도 싸다.", [item("본문이다.", "시사점이다.")]) == []

    def test_a_period_after_the_predicate_is_fine(self):
        assert lint_digest_prose("성립한다. 그 순간이 오지 않으면 끝이다.", []) == []

    def test_a_restructured_connective_is_fine(self):
        assert lint_digest_prose("성립하는데, 그 순간이 오지 않으면 끝이다.", []) == []

    def test_the_lead_is_checked_too(self):
        hits = lint_digest_prose("전제는 성립한다, 문제는 시점이다.", [])
        assert hits and hits[0].startswith("lead:")


class TestTheCommaCheckDoesNotFireOnASyllableThatMerelyReadsLikeAPredicate:
    """'다' is also an ordinary word-final syllable, and it legitimately precedes a comma in the
    quotative. The bare `[가-힣]다,` pattern fired on both, and every false hit bought a byte-identical
    ~50k-token re-ask that changed nothing."""

    NO_HIT = (
        "회의실에 준비한 건 소다, 커피, 물이 전부다.",
        "그가 좋아하는 건 바다, 산, 그리고 오래된 서점이다.",
        "가능성은 무한하다, 라고 그는 말했다.",
        "쉽지 않다, 라는 게 현장의 중론이다.",
    )

    @pytest.mark.parametrize("prose", NO_HIT)
    def test_a_list_separator_and_the_quotative_are_left_alone(self, prose):
        assert lint_digest_prose(prose, []) == []
        assert lint_digest_prose("리드다.", [item(prose, "")]) == []
        assert lint_digest_prose("리드다.", [item("본문이다.", prose)]) == []

    def test_a_second_clause_after_a_list_is_still_caught(self):
        # The tightening must not buy silence: once the segment after the comma stands on its own as a
        # clause, the form is the one the rules name.
        assert lint_digest_prose("커피, 물은 준비했다, 소다는 빠졌다.", [])


class TestOnlyRulesTheConfigStatesAreChecked:
    """A check with no rule behind it is a style opinion with a re-ask budget attached.

    An em-dash-after-a-predicate pattern once lived here, cited KOREAN_STYLE_RULES as its source of
    truth, and appears nowhere in it: the rules ban the colon and the comma BY NAME and say nothing
    about a dash. It fired on 3 of the 4 shipped digests over idiomatic Korean editorial prose, each
    hit costing a byte-identical ~50k-token Sonnet re-ask that kept the content anyway.

    The sentences below are verbatim from digest_2026-06-11/06-16/06-21."""

    SHIPPED_EM_DASH_PROSE = (
        "인간은 훨씬 적은 샘플로 훨씬 많은 것을 배운다 — 그렇다면 AI 진보의 진짜 병목은 데이터의 양이 아니다.",
        "이것은 단순한 철학적 선언이 아니다 — Microsoft는 모델 경쟁에서 이기는 전략보다 생태계를 택했다.",
        "돈은 넘치는데 M&A는 줄었다 — 이들이 '사는' 대신 '짓거나 투자하는' 전략으로 전환했다는 뜻이다.",
        "Anthropic이 그를 어디에 쓸지는 아직 모른다 — 그게 오히려 더 흥미롭다.",
    )

    def test_the_style_rules_name_the_colon_and_the_comma_and_no_dash(self):
        from shared.config import KOREAN_STYLE_RULES

        assert "colon" in KOREAN_STYLE_RULES
        assert "comma after a finished predicate" in KOREAN_STYLE_RULES
        assert "dash" not in KOREAN_STYLE_RULES

    @pytest.mark.parametrize("prose", SHIPPED_EM_DASH_PROSE)
    def test_shipped_prose_joined_by_a_dash_is_left_alone(self, prose):
        assert lint_digest_prose(prose, [item("본문이다.", "시사점이다.")]) == []
        assert lint_digest_prose("리드다.", [item(prose, "")]) == []
        assert lint_digest_prose("리드다.", [item("본문이다.", prose)]) == []


class TestTheShippedDigestCorpus:
    """The maintainer's stored digests are the only real sample of what the editor writes. A check
    that fires on them over a rule NOTHING states is a false positive by construction.

    digest_state/ is gitignored (it is run output, not source), so this is a local gate and skips
    where the corpus is absent. The verbatim sentences in TestOnlyRulesTheConfigStatesAreChecked are
    what pins the same regression in CI."""

    # Every check that has a rule behind it: KOREAN_STYLE_RULES names the comma, and DigestPrompt tells
    # the lead to add to items[0] rather than re-tell it (including its numbers). A hit outside this set
    # means a check was added without a rule to trace it to.
    RULE_BACKED = {
        "comma after a finished predicate",
        "every specific it names is already in items[0]",
        "repeats items[0]'s figure(s)",
    }

    @staticmethod
    def _stored_digests():
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "digest_state"
        for path in sorted(root.glob("digest_*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            content = (raw.get("digest_result") or raw).get("content")
            if content and content.get("items"):
                yield path.name, content

    def test_no_check_fires_on_shipped_prose_without_a_rule_behind_it(self):
        stored = list(self._stored_digests())
        if not stored:
            pytest.skip("no stored digests to lint (digest_state/ is gitignored run output)")
        fired = set()
        for name, content in stored:
            for hit in lint_digest_prose(
                content["lead"],
                [item(one.get("body", ""), one.get("implication", "")) for one in content["items"]],
            ):
                claim = hit.split("—")[0].split(": ", 1)[1].strip() if ": " in hit else hit
                fired.add(next((rule for rule in self.RULE_BACKED if claim.startswith(rule)), claim))
                print(f"{name}: {hit}")
        assert fired <= self.RULE_BACKED


class TestSpecifics:
    def test_numbers_and_latin_names_count(self):
        assert specifics("GPT-5는 40% 빨라졌다") >= {"gpt-5", "40"}

    def test_korean_only_prose_has_none(self):
        assert specifics("추론이 빨라졌다") == set()

    def test_a_single_latin_letter_is_not_a_name(self):
        assert specifics("A 모델") == set()


class TestFigures:
    def test_a_digit_inside_a_latin_name_is_not_a_figure_of_its_own(self):
        # 'GPT-5' is already compared as a Latin token; counting its 5 as a figure would make every
        # mention of the model name a repeated number.
        assert figures("GPT-5가 나왔다") == set()

    def test_a_digit_inside_a_korean_word_is_not_a_figure(self):
        # '제3자' is a word, not a quantity; it was the only hit on digest_2026-06-11.
        assert figures("제3자 테스트를 요구했다") == set()

    def test_precision_does_not_make_two_figures_out_of_one(self):
        assert figures("52.2%") == figures("52%")

    def test_a_thousands_separator_stays_one_figure(self):
        assert figures("6,700개") == {"6700"}


class TestLeadSpecificity:
    """The lead spent its longest sentence re-telling items[0]'s numbers — which DigestPrompt
    explicitly forbids — while the tuned word-level Jaccard metric passed it at 0.10."""

    def test_a_lead_that_only_repeats_the_headlines_specifics_is_flagged(self):
        hits = lead_specificity_hits("GPT-5가 40% 빨라졌다.", "GPT-5는 40% 빠르다.\n의미가 크다.")
        assert hits and "already in items[0]" in hits[0]

    def test_a_lead_adding_one_new_specific_passes(self):
        assert lead_specificity_hits("GPT-5는 Gemini와 다른 길을 갔다.", "GPT-5가 나왔다.\n의미가 크다.") == []

    def test_a_purely_qualitative_lead_is_not_flagged(self):
        # A lead that names no figure at all is a different shape; treating it as a violation would be
        # a NEW rule rather than a check on the one the prompt already states.
        assert lead_specificity_hits("이번 발표의 의미는 속도가 아니라 배포다.", "GPT-5가 40% 빨라졌다.") == []

    def test_only_the_headline_item_is_compared(self):
        # The lead is written about items[0]; a specific that appears only in a LATER story is still
        # new information relative to what the reader sees directly beneath the lead.
        hits = lint_digest_prose(
            "GPT-5가 40% 빨라졌다.",
            [item("GPT-5는 40% 빠르다.", "의미가 크다."), item("Gemini도 나왔다.", "")],
        )
        assert any("already in items[0]" in hit for hit in hits)


class TestLeadFigureRepeats:
    """The 'any novel specific' escape hatch let the shipped lead through: three novel specifics
    bought a pass while 15 of its 18 were repeats, and the root Threads post re-told reply 1."""

    def test_a_repeated_figure_is_flagged_even_when_the_lead_adds_new_ones(self):
        hits = lead_figure_repeats("점유율은 52%다. 신규 계약은 6700건이다.", "점유율 52.2%를 기록했다.")
        assert hits and "52" in hits[0]

    def test_the_escape_hatch_no_longer_covers_a_repeated_number(self):
        hits = lint_digest_prose(
            "점유율 52%가 말하는 건 Gemini의 부재다.",
            [item("점유율은 52.2%다.", "의미가 크다.")],
        )
        assert any("repeats items[0]'s figure" in hit for hit in hits)

    def test_a_lead_that_names_its_own_figures_passes(self):
        assert lead_figure_repeats("계약은 6700건이다.", "점유율은 52%다.") == []

    def test_a_qualitative_lead_passes(self):
        assert lead_figure_repeats("의미는 속도가 아니라 배포다.", "점유율은 52%다.") == []


class TestItemLength:
    """The renderer amputates whatever does not fit 500 chars and only logged a count; on
    digest_2026-07-12 that cost 2 of 5 posts their closing sentence."""

    def test_prose_over_its_own_budget_is_flagged(self):
        hits = lint_digest_prose("리드다.", [item("본" * 200, "시사점이다.", title="제목", budget=100)])
        assert hits and "items[0]" in hits[0] and "100-char budget" in hits[0]

    def test_prose_within_the_budget_is_fine(self):
        assert lint_digest_prose("리드다.", [item("본문이다.", "시사점이다.", title="제목", budget=100)]) == []

    def test_no_budget_means_nothing_to_check_against(self):
        assert lint_digest_prose("리드다.", [item("본" * 900, "시사점이다.", title="제목")]) == []


class TestCleanProse:
    def test_a_clean_digest_has_no_hits(self):
        assert (
            lint_digest_prose(
                "Anthropic이 Claude를 공개했다. 관건은 배포 속도다.",
                [item("Claude가 나왔다.", "값이 문제다."), item("다른 소식이다.", "지켜볼 일이다.")],
            )
            == []
        )

    def test_no_items_still_checks_the_lead(self):
        assert lint_digest_prose("깨끗한 리드다.", []) == []
