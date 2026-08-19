import re

import pytest

from shared.prompts import (
    DigestPrompt,
    GroundingCheckPrompt,
    RankingPrompt,
    TrendClassifyPrompt,
    VisualEditorPrompt,
)
from shared.utils import parse_json_from_llm_output

# Dummy inputs for every prompt's declared template variables, so we can actually render them.
_INPUTS = {
    RankingPrompt: {
        "items_text": "x",
        "engagement_guidance": "e",
        "ranking_categories": "c",
        "duplicate_score_penalty": 0.1,
        "scoring_rubric": "s",
        "audience": "a",
    },
    DigestPrompt: {
        "items_text": "x",
        "trends_context": "t",
        "language_rules": "l",
        "audience": "a",
        "voice_guidance": "v",
        "target_count": 5,
        "recent_leads": "r",
        "recent_titles": "t",
        "prose_budget_rule": " title, body and implication together must stay under 380 characters.",
        "lead_budget": 470,
    },
    TrendClassifyPrompt: {"existing_trends": "e", "todays_digest": "d"},
    GroundingCheckPrompt: {"digest_text": "d", "sources": "s"},
    VisualEditorPrompt: {"audience": "a", "format_guidance": "f", "items_text": "x", "on_image_language": "ko"},
}


def _render(prompt) -> str:
    return "\n".join(m.content for m in prompt.get_prompt().format_messages(**_INPUTS[prompt]))


class TestPromptJsonExamplesAreValid:
    @pytest.mark.parametrize(
        "prompt",
        [RankingPrompt, DigestPrompt, TrendClassifyPrompt, GroundingCheckPrompt, VisualEditorPrompt],
    )
    def test_json_example_block_parses(self, prompt):
        # Regression for the quadruple-brace bug: LangChain f-string templates render `{{`→`{`, so
        # a JSON example must use double braces to render as single-brace VALID JSON shown to the
        # model. Quadruple braces render as `{{`/`}}` — invalid JSON the model may mirror, causing
        # unparseable output (empty rankings / 0-item digest). Assert the rendered example parses.
        rendered = _render(prompt)
        assert "{{" not in rendered and "}}" not in rendered  # no doubled braces reach the model
        m = re.search(r"```json\s*(.+?)```", rendered, re.DOTALL)
        assert m, f"{prompt.__name__} has no ```json example block"
        parse_json_from_llm_output(m.group(1))  # raises if the example is not valid JSON


class TestDigestLeadSpec:
    """The lead IS the Threads root — the one line most feed readers ever see. 40 consecutive roots
    opened with the same fixed countdown sentence, and the top-viewed posts were the ones naming a
    concrete event, so the lead's first sentence must name the day's story."""

    def test_lead_must_name_the_story_concretely(self):
        rendered = _render(DigestPrompt)
        assert "name that story in ONE assertive sentence" in rendered
        # The generic "situate the reader in today's AI/ML landscape" opener is gone.
        assert "situates the reader" not in rendered

    def test_re_narration_ban_no_longer_forbids_naming(self):
        # The ban is narrowed to what actually duplicated the headline reply — replaying its
        # sequence of events and repeating its numbers — so the lead can still say WHO it is about.
        # The five-clause paragraph that said this is now two sentences; shared/prose_lint.py checks
        # the same thing in code, which is why the rule no longer has to be spelled out at length.
        rendered = _render(DigestPrompt)
        assert "no replay of its sequence and no repeat of its numbers" in rendered
        assert "numbers, names, or its sequence of events" not in rendered

    def test_items_are_still_requested_before_the_lead(self):
        # Load-bearing key ORDER: asking for the stories first is what stopped the lead from
        # re-narrating the headline (word overlap 0.21-0.41 → 0.03-0.21). Never reorder.
        rendered = _render(DigestPrompt)
        assert rendered.index('"items"') < rendered.index('"lead"')

    def test_lead_carries_a_character_budget(self):
        assert "under 470 characters" in _render(DigestPrompt)

    def test_trend_rule_bans_the_tracker_not_the_connection(self):
        # trends_context WAS injected and then ordered to be invisible ("not as something to
        # narrate", "sharpen the take implicitly"), which banned the one move the items cannot
        # make on their own: saying where today's story sits in a longer arc. The ban is narrowed
        # to the tracker itself — the streak, the day count, the metric — so the substantive
        # connection is asked for. Faithfulness already permits it: trend data counts as provided.
        rendered = _render(DigestPrompt)
        assert "place today's story in the arc it belongs to" in rendered
        assert "no streak or day count" in rendered
        assert "not as something to narrate" not in rendered
        assert "sharpen the take implicitly" not in rendered


class TestItemSpecStaysShort:
    """Both item fields were specified by menu, and a menu invites the safest option.

    10 of 20 shipped implications were 'IF X, THEN maybe Y' conditional-modal hedges (~다면/~라면 +
    ~수 있다) — the safest of the six shapes the prompt offered. 17 of 20 bodies were exactly three
    sentences with the same '핵심은...' middle beat, because the spec asked for '2-3'. Read as a
    thread, five items scanned as one template refilled. The remedy is a SHORTER spec, which is the
    only kind of prompt change this repo allows."""

    def test_the_implication_shape_menu_is_gone(self):
        rendered = _render(DigestPrompt)
        assert "VARY THE SHAPE" not in rendered
        assert "an open question to the reader" not in rendered
        assert "a falsifiable prediction" not in rendered

    def test_the_implication_demands_an_assertion_and_bans_the_hedge(self):
        rendered = _render(DigestPrompt)
        assert "ASSERTS something a reader could disagree with" in rendered
        assert "No conditional frame, no ~수 있다" in rendered

    def test_the_body_length_is_not_a_fixed_sentence_count(self):
        rendered = _render(DigestPrompt)
        assert "2-3 tight Korean sentences" not in rendered
        assert "As few tight Korean sentences as the story needs" in rendered

    def test_the_prose_budget_still_reaches_the_body_spec(self):
        # The item budget is code-derived (_item_prose_budget); the rewrite must not drop the slot it
        # is interpolated into, or every item silently loses its Threads length guidance.
        assert "prose_budget_rule" in DigestPrompt.input_variables
        assert _INPUTS[DigestPrompt]["prose_budget_rule"].strip() in _render(DigestPrompt)


class TestDeletedDuplicateRules:
    """Rules a code mechanism already enforces are deleted, not restated: every restatement is a
    rule the editor can contradict, and this repo has a documented add-rules regression pattern."""

    def test_headline_index_is_not_asked_for(self):
        # _parse_content pins headline_index to 1 so the lead and the visual can never point at
        # different stories, which makes asking the editor for it dead weight.
        rendered = _render(DigestPrompt)
        assert "headline_index" not in rendered

    def test_visual_editor_is_not_asked_for_item_number(self):
        # The headline is marked upstream and _pick_story never reads item_number back.
        rendered = _render(VisualEditorPrompt)
        assert "item_number" not in rendered

    def test_visual_expressibility_is_only_a_tie_break(self):
        # Kept (the visual editor's skip path must stay rare) but scoped to equally important
        # stories, instead of steering the headline away from deep-technical news outright.
        rendered = _render(DigestPrompt)
        assert "break a tie between equally important ones" in rendered

    def test_faithfulness_restatement_is_gone_but_the_attribution_convention_stays(self):
        # The definite-verb sentence restated the sentence above it; the code-side grounding pass
        # enforces the same thing over the real sources.
        rendered = _render(DigestPrompt)
        assert "공개했다/밝혔다" not in rendered
        assert "보도에 따르면" in rendered


class TestRankingCostShape:
    """The ranker scores every collected item (~100/day) but only the ~8 that survive selection reach
    the digest editor, so anything it emits per item is paid for ~12x over and mostly discarded — at
    Opus output rates ($25/Mtok) that was the second-largest line in the Bedrock bill."""

    def test_reasoning_is_a_phrase_not_sentences(self):
        rendered = _render(RankingPrompt)
        assert "short phrase" in rendered
        assert "1-2 sentence justification" not in rendered

    def test_score_is_requested_before_reasoning(self):
        # Load-bearing for the change above: with `score` emitted FIRST, the justification is
        # post-hoc and cannot act as chain-of-thought, so shortening it cannot move a score. If the
        # keys are ever reordered so reasoning precedes score, shortening it becomes a quality risk.
        rendered = _render(RankingPrompt)
        assert rendered.index('"score"') < rendered.index('"reasoning"')


class TestRankingPromptCarriesNoCodeOwnedOrDoubleCountedRules:
    """Pure deletions. The medium-neutrality paragraph already says a substantive talk or interview
    scores like an equivalent article, and output slots / platform diversity are decided by code
    (_apply_source_slots, source_slots, max_per_origin) — a batch of three items cannot honour
    either, and the request contradicted the absolute-scoring instruction above it."""

    def test_no_standalone_interview_bonus(self):
        rendered = _render(RankingPrompt)
        assert "Interviews/podcasts with substance" not in rendered
        # The neutrality paragraph it double-counted stays.
        assert "Score the ideas, not the medium." in rendered

    def test_no_output_slot_or_platform_diversity_instruction(self):
        rendered = _render(RankingPrompt)
        assert "output slots" not in rendered
        assert "platform diversity" not in rendered
        # Same-event clustering — the one diversity decision the model CAN make in a batch — stays.
        assert "Cluster same-EVENT items" in rendered

    def test_absolute_scoring_is_still_the_instruction(self):
        assert "Score each item ABSOLUTELY" in _render(RankingPrompt)
