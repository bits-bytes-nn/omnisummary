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
        assert "NAMING that story" in rendered
        # The generic "situate the reader in today's AI/ML landscape" opener is gone.
        assert "situates the reader" not in rendered

    def test_re_narration_ban_no_longer_forbids_naming(self):
        # The ban is narrowed to what actually duplicated the headline reply — replaying its
        # sequence of events and repeating its numbers — so the lead can still say WHO it is about.
        rendered = _render(DigestPrompt)
        assert "no replaying its sequence of events and no repeating its numbers" in rendered
        assert "numbers, names, or its sequence of events" not in rendered

    def test_items_are_still_requested_before_the_lead(self):
        # Load-bearing key ORDER: asking for the stories first is what stopped the lead from
        # re-narrating the headline (word overlap 0.21-0.41 → 0.03-0.21). Never reorder.
        rendered = _render(DigestPrompt)
        assert rendered.index('"items"') < rendered.index('"lead"')

    def test_lead_carries_a_character_budget(self):
        assert "under 470 characters" in _render(DigestPrompt)


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
