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
