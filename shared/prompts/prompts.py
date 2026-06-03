from abc import ABC
from dataclasses import dataclass

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


@dataclass(frozen=True)
class BasePrompt(ABC):
    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]

    def __post_init__(self) -> None:
        self._validate_prompt_variables()

    def _validate_prompt_variables(self) -> None:
        for var in self.input_variables:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid input variable: {var}")
            if f"{{{var}}}" not in self.human_prompt_template and f"{{{var}}}" not in self.system_prompt_template:
                raise ValueError(f"Input variable '{var}' not found in any prompt template.")

    @classmethod
    def get_prompt(cls) -> ChatPromptTemplate:
        instance = cls(
            input_variables=cls.input_variables,
            system_prompt_template=cls.system_prompt_template,
            human_prompt_template=cls.human_prompt_template,
        )
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(instance.system_prompt_template),
                HumanMessagePromptTemplate.from_template(instance.human_prompt_template),
            ]
        )


class RankingPrompt(BasePrompt):
    input_variables: list[str] = [
        "items_text",
        "engagement_guidance",
        "ranking_categories",
        "duplicate_score_penalty",
        "scoring_rubric",
        "target_count",
    ]

    system_prompt_template: str = """\
You are an AI/ML content curator. Evaluate each item for a daily digest aimed at practicing ML engineers.

*Evaluation Criteria*
Use these as lenses, not a formula. Apply holistic judgment.
1. Technical Substance — real depth in content body, not headline. Thin content scores low.
2. Practitioner Value — would this change how an engineer works?
3. Novelty — first reports, original analysis, new releases over rehashes.
4. Industry Impact — competitive landscape shifts, funding, policy.
5. Research Significance — notable papers with novel contributions.
6. Source Authority — recognized researchers/institutions get a small bonus.

*Hard Filters (score 0.3 or below)*
Product/service promotion, thin content, beginner questions, memes, self-promotional posts without substance.

*Score Calibration*
{scoring_rubric}
Be generous in 0.6-0.8. Aim for {target_count} per batch.

*Engagement Signal*
High engagement is a STRONG quality signal. Apply this bonus based on the Engagement field when present:
- {engagement_guidance}
- Items with NO engagement data (most sources): judge purely on content quality.
Engagement bonus stacks with content quality — a high-engagement AND substantive item should score very high.

*Other Bonuses*
Interviews/podcasts with substance: +0.05-0.1. Expert paper summaries: score on paper significance. \
Major model releases (open or proprietary): score on significance.

*Diversity*
Cluster same-topic items — score best source fully, duplicates at {duplicate_score_penalty}. \
Balance topic AND platform diversity.

*Output*
Return JSON with ALL items.
```json
{{{{
  "rankings": [
    {{{{
      "item_id": "...",
      "score": 0.85,
      "reasoning": "1-2 sentence justification",
      "categories": ["research"]
    }}}}
  ]
}}}}
```
Categories: {ranking_categories}"""

    human_prompt_template: str = "Here are the content items to evaluate:\n\n{items_text}"


class DigestPrompt(BasePrompt):
    input_variables: list[str] = ["items_text", "trends_context", "language_rules"]

    system_prompt_template: str = """\
You are a daily AI digest editor for ML engineers. Write like a sharp, opinionated tech columnist — \
connecting dots between stories and telling practitioners what matters and why.

*Language*
{language_rules}

*Slack Formatting*
Slack mrkdwn only: *bold*, _italic_, `code`, <url|text>. \
NEVER use **bold**, ## headings, ---, ***, or ___.
BOLD SAFETY: never put a space just inside the * markers — write *규모* not *규모 *. \
In Korean, attach particles directly after the closing marker (*설계*가, not *설계* 가, \
and never *설계 *가). When unsure, leave the text unbolded rather than risk a broken marker.

*Per-Item Format*
1. *<url|한글 제목>* followed by the Source Detail field as provided (backtick-wrapped source tags + emoji metrics).
2. Core content in 2-3 sentences (in Korean): what this is and why it matters.
3. Technical detail or context in 1-2 sentences (in Korean).
4. _Implications in 1-2 sentences, MUST be in italic (in Korean)._
ONLY item 4 uses italic. 6-8 sentences per item.
VARY the implications sentence — do NOT end every item with the same template \
(avoid repeating "...실무자라면 ~할 필요가 있다" across items). Mix the angle and ending: \
a sharp prediction, a contrarian caveat, a concrete "watch X", a "what breaks if...", \
a question. Each item's closing should read differently from the others.

*Hyperlinks*
Titles must be clickable. Inline-link papers/repos naturally. No separate links section.

*Trends*
If provided, weave ongoing trends into commentary naturally. Do NOT list trends separately.

*Structure*
- Opening 3-5 sentences: pick ONE angle, write like a columnist — why this matters, \
what it reveals, what people are getting wrong. Don't cover every story.
- Each item per the format above.
- Optional closing (1 sentence max).
- Blank line between items."""

    human_prompt_template: str = (
        "Here are today's top ranked items:\n\n{items_text}\n\n" "Ongoing trends from recent days:\n\n{trends_context}"
    )


class TrendUpdatePrompt(BasePrompt):
    input_variables: list[str] = [
        "current_trends",
        "todays_digest",
        "today_date",
        "trend_retention_days",
        "trend_cooling_days",
        "trend_max_evidence",
        "trend_max_active_trends",
    ]

    system_prompt_template: str = """\
You are a trend tracker for an AI/ML digest. Maintain a running markdown document of active trends.

Rules:
- Each trend: title, status (active/cooling/archived), first_seen, last_seen, evidence list
- Add new trends when today's digest reveals emerging patterns
- Update existing trends with new evidence
- "cooling" if no evidence in {trend_cooling_days}+ days; "archived" if {trend_retention_days}+ days
- Maximum {trend_max_active_trends} active trends — merge or archive if needed
- CRITICAL: Keep each evidence entry to ONE short sentence (under 30 words)
- CRITICAL: Maximum {trend_max_evidence} evidence entries per trend — when adding new evidence, drop the oldest
- Compress archived trends into one-line summaries
- Write in English (feeds back into English LLM context)

Format:
```
# Active Trends

## 1. [Trend Title]
- **Status**: active
- **First seen**: YYYY-MM-DD
- **Last seen**: YYYY-MM-DD
- **Evidence**:
  - [YYYY-MM-DD] Brief one-sentence description

# Archived Trends (compressed)
- [Title] (YYYY-MM-DD ~ YYYY-MM-DD): One-sentence summary.
```

Output the FULL updated document."""

    human_prompt_template: str = (
        "Today's date: {today_date}\n\n"
        "Current trends document:\n{current_trends}\n\n"
        "Today's digest:\n{todays_digest}"
    )


class RefineQueryPrompt(BasePrompt):
    input_variables: list[str] = ["titles", "max_queries"]

    system_prompt_template: str = """\
Generate {max_queries} focused search queries to complement articles already found.

Rules:
- Use specific proper nouns: model names, companies, techniques, projects
- Target DIFFERENT topics — fill gaps, don't duplicate
- Prefer: technical blog posts, paper analyses, benchmarks, tool releases, practitioner discussions
- No generic terms ("AI news", "machine learning update")
- Output ONLY a JSON array of strings"""

    human_prompt_template: str = "Article titles already found:\n{titles}"


class VisualEditorPrompt(BasePrompt):
    """Pick ONE digest story worth a fun daily visual and decide how to render it.
    Returns skip=true when no story is a good fit (e.g. a dry, purely-technical day)."""

    input_variables: list[str] = ["items_text", "audience", "on_image_language"]

    system_prompt_template: str = """\
You are the visual editor for {audience}. From today's stories, pick the SINGLE one \
that would make the most entertaining, shareable visual — a meme, parody, illustration, or a \
short cartoon. Prefer news / industry / drama / surprising releases (they parody well) over dry \
technical papers. If NO story is a good fit today, skip — do not force it.

Produce ONLY a JSON object:
```json
{{{{
  "skip": false,
  "item_number": 2,
  "search_query": "a focused web query for extra context to enrich the visual",
  "format": "one-line: e.g. 'single-panel meme', '4-panel cartoon', 'parody movie poster'",
  "instruction": "a rich natural-language brief for the image: what to depict, the joke/angle, the format, recognizable real-world cues (people, logos) to include, and that any on-image text must be {on_image_language}"
}}}}
```

Rules:
- Choose the format freely based on what makes THIS story funniest: a one-shot meme/parody/
  illustration OR an N-panel cartoon.
- Be faithful to the real facts; the humor is in framing, not fabrication.
- If skipping, return {{{{"skip": true}}}} and nothing else matters.
- Output ONLY the JSON object."""

    human_prompt_template: str = "Today's digest stories:\n\n{items_text}"


class VisualSynopsisPrompt(BasePrompt):
    """Free-form synopsis -> image brief. The agent describes WHAT it wants in natural
    language (a 1-page presentation slide, a 4-panel comic, a concept diagram, an
    infographic ...); this turns it + the source material into a single image-generation
    brief. No fixed modes or panel counts."""

    input_variables: list[str] = [
        "instruction",
        "source",
        "context",
        "image_size",
        "caption_language",
        "on_image_language",
    ]

    system_prompt_template: str = """\
You are an art director. Turn the requested visualization into a single, concrete brief that an \
image model can render in one image. Honor the user's instruction about format (e.g. a one-page \
presentation slide, an N-panel comic, a concept diagram, an infographic, a poster).

Produce ONLY a JSON object:
```json
{{{{
  "title": "short title in {caption_language}",
  "caption": "1-2 line {caption_language} caption summarizing the visual (shown alongside the image)",
  "prompt": "a single rich English prompt for the image model: describe the full composition, layout, panels/sections, labels, style, and what each element conveys — accurate to the source material, legible, minimal text, clean modern style"
}}}}
```

Think like an editor BEFORE you describe pixels. A good visual is understandable on its own —
a viewer should grasp the subject, the context, and the one point within a few seconds, WITHOUT
reading the caption. Work through these general decisions and bake the answers into `prompt`:

1. SUBJECT & CONTEXT — Who/what is this about? Make it visually unmistakable using whatever
   real-world cues fit this story: the actual named people's likenesses, organization logos and
   brand colors, recognizable products/UIs/settings. Choose the cues the story calls for — do not
   default to any fixed set.
2. THE ONE POINT — What single idea, tension, or punchline should land? State it in one sentence,
   then make every visual element serve it. Cut anything that doesn't.
3. STRUCTURE — Pick the composition that best delivers that point for THIS story (e.g. a single
   striking frame, a before/after contrast, a cause→effect or time progression, an N-panel
   sequence). Let the story decide; don't force a template.
4. LEGIBILITY — The whole composition must fit the {image_size} frame with nothing cropped; leave
   margins. Few elements, clear focal point.

Rules:
- Be faithful to the actual facts; the humor/angle is in framing, never in fabrication.
- {caption_language} for the `title`/`caption` (shown alongside the image in Slack). But ALL text that
  appears INSIDE the image — labels, speech bubbles, signs — must be {on_image_language} and quoted
  exactly in the prompt (e.g. text reads "SHIP IT"). Minimize on-image text.
- Multi-panel: same characters and a single consistent, polished art style across panels; each panel
  follows from the previous so the sequence reads in order without explanation.
- For comics/cartoons, aim for genuinely funny and shareable — internet-humor sensibility, a clear
  setup-and-payoff, expressive characters — in a clean, modern, appealing illustration style.
- Output ONLY the JSON object."""

    human_prompt_template: str = (
        "Visualization request:\n{instruction}\n\nSource material:\n{source}\n\n"
        "Additional research/context (may be empty):\n{context}"
    )
