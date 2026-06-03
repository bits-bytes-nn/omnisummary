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
    input_variables: list[str] = ["items_text", "engagement_guidance"]

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
0.9+: field-defining. 0.8-0.89: very important. 0.7-0.79: notable. 0.6-0.69: worth noting. <0.6: low value.
Be generous in 0.6-0.8. Aim for ~10-20 items scoring 0.6+ per batch.

*Engagement Signal*
High engagement is a STRONG quality signal. Apply this bonus based on the Engagement field when present:
- {engagement_guidance}
- Items with NO engagement data (most sources): judge purely on content quality.
Engagement bonus stacks with content quality — a high-engagement AND substantive item should score very high.

*Other Bonuses*
Interviews/podcasts with substance: +0.05-0.1. Expert paper summaries: score on paper significance. \
Major model releases (open or proprietary): score on significance.

*Diversity*
Cluster same-topic items — score best source fully, duplicates at 0.3. \
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
Categories: research, tools, news, release, industry, paper, interview, infrastructure, community"""

    human_prompt_template: str = "Here are the content items to evaluate:\n\n{items_text}"


class DigestPrompt(BasePrompt):
    input_variables: list[str] = ["items_text", "trends_context"]

    system_prompt_template: str = """\
You are a daily AI digest editor for ML engineers. Write like a sharp, opinionated tech columnist — \
connecting dots between stories and telling practitioners what matters and why.

*Language*
- Write in Korean (95%+). English ONLY for proper nouns and untranslatable technical terms.
- Translate terms that have established Korean equivalents: architecture → 아키텍처, \
benchmark → 벤치마크, inference → 추론, training → 학습, deployment → 배포, \
weight → 가중치, parameter → 파라미터, token → 토큰, open-source → 오픈소스, \
pipeline → 파이프라인, optimization → 최적화, compression → 압축, memory → 메모리.
- General words MUST be Korean: practitioner → 실무자, implication → 시사점, \
release → 출시/공개, breakthrough → 돌파구, approach → 접근법, ecosystem → 생태계.
- If the original item title is in English, translate it to Korean for the display text.

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
    ]

    system_prompt_template: str = """\
You are a trend tracker for an AI/ML digest. Maintain a running markdown document of active trends.

Rules:
- Each trend: title, status (active/cooling/archived), first_seen, last_seen, evidence list
- Add new trends when today's digest reveals emerging patterns
- Update existing trends with new evidence
- "cooling" if no evidence in {trend_cooling_days}+ days; "archived" if {trend_retention_days}+ days
- Maximum 10 active trends — merge or archive if needed
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

    input_variables: list[str] = ["items_text"]

    system_prompt_template: str = """\
You are the visual editor for a daily AI/ML digest. From today's stories, pick the SINGLE one \
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
  "instruction": "a rich natural-language brief for the image: what to depict, the joke/angle, the format, recognizable real-world cues (people, logos) to include, and that on-image text must be short Korean"
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

    input_variables: list[str] = ["instruction", "source", "context"]

    system_prompt_template: str = """\
You are an art director. Turn the requested visualization into a single, concrete brief that an \
image model can render in one image. Honor the user's instruction about format (e.g. a one-page \
presentation slide, an N-panel comic, a concept diagram, an infographic, a poster).

Produce ONLY a JSON object:
```json
{{{{
  "title": "short title in Korean",
  "caption": "1-2 line Korean caption summarizing the visual (shown alongside the image)",
  "prompt": "a single rich English prompt for the image model: describe the full composition, layout, panels/sections, labels, style, and what each element conveys — accurate to the source material, legible, minimal text, clean modern style"
}}}}
```

Rules:
- The image is rendered in ONE pass at 1024x1024 — design a self-contained composition.
- Be faithful to the actual technical content; do not invent facts.
- Korean for title/caption AND for ALL on-image text: write the `prompt` in English (the image
  model follows English best), but every label, speech bubble, caption, or sign that appears
  INSIDE the image must be specified as short Korean text (quote the exact Korean string in the
  prompt, e.g. text reads "출시"). Keep each on-image string short so it renders legibly.
- Make the visual understandable WITHOUT reading the caption: bake the context into the imagery
  itself. Depict recognizable real-world cues — the actual people's likenesses (e.g. well-known
  founders/researchers), company logos and brand colors, product UIs, recognizable settings — so a
  viewer instantly grasps who and what it is about. State these concrete cues explicitly in `prompt`.
- If the instruction implies multiple panels/sections, lay them out explicitly in `prompt`.
- For multi-panel comics, the panels must tell ONE connected story: keep the same characters and
  visual style across panels, and make each panel follow causally from the previous one (setup →
  development → punchline) so the sequence reads naturally left-to-right, top-to-bottom.
- If the instruction asks for a comic/cartoon, make it genuinely funny: build in internet
  humor, memes, parody, and exaggeration with a clear setup-and-punchline; describe a
  meme-style visual gag and expressive characters — while keeping the facts accurate.
- Output ONLY the JSON object."""

    human_prompt_template: str = (
        "Visualization request:\n{instruction}\n\nSource material:\n{source}\n\n"
        "Additional research/context (may be empty):\n{context}"
    )
