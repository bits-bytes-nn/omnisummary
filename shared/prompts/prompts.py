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
    input_variables: list[str] = ["items_text"]

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
High engagement is a STRONG quality signal. Apply these bonuses based on the Engagement field:
- Reddit: 100+ upvotes → +0.05, 500+ → +0.1, 1000+ → +0.15. High comment count amplifies further.
- YouTube: 10K+ views → +0.05, 100K+ → +0.1, 500K+ → +0.15.
- Items with NO engagement data: judge purely on content quality.
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

*Per-Item Format*
1. *<url|한글 제목>* followed by the Source Detail field as provided (backtick-wrapped source tags + emoji metrics).
2. Core content in 2-3 sentences (in Korean): what this is and why it matters.
3. Technical detail or context in 1-2 sentences (in Korean).
4. _Implications in 1-2 sentences, MUST be in italic (in Korean)._
ONLY item 4 uses italic. 6-8 sentences per item.

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
    input_variables: list[str] = ["current_trends", "todays_digest", "today_date", "trend_retention_days"]

    system_prompt_template: str = """\
You are a trend tracker for an AI/ML digest. Maintain a running markdown document of active trends.

Rules:
- Each trend: title, status (active/cooling/archived), first_seen, last_seen, evidence list
- Add new trends when today's digest reveals emerging patterns
- Update existing trends with new evidence
- "cooling" if no evidence in 7+ days; "archived" if {trend_retention_days}+ days
- Maximum 10 active trends — merge or archive if needed
- CRITICAL: Keep each evidence entry to ONE short sentence (under 30 words)
- CRITICAL: Maximum 5 evidence entries per trend — when adding new evidence, drop the oldest
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


class ComicSynopsisPrompt(BasePrompt):
    input_variables: list[str] = ["title", "content", "panels"]

    system_prompt_template: str = """\
You are a comic scriptwriter turning an AI/ML news item into a witty, accessible {panels}-panel comic \
that helps a reader grasp the core idea at a glance.

Produce a JSON object describing the comic:
```json
{{{{
  "title": "short comic title (Korean)",
  "style": "one-line visual style note (e.g. hand-drawn webcomic, warm flat colors)",
  "panels": [
    {{{{"caption": "Korean caption shown in/under the panel", "visual": "English description of what to draw"}}}}
  ]
}}}}
```

Rules:
- Exactly {panels} panel(s).
- Captions in Korean, concise and punchy. Visual descriptions in English (for the image model).
- Explain the actual technical idea — accurate, not generic. Use a clear metaphor or scene.
- Keep it friendly and clever, never cynical. No text-heavy panels.
- Output ONLY the JSON object."""

    human_prompt_template: str = "News item title: {title}\n\nContent:\n{content}"


class VisualizationBriefPrompt(BasePrompt):
    input_variables: list[str] = ["title", "content"]

    system_prompt_template: str = """\
You design a single explanatory diagram that helps a reader understand the core technical idea \
of an AI/ML news item — not a comic, but an illustrative concept visualization (e.g. a flow, \
architecture sketch, comparison, or labeled schematic).

Produce a JSON object:
```json
{{{{
  "title": "short diagram title (Korean)",
  "visual": "detailed English description of the diagram: what boxes/arrows/labels/axes to draw, layout, and what each element represents"
}}}}
```

Rules:
- One coherent diagram that captures the key mechanism or comparison accurately.
- Labels should be short English terms (model/method names) where natural.
- Clean, modern infographic style; minimal decorative clutter.
- Output ONLY the JSON object."""

    human_prompt_template: str = "News item title: {title}\n\nContent:\n{content}"
