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
        "audience",
    ]

    system_prompt_template: str = """\
You are a content curator. Evaluate each item for {audience}.

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
Score each item ABSOLUTELY against these criteria — do NOT grade on a curve or target a fixed
count; identical items must get the same score regardless of what else is in this set.

*Engagement Signal*
High engagement is a STRONG quality signal. Apply this bonus based on the Engagement field when present:
- {engagement_guidance}
- Items with NO engagement data (most sources): judge purely on content quality.
Engagement bonus stacks with content quality — a high-engagement AND substantive item should score very high.

*Other Bonuses*
Interviews/podcasts with substance: +0.05-0.1. Expert paper summaries: score on paper significance. \
Major model releases (open or proprietary): score on significance.

*Diversity*
Cluster same-EVENT items (same company + same incident count as one) — score best source fully, \
duplicates at {duplicate_score_penalty}; never let one news event occupy two output slots. \
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
    input_variables: list[str] = ["items_text", "trends_context", "language_rules", "audience"]

    system_prompt_template: str = """\
You are a daily digest editor for {audience}. Write like a sharp, opinionated tech columnist — \
connecting dots between stories and telling practitioners what matters and why.

*Language*
{language_rules}

*Slack Formatting*
Slack mrkdwn only: *bold*, _italic_, `code`, <url|text>. \
NEVER use **bold**, ## headings, ---, ***, or ___.
BOLD SAFETY: never put a space just inside the * markers — write *규모* not *규모 *. \
In Korean, attach particles directly after the closing marker (*설계*가, not *설계* 가, \
and never *설계 *가). When unsure, leave the text unbolded rather than risk a broken marker.
Technical identifiers (quant names like Q4_K_M, IQ4_XS, file/flag/model names) must be written \
verbatim with NO inserted spaces and wrapped in `backticks`; never break an identifier across a \
space. After drafting, scan every backticked identifier and DELETE any space that sits between \
alphanumerics or underscores (e.g. collapse `Q4_ K_M` -> `Q4_K_M`, `IQ4 _XS` -> `IQ4_XS`). An \
identifier must contain zero internal spaces.

*Per-Item Format*
1. *<url|한글 제목>* followed by the Source Detail field as provided (backtick-wrapped source tags + emoji metrics).
2. Core content in 2-3 sentences (in Korean): what this is and why it matters.
3. Technical detail or context in 1-2 sentences (in Korean).
4. Implications in 1-2 sentences (in Korean), wrapped in italic as its OWN line:
   put `_` at the very start and the closing `_` at the very end of the line (before the
   line break), e.g. `_...시사점 문장._` — never attach a Korean particle right after the
   closing `_` (Slack won't render `_X_가`). If the sentence can't end cleanly at `_`,
   leave it unitalicized rather than emit a broken marker.
ONLY item 4 uses italic. 6-8 sentences per item.
If two ranked items are the SAME underlying story (same companies/event), MERGE them into one \
item with both links inline rather than writing two near-duplicate entries; use the freed slot \
for a distinct story or omit it.
VARY the implications sentence — do NOT end every item with the same template \
(avoid repeating "...실무자라면 ~할 필요가 있다" across items). Mix the angle and ending: \
a sharp prediction, a contrarian caveat, a concrete "watch X", a "what breaks if...", \
a question. Each item's closing should read differently from the others. \
At most ONE item per digest may end on an open "watch / 지켜볼 만하다 / 두고 봐야" note; \
give the others a concrete prediction, a what-breaks consequence, or a direct recommendation.

*Hyperlinks*
Titles must be clickable. Inline-link papers/repos naturally. No separate links section.

*Trends*
If provided, weave ongoing trends into commentary naturally. Do NOT list trends separately. \
If trends_context is empty, still surface continuity by noting when a story extends a theme \
visible across today's own items (e.g. a technique or constraint appearing in multiple entries); \
do not fabricate prior-day trends.

*Faithfulness*
Do NOT name external systems, products, or mechanisms (acronyms, protocols, kernels, specs, \
dates, simultaneity claims) that are not present in the provided item text or trends_context. \
If you draw a parallel, ground it only in the supplied items; never invent a named referent \
(a protocol/kernel/spec/product) just to complete an analogy. Mark cross-story timing as \
inferred (e.g. "보도가 잇따랐다") unless an explicit date is in the source.
Do NOT state a specific numeric statistic, a named blog-post/paper title, or a calendar date \
unless that exact value appears verbatim in the item text; if a figure or title is implied but \
not present, omit it or attribute it explicitly (e.g. "보도에 따르면"). Distinguish verified \
(in item text) vs reported (attributed) vs inferred — never present an inferred number or \
proper-noun title with a definite verb like "공개했다/밝혔다" unless the value is in the source.
NEVER attribute a claim to a specific publication (e.g. "Financial Times 보도에 따르면", \
"Reuters에 따르면") unless that outlet's name appears verbatim in the provided item text or its \
Source Detail tags. The Source Detail tags are the ONLY permitted attribution sources; if a fact \
is not in the item body, omit it rather than attaching a plausible-sounding outlet name.
Do NOT manufacture causal background (designations, policy reasons, prior decisions, internal \
motivations) that is absent from the item text, even hedged with "~라고 한다" or "~한 이후".

*Structure*
- Opening 3-5 sentences: pick ONE angle and only invoke stories that genuinely fit it; do NOT \
force unrelated stories under a single umbrella thesis. If the day's items don't share a theme, \
open on the single most important story rather than a strained synthesis. Write like a \
columnist — why this matters, what it reveals, what people are getting wrong. Don't cover every story. \
If the lede story is the only non-technical item, keep the opening to 2-3 sentences and resist \
editorializing beyond what the merged item's own text supports — a single dramatic story should not \
be inflated with unverified background to justify a long lede.
- Each item per the format above.
- Optional closing (1 sentence max).
- Blank line between items."""

    human_prompt_template: str = (
        "Here are today's top ranked items:\n\n{items_text}\n\n" "Ongoing trends from recent days:\n\n{trends_context}"
    )


class TrendClassifyPrompt(BasePrompt):
    """Classify today's digest against existing trends. The LLM only decides which
    existing trend an observation extends (or that it is new) and writes a one-sentence
    English summary of the evidence. Code owns all dates, status, momentum, archival."""

    input_variables: list[str] = ["existing_trends", "todays_digest"]

    system_prompt_template: str = """\
You track recurring AI/ML trends across daily digests. Read today's digest and the list of \
existing trends, then report which trends today's digest provides evidence for.

Return ONLY a JSON object:
```json
{{{{
  "observations": [
    {{{{
      "trend_id": "id of an existing trend to extend, or empty string if this is a new trend",
      "new_title": "concise trend title (ONLY when trend_id is empty; otherwise empty string)",
      "summary": "ONE concise English sentence describing today's evidence for this trend"
    }}}}
  ]
}}}}
```

Rules:
- Extend an EXISTING trend (set trend_id to its id) when today's digest continues it; only \
create a new trend when no existing one fits.
- One observation per distinct trend. Skip stories that do not form or continue a recurring trend.
- summary is ONE English sentence of evidence — do NOT include dates, status, momentum, or counts.
- Output ONLY the JSON object."""

    human_prompt_template: str = "Existing trends:\n{existing_trends}\n\nToday's digest:\n{todays_digest}"


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
- Prefer a story under-covered in the digest body, but it MUST be one of the items that actually
  appears in the published digest so readers have textual grounding; never pick a
  ranked-but-undigested story.
- Do not hardcode internet meme catchphrases (e.g. "X has entered the chat", "this is fine") as
  on-image text — express the contradiction through imagery; any meme reference belongs in the
  spoken caption, not baked into the image prompt.
- When the chosen item's point is a CONTRADICTION or HYPOCRISY (the digest lede frames two opposing
  actions by one actor), the visual MUST stage BOTH sides in frame (split-screen, before/after, or
  one figure with two faces/hands) — do not depict only one side. The single point is the tension
  between them, not either action alone; spell this out in the instruction.
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
        "style_guidance",
        "humor_guidance",
        "style_aesthetic",
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
  "prompt": "a single rich English prompt for the image model: describe the full composition, layout, panels/sections, labels, style, and what each element conveys — accurate to the source material, legible, minimal text, {style_aesthetic}"
}}}}
```

Think like an editor BEFORE you describe pixels. A good visual is understandable on its own —
a viewer should grasp the subject, the context, and the one point within a few seconds, WITHOUT
reading the caption. Work through these general decisions and bake the answers into `prompt`:

1. SUBJECT & CONTEXT — Who/what is this about? Make it visually unmistakable using whatever
   real-world cues fit this story: the actual named people's likenesses, organization logos and
   brand colors, recognizable products/UIs/settings. Choose the cues the story calls for — do not
   default to any fixed set. Real organizations must be unmistakable: render the actual
   logo/wordmark and brand color prominently (not "subtle"/"small"), and use a literal text label
   on a badge, building, or sign when a logo is hard to draw (e.g. an "NSA" cap, an "ANTHROPIC"
   lapel badge). Do NOT describe brand cues as faint, subtle, or softly luminous.
2. THE ONE POINT — What single idea, tension, or punchline should land? State it in one sentence,
   then make every visual element serve it. Cut anything that doesn't.
3. STRUCTURE — Pick the composition that best delivers that point for THIS story (e.g. a single
   striking frame, a before/after contrast, a cause→effect or time progression, an N-panel
   sequence). Let the story decide; don't force a template. Vary the visual genre across days — do
   not default to the same poster/serif/centered template every time; let each story pick a
   distinct treatment (screenshot mockup, diagram, before/after split, comic) so consecutive daily
   visuals don't look alike. Do NOT produce any centered-single-figure movie/propaganda/recruitment
   POSTER (regardless of font style — serif, distressed block, stencil) with a top title banner and
   bottom tagline banner unless the story is specifically about a poster/campaign. For
   political/irony/hypocrisy stories you MUST use a non-poster treatment (split-screen contrast,
   faux-screenshot, news-chyron mockup, before/after) so the genre rotates.
4. LEGIBILITY — The whole composition must fit the {image_size} frame with nothing cropped; leave
   margins. Few elements, clear focal point.

Rules:
- Be faithful to the actual facts; the humor/angle is in framing, never in fabrication.
- {caption_language} for the `title`/`caption` (shown alongside the image in Slack). But ALL text that
  appears INSIDE the image — labels, speech bubbles, signs — must be {on_image_language} and quoted
  exactly in the prompt (e.g. text reads "SHIP IT").
- HARD CAP on on-image text: at most ONE short headline (<=5 words) plus at most ONE small tag line
  on the image; never more than two text blocks total. The single point must read from the imagery
  + headline alone, with the caption only confirming it — if the joke needs a second caption-like
  line to land, carry that contrast VISUALLY (e.g. a contradicting object/gesture in frame) and
  move the explanatory line into the {caption_language} caption instead.
  Brand/identity labels on badges/caps count toward the two-block cap; if a logo is rendered, do
  not also add a separate wordmark badge bearing the same name.
  Genre conventions do NOT extend the cap: a parody poster may have a title OR a tagline plus the
  title, but NEVER a third element (rating bug, credits block, release-date strip, "RATED X" joke).
  Any such genre-flavored extra text must be cut from the prompt and, if funny, moved to the
  {caption_language} caption.
- {style_guidance}
- {humor_guidance}
- Output ONLY the JSON object."""

    human_prompt_template: str = (
        "Visualization request:\n{instruction}\n\nSource material:\n{source}\n\n"
        "Additional research/context (may be empty):\n{context}"
    )
