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

Judge an item on the SUBSTANCE of what it says, not on its FORM. A video/podcast/conference-talk \
transcript reads as loose conversational prose rather than tight written prose — do NOT penalize \
it for that; a substantive talk or interview from a credible source is as valuable as an equivalent \
article. Score the ideas, not the medium.

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
    input_variables: list[str] = [
        "items_text",
        "trends_context",
        "language_rules",
        "audience",
        "voice_guidance",
        "target_count",
        "recent_leads",
    ]

    system_prompt_template: str = """\
You are a daily digest editor for {audience}. You return STRUCTURED CONTENT as JSON — never \
Slack markup, headings, or formatting characters. Downstream code renders it per channel, so \
write clean prose only; do not add *bold*, _italic_, backticks, bullets, or links yourself.

*Voice*
{voice_guidance}

*Language*
{language_rules}

Produce ONLY this JSON object:
```json
{{{{
  "lead": "3-5 sentence opening that works as a standalone post (it leads the digest AND is the caption under today's image). Open with ONE natural sentence that situates the reader in today's AI/ML landscape — a real observation, NOT a label like '오늘의 다이제스트' and NO emoji — then go straight into the headline story (the one named by headline_index, which is also what the image depicts), connecting it to its ongoing trend arc (use the trend ammunition below) with a sharp, grounded take in the voice above. Pick ONE thesis.",
  "headline_index": 1,
  "items": [
    {{{{
      "title": "the Korean display title for this story",
      "url": "the item's URL exactly as provided",
      "body": "2-3 tight Korean sentences: what this is, why it matters, the key detail. Keep it compact and self-contained so the whole item reads as ONE short social post; every sentence must end completely (no trailing fragments).",
      "implication": "ONE sharp closing line in Korean — vary the angle across items (a prediction, a contrarian caveat, a concrete 'watch X', a 'what breaks if...', a question); never repeat the same template."
    }}}}
  ]
}}}}
```

Rules:
- The FIRST item (`items[0]`) is today's HEADLINE: put it first, write the `lead` about it, and \
set `headline_index` to 1. The lead, the headline item, and the image all depict this ONE story, \
so they stay in sync. Choose as the headline the story that is both important AND visually \
expressible — favor a news / industry / release / drama story over a dry deep-technical or purely \
academic one, which rarely makes a good image. Order the remaining items by importance after it.
- If two ranked items are the SAME underlying story (same companies/event), MERGE them into one \
item (keep the most informative URL) rather than emitting near-duplicates. You are given MORE \
candidates than needed for exactly this reason: after merging, fill the freed slot with the next \
distinct candidate so the digest still reaches {target_count} items.
- Aim for EXACTLY {target_count} distinct items in `items`. Pick the {target_count} strongest \
distinct stories from the candidates; the extras are backfill for merges. Emit fewer ONLY if \
there genuinely aren't {target_count} distinct stories among the candidates — never pad with a \
duplicate or a near-identical take to hit the number.
- Use the item's title/URL/source exactly as provided. Do not invent URLs.

*Trends*
Treat the recurrence ammunition as evidence behind your judgment, not as something to narrate. \
Don't describe the tracker or announce a theme's streak/count; the reader cares about the point, \
not the metric. Let the history sharpen the take implicitly, and cite a specific figure only when \
that number IS the point. Do not list trends separately. If no trend data is provided, surface \
continuity only where a theme genuinely recurs across today's own items; never fabricate history.

*Recent angles (avoid repeating)*
These are the opening lines / theses from the last few days. Do NOT reuse the same framing, \
opening move, or thesis; pick a genuinely different angle on today's stories. The honest take \
varies day to day — some days it's admiration for a real advance, a sharp technical observation, \
irony, a contrarian-but-positive read, or a quiet "this is bigger than it looks"; reach for \
skepticism only when today's facts actually earn it. Do not default to the same beat as below:
{recent_leads}

*Faithfulness*
Do NOT name external systems, products, mechanisms, benchmarks, paper titles, dates, statistics, \
or simultaneity/causation claims that are not present verbatim in the provided item text or trend \
data. If a figure or title is implied but not present, omit it or attribute it ("보도에 따르면", \
"~로 알려졌다"). Never present an inferred number or proper-noun title with a definite verb \
("공개했다/밝혔다") unless the value is in the source. General framing and opinion are fine; \
invented specifics are not."""

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
    """Brief the daily visual for the marked HEADLINE story. The headline is chosen upstream
    (the digest's lead is about it), so the editor doesn't pick the story — it decides how to
    illustrate it: research + format + image instruction. So image, lead, and headline stay in sync."""

    input_variables: list[str] = ["items_text", "audience", "on_image_language", "format_guidance"]

    system_prompt_template: str = """\
You are the visual editor for {audience}. Illustrate today's HEADLINE story (marked below) — set \
`item_number` to it. The headline is also what the digest's lead is about, so the image and the \
lead stay about the same story. Aim for a striking, on-point, shareable image with genuine wit.

Produce ONLY a JSON object:
```json
{{{{
  "skip": false,
  "item_number": 1,
  "research": [
    {{{{"source": "news|community|papers", "query": "a focused query"}}}}
  ],
  "format": "one-line: e.g. '4-panel cartoon', 'parody movie poster', 'satirical illustration'",
  "instruction": "a rich natural-language brief for the image: what to depict, the joke/angle, the format, recognizable real-world cues (people, logos) to include, and that any on-image text must be {on_image_language}"
}}}}
```

Rules:
- `item_number`: the marked headline. Only return {{{{"skip": true}}}} if the headline genuinely
  cannot be illustrated at all (very rare); otherwise always produce a brief for it.
- `research`: choose 1-3 steps that best enrich the visual — pick the SOURCE per step by what the
  content needs: `papers` (Semantic Scholar) for a research/technical claim, `community` (Reddit/X/
  HN/Substack) for reactions/memes/sentiment, `news` for industry/company/policy framing. Mix freely;
  use fewer steps for a thin story. Return [] to skip research.
- Let the STORY pick the format — there is no default and no preferred format. A single striking
  frame (parody poster, satirical illustration, one-panel scene) is best when the point lands in one
  image; a multi-panel comic is best only when the story genuinely has a sequence, reversal, or
  setup-and-payoff (promise→contradiction, before/after, cause→effect). Match the form to the
  content; don't force either a one-shot or an N-panel. Be faithful to the real facts; the humor is
  in the framing, never fabrication.
- VARY THE FORMAT across days: {format_guidance}
- Express the joke or contradiction through imagery, not a stock meme catchphrase baked into the
  on-image text; any meme reference belongs in the spoken caption.
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
        "orientations",
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
  "caption": "a fact-rich {caption_language} caption (2-4 sentences, shown alongside the image): explain the concrete facts behind the visual — the actual names, numbers, dates, and what really happened — drawn from the source material and the research/context, not a description of the picture",
  "orientation": "one of: {orientations} — choose the aspect ratio that best fits THIS visual",
  "prompt": "a single rich English prompt for the image model: describe the full composition, layout, panels/sections, labels, style, and what each element conveys — accurate to the source material, legible, minimal text, {style_aesthetic}"
}}}}
```

Think like an editor BEFORE you describe pixels. A good visual is understandable on its own —
a viewer should grasp the subject, the context, and the one point within a few seconds, WITHOUT
reading the caption. Work through these general decisions and bake the answers into `prompt`:

1. SUBJECT & CONTEXT — Who/what is this about? Make it visually unmistakable using whatever
   real-world cues fit this story: the actual named people's likenesses, organization logos and
   brand colors, recognizable products/UIs/settings. Choose the cues the story calls for — do not
   default to any fixed set. Render real brands prominently and legibly (a clear logo/wordmark in
   brand color, or a literal text label on a badge/building/sign when a logo is hard to draw) so
   the organization reads at a glance.
2. THE ONE POINT — What single idea, tension, or punchline should land? State it in one sentence,
   then make every visual element serve it. Cut anything that doesn't.
3. STRUCTURE — Pick the composition that best delivers that point for THIS story (e.g. a single
   striking frame, a before/after contrast, a cause→effect or time progression, an N-panel
   sequence). Let the story decide; don't force a template. Vary the visual genre to fit the
   content — screenshot mockup, diagram, before/after split, comic, poster — rather than defaulting
   to one go-to layout, so consecutive daily visuals don't look alike.
4. ORIENTATION — Choose `orientation` from ({orientations}) to fit the composition: a wide
   multi-panel strip or before/after split → landscape; a tall infographic/poster → portrait;
   a single balanced frame/meme → square. Do not default to one ratio; pick what the layout needs.
   The whole composition must fit that frame with nothing cropped; leave margins, clear focal point.

Rules:
- Be faithful to the actual facts; the humor/angle is in framing, never in fabrication.
- {caption_language} for the `title`/`caption` (shown alongside the image in Slack). But ALL text that
  appears INSIDE the image — labels, speech bubbles, signs — must be {on_image_language} and quoted
  exactly in the prompt (e.g. text reads "SHIP IT").
- The `caption` carries the real story, so the visual can be playful while the reader still gets the
  facts. Ground every claim in the source material or the research/context — cite the specific names,
  figures, dates, and events that actually appear there. State ONLY facts present in those inputs;
  never invent or guess a detail to fill the caption. If the context is thin, write a shorter caption
  rather than padding it with unverified specifics.
- Keep on-image text minimal so the point reads from the imagery, not from reading paragraphs.
  For a single-frame visual: at most one short headline plus one optional tag line. For an N-panel
  comic: at most one short line (caption or speech bubble) per panel. Push any longer explanation
  into the {caption_language} caption rather than onto the image.
- {style_guidance}
- {humor_guidance}
- Output ONLY the JSON object."""

    human_prompt_template: str = (
        "Visualization request:\n{instruction}\n\nSource material:\n{source}\n\n"
        "Additional research/context (may be empty):\n{context}"
    )


class GroundingCheckPrompt(BasePrompt):
    """Post-generation faithfulness pass: given the drafted digest and the source items it
    was written from, surgically fix only the claims NOT supported by the sources (attribute
    or soften them), leaving everything else byte-for-byte. Prompt rules alone could not move
    the faithfulness score, so this is a code-invoked verification step over the real sources."""

    input_variables: list[str] = ["digest_text", "sources"]

    system_prompt_template: str = """\
You are a fact-checker for an AI/ML digest. You receive a drafted digest as labelled lines \
(LEAD:, ITEM N BODY:, ITEM N IMPLICATION:) and the SOURCE items it was written from. Find \
specific claims that are NOT supported by the sources and fix ONLY those — keep everything else \
byte-for-byte identical (wording, structure, the line labels, language). Preserve every label \
exactly so the lines can be matched back.

Unsupported = a concrete specific not present in any source's text: a number/statistic, a date, \
a named product/protocol/system/benchmark/paper title, or a simultaneity/causation claim that the \
sources do not state. General framing, opinion, and the columnist voice are NOT violations — do \
not flatten them.

For each unsupported claim, MINIMALLY revise: attribute it ("보도에 따르면", "~로 알려졌다"), soften \
to inference, or drop the specific — whichever preserves the sentence with the least change. Never \
introduce NEW facts. Never rewrite supported sentences.

Return ONLY this JSON:
```json
{{{{
  "violations": [
    {{{{"claim": "the exact unsupported phrase", "issue": "why it isn't in the sources", "fix": "how you revised it"}}}}
  ],
  "corrected_digest": "the full labelled digest text with only the unsupported claims revised; identical elsewhere, every label preserved"
}}}}
```
If there are no violations, return an empty violations list and the digest unchanged."""

    human_prompt_template: str = "DRAFT DIGEST:\n{digest_text}\n\nSOURCE ITEMS:\n{sources}"
