from abc import ABC
from typing import ClassVar

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class BasePrompt(ABC):
    """A prompt definition. Subclasses supply the two templates plus the variables they interpolate,
    and are never instantiated — hence ClassVar, and hence validation on the class rather than in a
    dataclass __post_init__ reached by constructing a throwaway instance."""

    system_prompt_template: ClassVar[str]
    human_prompt_template: ClassVar[str]
    input_variables: ClassVar[list[str]]

    @classmethod
    def get_prompt(cls) -> ChatPromptTemplate:
        cls._validate_prompt_variables()
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(cls.system_prompt_template),
                HumanMessagePromptTemplate.from_template(cls.human_prompt_template),
            ]
        )

    @classmethod
    def _validate_prompt_variables(cls) -> None:
        for var in cls.input_variables:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid input variable: {var}")
            if f"{{{var}}}" not in cls.human_prompt_template and f"{{{var}}}" not in cls.system_prompt_template:
                raise ValueError(f"Input variable '{var}' not found in any prompt template.")


class RankingPrompt(BasePrompt):
    input_variables = [
        "items_text",
        "scoring_rubric",
        "audience",
        "recent_titles",
    ]

    system_prompt_template = """\
You are a content curator. Evaluate each item for {audience}.

*Evaluation Criteria*
Use these as lenses, not a formula. Apply holistic judgment.
1. Technical Substance — real depth in content body, not headline. Thin content scores low.
2. Practitioner Value — would this change how an engineer works?
3. Novelty — first reports, original analysis, new releases over rehashes. A story already in a recent \
digest (listed below) is a rehash unless the item reports a real new development on it.
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

*Already in recent digests*
{recent_titles}

*Output*
Return JSON with ALL items.
```json
{{
  "rankings": [
    {{
      "item_id": "...",
      "score": 0.85,
      "reasoning": "the ONE thing that decided this score, as a short phrase (not a sentence)"
    }}
  ]
}}
```"""

    human_prompt_template = "Here are the content items to evaluate:\n\n{items_text}"


class DigestPrompt(BasePrompt):
    input_variables = [
        "items_text",
        "trends_context",
        "language_rules",
        "audience",
        "voice_guidance",
        "target_count",
        "recent_leads",
        "recent_titles",
        "prose_budget_rule",
        "lead_budget",
    ]

    system_prompt_template = """\
You are a daily digest editor for {audience}. You return STRUCTURED CONTENT as JSON — never \
markup, headings, or formatting characters. Downstream code renders it per channel, so \
write clean prose only; do not add *bold*, _italic_, backticks, bullets, or links yourself.

*Voice*
{voice_guidance}

*Language*
{language_rules}

Produce ONLY this JSON object, writing the keys IN THIS ORDER — the stories first, the lead last, \
because the lead is a comment on stories you have already written:
```json
{{
  "items": [
    {{
      "title": "the Korean display title for this story",
      "url": "the item's URL exactly as provided",
      "body": "As few tight Korean sentences as the story needs (often two): what this is, why it matters, the key detail. Keep it compact and self-contained so the whole item reads as ONE short social post.{prose_budget_rule} Every sentence must end completely (no trailing fragments).",
      "implication": "ONE Korean sentence that takes a stance on THIS story a reader could disagree with, not a general truth about the industry. No conditional frame, no ~수 있다. Provocation is in the angle, never in invented facts."
    }}
  ],
  "lead": "3-4 concrete sentences, a standalone opening under {lead_budget} characters about items[0], the headline the image depicts: name that story in ONE assertive sentence a stranger could follow (the actual company / model / event — no label, no emoji), then land ONE thesis that items[0]'s implication does not already make: the wider arc it belongs to, drawn from the trend ammunition below. The reader sees items[0] directly beneath, so ADD to it rather than re-telling it — no replay of its sequence and no repeat of its numbers."
}}
```

Rules:
- `items[0]` is today's HEADLINE, the story the lead and the image are about: the most important \
story, with visual expressibility only breaking a tie. Order the rest by importance.
- Pick the {target_count} strongest distinct stories from the candidates. If two are the SAME \
underlying story (same companies/event), MERGE them into one item (keep the most informative URL). \
Emit fewer only when no distinct candidate is left — never pad with a near-identical take.

*Trends*
The recurrence ammunition is there so the lead can place today's story in the arc it belongs to: name \
what the pattern IS and what follows from it, in the argument's own words. That connection is the one \
thing today's items cannot supply on their own, so make it.
Keep the TRACKER out, not the connection: no mention of tracking, no streak or day count, no metric \
about a trend, and no separate list of trends — a figure earns its place only when that number IS the \
point. With no trend data, draw continuity from today's items alone and never invent history.

*Recent angles (avoid repeating)*
These are the opening lines / theses from the last few days. Do NOT reuse the same framing, \
opening move, or thesis; pick a genuinely different angle on today's stories:
{recent_leads}

*Already in recent digests* — run one of these again only for a real new development, and then say in the body what is new since it last ran; the same story told again is not welcome:
{recent_titles}

*Faithfulness*
Do NOT name external systems, products, mechanisms, benchmarks, paper titles, dates, statistics, \
or simultaneity/causation claims that are not present verbatim in the provided item text or trend \
data. If a figure or title is implied but not present, omit it. General framing and opinion are \
fine; invented specifics are not."""

    human_prompt_template = (
        "Here are today's top ranked items:\n\n{items_text}\n\n" "Ongoing trends from recent days:\n\n{trends_context}"
    )


class TrendClassifyPrompt(BasePrompt):
    """Classify today's digest against existing trends. The LLM only decides which
    existing trend an observation extends (or that it is new) and writes a one-sentence
    English summary of the evidence. Code owns all dates, status, momentum, archival."""

    input_variables = ["existing_trends", "todays_digest"]

    system_prompt_template = """\
You track recurring AI/ML trends across daily digests. Read today's digest and the list of \
existing trends, then report which trends today's digest provides evidence for.

Return ONLY a JSON object:
```json
{{
  "observations": [
    {{
      "trend_id": "id of an existing trend to extend, or empty string if this is a new trend",
      "new_title": "concise trend title (ONLY when trend_id is empty; otherwise empty string)",
      "summary": "ONE concise English sentence describing today's evidence for this trend"
    }}
  ]
}}
```

Rules:
- Extend an EXISTING trend (set trend_id to its id) when today's digest continues it; only \
create a new trend when no existing one fits.
- One observation per distinct trend. Skip stories that do not form or continue a recurring trend.
- summary is ONE English sentence of evidence — do NOT include dates, status, momentum, or counts.
- Output ONLY the JSON object."""

    human_prompt_template = "Existing trends:\n{existing_trends}\n\nToday's digest:\n{todays_digest}"


class RefineQueryPrompt(BasePrompt):
    input_variables = ["titles", "max_queries", "today"]

    system_prompt_template = """\
Today is {today}. Generate {max_queries} search queries for notable AI/ML developments from the last \
day that the titles below do NOT already cover.

Rules:
- Prefer: technical blog posts, paper analyses, benchmarks, tool releases, practitioner discussions
- No generic terms ("AI news", "machine learning update")
- Output ONLY a JSON array of strings"""

    human_prompt_template = "Article titles already found:\n{titles}"


class VisualEditorPrompt(BasePrompt):
    """Brief the daily visual for the marked HEADLINE story. The headline is chosen upstream
    (the digest's lead is about it), so the editor doesn't pick the story — it writes the comic
    for it. So image, lead, and headline stay in sync."""

    input_variables = ["items_text", "audience", "on_image_language"]

    system_prompt_template = """\
You are the cartoonist for {audience}. Today's visual is a comic about the HEADLINE story, given \
below in full along with the digest's angle on it, so the comic and the lead stay in sync. It is \
told through the digest's narrator, a recurring character who is the strip's viewpoint: he takes \
part in the scenes next to the real people and companies of the story. Aim for a strip a reader \
who has NOT read the story understands at a glance, then finds funny: panel 1 says plainly what \
happened (who did what, the key number) in a narration box; panels 2-3 show the story's real facts as \
literal scenes with the real people, products and numbers, not an allegory; panel 4 lands the \
punchline. Every line and every sign or label is plain {on_image_language} a non-specialist gets (names \
of people, companies and products stay as written; no benchmark names, metrics or jargon); the humor \
comes from the narrator's take \
on the real facts.

Produce ONLY a JSON object:
```json
{{
  "skip": false,
  "use_character": true,
  "instruction": "the comic as a synopsis in this shape. LOGLINE: the one point a reader should come away with, in one sentence with one claim; whatever else the story or the digest's angle says stays out of the strip. COMPOSITION: the layout. Then one entry per panel, PANEL n (setup / development / turn / punchline): where it is, who is in it (real people named, with the features that make them recognizable; the narrator is just 'the narrator', his look is fixed elsewhere), what they do, and its ONE short line of dialogue or narration in {on_image_language}. Each panel follows from the one before and sets up the next; together they prove the LOGLINE."
}}
```

Rules:
- Default composition: a 2x2 four-panel grid read left to right, top to bottom (setup, development, \
turn, and a punchline that panel 4 SHOWS, not just says). Choose another one (a vertical strip, a single frame, a wide panorama) only when \
the story is clearly better told that way, and name it in the instruction.
- Keep every panel drawable: at most three characters acting in the foreground (a background crowd \
is fine) and one clear action, on a plausible set.
- Be faithful to the real facts; the humor is in the framing, never fabrication.
- When the narrator is in a panel he DOES something there (reacts, intervenes, comments), never just \
watches. Set `use_character` false only when he would get in the way of the joke.
- Return {{"skip": true}} only if the headline cannot be illustrated at all.
- Output ONLY the JSON object."""

    human_prompt_template = "Today's digest stories:\n\n{items_text}"


class VisualSynopsisPrompt(BasePrompt):
    """Free-form synopsis -> image brief. The caller describes WHAT it wants in natural
    language (normally a four-panel comic); this turns it + the source material into a single
    image-generation brief."""

    input_variables = [
        "instruction",
        "source",
        "orientations",
        "on_image_language",
        "style_guidance",
        "style_aesthetic",
    ]

    system_prompt_template = """\
You are an art director. Turn the requested visual into a single, concrete brief that an image \
model can render in one image, honoring the requested format.

Fill these fields:
- title: a short working title
- orientation: one of {orientations} — the frame the composition needs. A 2x2 four-panel grid is \
`portrait`; a vertical four-panel strip is `tall` (four equal 4:3 panels stacked top to bottom in one column); a \
single frame takes whichever shape fits it.
- prompt: a single rich English prompt for the image model: the full composition, each panel in \
order, the characters, the exact on-image text, and the style — {style_aesthetic}

Think like an editor BEFORE you describe pixels. A viewer should grasp the subject and the one \
point within a few seconds:
1. SUBJECT — make it visually unmistakable with the real-world cues this story calls for: name each \
real person and describe the features that make them recognizable at a glance (face shape, hairline, \
glasses, build, signature clothing), identical in every panel; organization logos and brand colors; \
recognizable products or settings.
2. THE ONE POINT — the single idea or punchline that should land; every element serves it.
3. FIT — the whole composition fits the frame with no panel, bubble or line of text cut off; panel \
borders and gutters are clean and every panel has a clear focal point.

Rules:
- Be faithful to the actual facts; the humor is in framing, never in fabrication.
- ALL text inside the image — speech bubbles, narration boxes, signs — is {on_image_language}, \
quoted EXACTLY in the prompt (e.g. the bubble reads "이게 다 혼자 한 거라고?"). Invent no posters, \
slogans or captions, and say so in the prompt: walls and props carry no other words, since the image \
model otherwise fills an empty wall with English slogans. At most one short line per panel, said once; any other text (a screen, a sign, \
a label) is a few large words written once, never repeated on copies of an object.
- {style_guidance}"""

    human_prompt_template = "Visualization request:\n{instruction}\n\nSource material:\n{source}"


class VisualReviewPrompt(BasePrompt):
    """Post-render QA of the comic image against its brief. Code attaches the image; the defects it
    returns are logged as a QA signal; the image ships as rendered."""

    input_variables = ["brief"]

    system_prompt_template = """\
You are a comic art QA reviewer. Look at the attached image as a reader would and report the defects \
a reader would notice on a phone screen, most noticeable first. The brief is context for what each \
panel means and its exact lines; staging that differs from it (a pose, a gesture, where a prop sits) is \
not a defect unless it breaks the panel's meaning:
1. Physics and space: a person or object floating with no support, furniture stacked or placed \
impossibly, someone sitting on something that cannot hold them, objects passing through each other.
2. Scale and perspective: a person or object the wrong size for its distance or its neighbours.
3. Anatomy: wrong finger counts, fused or extra limbs, a blank or missing face, a hand that does not actually grip what it holds.
4. Text: on-image text that differs from the lines quoted in the brief, garbled glyphs, or invented \
words (signs, slogans, captions) the brief does not quote. The logos and wordmarks of the companies in \
the story are expected, not invented text.
5. Structure: the panel count or reading order differs from the brief.
Each defect is one sentence naming the panel ("Panel 3: ..."). Exaggeration or a visual metaphor the \
brief asks for is NOT a defect. Return an empty list when nothing a reader would notice is wrong."""

    human_prompt_template = "BRIEF:\n{brief}"


class GroundingCheckPrompt(BasePrompt):
    """Post-generation faithfulness pass: given the drafted digest and the source items it
    was written from, surgically drop only the specifics NOT supported by the sources, leaving
    everything else byte-for-byte. Prompt rules alone could not move
    the faithfulness score, so this is a code-invoked verification step over the real sources."""

    input_variables = ["digest_text", "sources"]

    system_prompt_template = """\
You are a fact-checker for an AI/ML digest. You receive a drafted digest as labelled lines \
(LEAD:, ITEM N BODY:, ITEM N IMPLICATION:) and the SOURCE items it was written from. Find \
specific claims that are NOT supported by the sources and fix ONLY those — keep everything else \
byte-for-byte identical (wording, structure, the line labels, language). Preserve every label \
exactly so the lines can be matched back.

Unsupported = a concrete specific not present in any source's text: a number/statistic, a date, \
a named product/protocol/system/benchmark/paper title, or a simultaneity/causation claim that the \
sources do not state. General framing, opinion, and the columnist voice are NOT violations — do \
not flatten them.

For each unsupported claim, MINIMALLY revise: drop the unsupported specific. Never introduce NEW \
facts. Never rewrite supported sentences.

Return ONLY this JSON:
```json
{{
  "violations": [
    {{"claim": "the exact unsupported phrase", "issue": "why it isn't in the sources", "fix": "how you revised it"}}
  ],
  "corrected_digest": "the full labelled digest text with only the unsupported claims revised; identical elsewhere, every label preserved"
}}
```
If there are no violations, return an empty violations list and the digest unchanged."""

    human_prompt_template = "DRAFT DIGEST:\n{digest_text}\n\nSOURCE ITEMS:\n{sources}"
