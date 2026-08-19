"""Deterministic checks on the Korean prose the digest editor writes.

Two defects survived every prompt rule written against them, and both shipped in the newest artifact:

- KOREAN_STYLE_RULES bans a comma after a finished predicate BY NAME, and PipelineConfig spells out
  the exact form with examples, yet digest_2026-07-12 items[3].implication shipped '못 쓴다,'.
- The lead spent its longest sentence re-telling items[0]'s numbers, which DigestPrompt explicitly
  forbids, while the tuned word-level Jaccard metric passed it at 0.10.

The repo already accepts verify-then-reask for a defect prompt rules could not move (_verify_grounding),
so these are checked in CODE. Every check is a fixed pattern with no threshold to tune: the point is
that the same prose gets the same verdict every run, not that the checker is clever.

Every check here must trace back to a rule KOREAN_STYLE_RULES or DigestPrompt ALREADY states. A check
invented here is a style opinion with a re-ask budget attached: an em-dash-after-a-predicate pattern
once lived here and fired on 3 of the 4 shipped digests over idiomatic Korean ('훨씬 많은 것을
배운다 — 그렇다면 ...'), because no rule anywhere bans it. Add prose rules to the config the prompt
reads, and only then check them here.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# A plain-declarative predicate ending in '다' followed by a comma and a new clause — the exact form
# KOREAN_STYLE_RULES names ('성립한다, 그 순간이 오지 않으면 ...' / '시대다, 토큰이 곧 ...'). Once a
# clause ends in a final verb form the sentence is over, so what follows the comma is a second
# sentence glued on with punctuation Korean does not use that way.
#
# '다' is also an ordinary word-final syllable ('소다', '바다'), and it legitimately precedes a comma in
# the quotative ('무한하다, 라고 그는 말했다' is ONE sentence). Both forms hit the bare `[가-힣]다,`
# pattern and each false hit buys a byte-identical ~50k-token re-ask, so two conditions were added: the
# quotative is excluded outright, and what FOLLOWS the comma must itself stand as a clause — which is
# what separates a second sentence glued on from a list separator ('소다, 커피, 물이 전부다').
_QUOTATIVE = r"라[고는며]"
_COMMA_AFTER_PREDICATE = re.compile(rf"[가-힣]다(,)\s*(?!{_QUOTATIVE})[가-힣]")
# Where the segment after the comma ends: the next comma (a list) or the end of the sentence.
_SEGMENT_END = re.compile(r"[,.!?…]")
# A segment that stands on its own as a clause: it closes on a sentence-final ending. A bare list item
# ('커피', '산') does not, which is the whole point.
_CLAUSE_END = re.compile(r"[가-힣](?:다|[요죠네])[\"'”’)\]]*$")

# What counts as a "specific" when comparing two pieces of prose: a numeric figure, or a Latin-script
# token (model names, companies, benchmarks). Crude on purpose — it is only ever used SYMMETRICALLY,
# to ask whether the lead carries anything the headline item does not, never to grade prose on its own.
_DIGIT_RUN = re.compile(r"\d+(?:[.,]\d+)*")
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9.\-]+")

# A figure the prose states in its own right, as opposed to a digit inside a name: the '5' of 'GPT-5'
# (already compared as a Latin token) or of '제3자' (a word, not a quantity — it was the only hit on
# digest_2026-06-11). A stated figure opens its token, so anything glued to its left disqualifies it.
# Thousands separators are part of the figure.
_STANDALONE_FIGURE = re.compile(r"(?<![A-Za-z0-9.,\-가-힣])\d+(?:,\d{3})*(?:\.\d+)?")

# How much of an offending line to quote in the hit list. Long enough to locate the sentence in the
# stored digest, short enough to keep one log line readable.
_QUOTE_CHARS = 60


class ItemProse(BaseModel):
    """One story's editor-authored prose, plus the character budget code TOLD the editor it had.

    The title is in here because the editor authors it and it occupies the same post; `budget` is 0
    for a caller that only wants the style checks (the length check then has nothing to check
    against)."""

    title: str = ""
    body: str = ""
    implication: str = ""
    budget: int = Field(default=0, ge=0)

    @property
    def prose_chars(self) -> int:
        return len(self.title) + len(self.body) + len(self.implication)


def _quote(text: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - _QUOTE_CHARS // 2)
    return text[start : match.end() + _QUOTE_CHARS // 2].strip()


def _is_second_clause(text: str, comma_end: int) -> bool:
    """Whether what follows the comma is a clause of its own rather than the next item of a list."""
    end = _SEGMENT_END.search(text, comma_end)
    segment = text[comma_end : end.start() if end else len(text)].strip()
    return bool(_CLAUSE_END.search(segment))


def _punctuation_hits(label: str, text: str) -> list[str]:
    for match in _COMMA_AFTER_PREDICATE.finditer(text):
        if _is_second_clause(text, match.end(1)):
            return [f"{label}: comma after a finished predicate — '{_quote(text, match)}'"]
    return []


def specifics(text: str) -> set[str]:
    """The concrete, checkable tokens in a piece of prose: numeric figures and Latin-script names."""
    numbers = {match.group(0) for match in _DIGIT_RUN.finditer(text)}
    latin = {match.group(0).lower().rstrip(".-") for match in _LATIN_TOKEN.finditer(text)}
    return numbers | {token for token in latin if len(token) >= 2}


def lead_specificity_hits(lead: str, headline_prose: str) -> list[str]:
    """The lead borrowed specifics from the headline item and added none of its own.

    The reader sees items[0] directly beneath the lead, so a lead whose only concrete content is
    already in the story below is spending its longest sentence on a second telling. Flagged ONLY when
    the lead does carry specifics: a lead that names no figure at all is a different shape, and
    treating it as a violation would be a new rule rather than a check on an existing one."""
    lead_specifics = specifics(lead)
    if not lead_specifics:
        return []
    if lead_specifics - specifics(headline_prose):
        return []
    return [
        "lead: every specific it names is already in items[0] "
        f"({', '.join(sorted(lead_specifics))}) — it re-tells the story instead of adding to it"
    ]


def figures(text: str) -> set[str]:
    """The numeric figures the prose states, normalized so the same figure written to different
    precision is the same figure ('52' and '52.2' are one claim told twice, not two)."""
    return {match.group(0).replace(",", "").split(".")[0] for match in _STANDALONE_FIGURE.finditer(text)}


def lead_figure_repeats(lead: str, headline_prose: str) -> list[str]:
    """A figure the lead borrowed from items[0], which DigestPrompt forbids by name ('no repeat of
    its numbers').

    Separate from lead_specificity_hits because that check passes the lead as soon as it names ANY
    novel specific: on digest_2026-07-12 three novel ones (a rounded figure, a bare count, one new
    number) bought a pass while 15 of the lead's 18 specifics were repeats, and the root Threads post
    read as a second telling of reply 1."""
    repeats = sorted(figures(lead) & figures(headline_prose), key=lambda figure: (len(figure), figure))
    if not repeats:
        return []
    return [
        f"lead: repeats items[0]'s figure(s) ({', '.join(repeats)}) — the reader sees that story "
        "directly beneath, so the lead has to add to it"
    ]


def item_length_hits(label: str, item: ItemProse) -> list[str]:
    """The item's prose exceeds the budget code told the editor it had. Nothing verified this before:
    the budget was stated in the prompt and the renderer silently amputated whatever did not fit,
    which cost the item its closing sentence on 2 of the 5 posts of digest_2026-07-12."""
    if item.budget <= 0 or item.prose_chars <= item.budget:
        return []
    return [
        f"{label}: title + body + implication spends {item.prose_chars} of its {item.budget}-char "
        "budget — the renderer drops the trailing sentence that does not fit"
    ]


def lint_digest_prose(lead: str, items: list[ItemProse]) -> list[str]:
    """Every prose defect found in the digest, as human-readable lines (empty when clean).

    `items` is one ItemProse per story, in order — the fields the editor authors. URLs and source tags
    are code-owned and excluded, exactly as the grounding payload excludes them."""
    hits = _punctuation_hits("lead", lead)
    for index, item in enumerate(items):
        label = f"items[{index}]"
        hits.extend(_punctuation_hits(f"{label}.body", item.body))
        hits.extend(_punctuation_hits(f"{label}.implication", item.implication))
        hits.extend(item_length_hits(label, item))
    if items:
        headline_prose = f"{items[0].body}\n{items[0].implication}"
        hits.extend(lead_specificity_hits(lead, headline_prose))
        hits.extend(lead_figure_repeats(lead, headline_prose))
    return hits
