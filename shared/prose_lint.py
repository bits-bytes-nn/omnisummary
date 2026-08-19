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

# A plain-declarative predicate ending in '다' followed by a comma and a new clause — the exact form
# KOREAN_STYLE_RULES names ('성립한다, 그 순간이 오지 않으면 ...' / '시대다, 토큰이 곧 ...'). Once a
# clause ends in a final verb form the sentence is over, so what follows the comma is a second
# sentence glued on with punctuation Korean does not use that way.
_COMMA_AFTER_PREDICATE = re.compile(r"[가-힣]다,\s*[가-힣]")

# What counts as a "specific" when comparing two pieces of prose: a numeric figure, or a Latin-script
# token (model names, companies, benchmarks). Crude on purpose — it is only ever used SYMMETRICALLY,
# to ask whether the lead carries anything the headline item does not, never to grade prose on its own.
_DIGIT_RUN = re.compile(r"\d+(?:[.,]\d+)*")
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9.\-]+")

# How much of an offending line to quote in the hit list. Long enough to locate the sentence in the
# stored digest, short enough to keep one log line readable.
_QUOTE_CHARS = 60


def _quote(text: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - _QUOTE_CHARS // 2)
    return text[start : match.end() + _QUOTE_CHARS // 2].strip()


def _punctuation_hits(label: str, text: str) -> list[str]:
    match = _COMMA_AFTER_PREDICATE.search(text)
    if not match:
        return []
    return [f"{label}: comma after a finished predicate — '{_quote(text, match)}'"]


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


def lint_digest_prose(lead: str, items: list[tuple[str, str]]) -> list[str]:
    """Every prose defect found in the digest, as human-readable lines (empty when clean).

    `items` is (body, implication) per story, in order — the fields the editor authors. Titles, URLs
    and source tags are code-owned and excluded, exactly as the grounding payload excludes them."""
    hits = _punctuation_hits("lead", lead)
    for index, (body, implication) in enumerate(items):
        hits.extend(_punctuation_hits(f"items[{index}].body", body))
        hits.extend(_punctuation_hits(f"items[{index}].implication", implication))
    if items:
        body, implication = items[0]
        hits.extend(lead_specificity_hits(lead, f"{body}\n{implication}"))
    return hits
