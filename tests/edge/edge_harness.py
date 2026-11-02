"""Scripted LanguageModels for end-to-end runs: every get_model() returns a RunnableLambda that routes
by stage to a handler and records each call, so a test can count LLM calls per stage."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

ITEM_RE = re.compile(r"=== Item (\d+) ===\nTitle: (.*)")


class ScriptedModels:
    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.calls: dict[str, int] = defaultdict(int)
        self.prompts: dict[str, list[str]] = defaultdict(list)
        self.truncate_to_tokens = MagicMock(side_effect=lambda text, max_tokens: text)

    def get_model(self, model_id, *, stage: str = "", **kwargs):
        def answer(prompt_value):
            text = prompt_value.to_string()
            self.calls[stage] += 1
            self.prompts[stage].append(text)
            handler = self.handlers.get(stage)
            if handler is None:
                raise AssertionError(f"unexpected LLM stage {stage!r}")
            out = handler(text)
            if isinstance(out, BaseException):
                raise out
            return AIMessage(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))

        async def aanswer(prompt_value):
            return answer(prompt_value)

        return RunnableLambda(func=answer, afunc=aanswer)


def score_by_title(scores: dict[str, float], default: float = 0.7):
    """A ranking handler: scores each `=== Item N ===` by its title."""

    def handler(text: str):
        return {
            "rankings": [
                {"item": int(n), "score": scores.get(title.strip(), default), "reasoning": "r"}
                for n, title in ITEM_RE.findall(text)
            ]
        }

    return handler


def prose(_text: str) -> str:
    return "I'm sorry, but I can't score these items as JSON. Here is my take in prose instead."


def editor_picking(urls: list[str], lead: str = "오늘의 리드 문장이다. 두 번째 문장이다."):
    def handler(_text: str):
        return {
            "items": [
                {"title": f"제목 {i}", "url": u, "body": "본문 문장이다.", "implication": "시사점이다."}
                for i, u in enumerate(urls)
            ],
            "lead": lead,
        }

    return handler


def no_violations(_text: str):
    return {"violations": [], "corrected_digest": ""}


def no_groups(_text: str):
    return {"same_story": []}


URL_RE = re.compile(r"^URL: (\S+)", re.M)


def editor_first_candidates(n: int = 2, lead: str = "오늘의 리드 문장이다. 두 번째 문장이다."):
    """An editor that picks the first n candidates it was shown."""

    def handler(text: str):
        return editor_picking(URL_RE.findall(text)[:n], lead)(text)

    return handler
