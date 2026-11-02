"""BedrockLanguageModelFactory.truncate_to_tokens with a fake CountTokens client."""

from __future__ import annotations

import math

import pytest

from adapters.bedrock import BedrockLanguageModelFactory


class FakeCountClient:
    """Counts ceil(chars/4) + overhead tokens; rejects empty text the way Bedrock's validation does."""

    def __init__(self, overhead=0, fail_after=None):
        self.calls = []
        self.overhead = overhead
        self.fail_after = fail_after

    def count_tokens(self, *, modelId, input):
        text = input["converse"]["messages"][0]["content"][0]["text"]
        self.calls.append(len(text))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("ThrottlingException")
        if not text:
            raise RuntimeError("ValidationException: text content blocks must be non-empty")
        return {"inputTokens": math.ceil(len(text) / 4) + self.overhead}


def _factory(client) -> BedrockLanguageModelFactory:
    factory = object.__new__(BedrockLanguageModelFactory)
    factory._client = client
    return factory


def test_max_tokens_one():
    client = FakeCountClient()
    factory = _factory(client)
    out = factory.truncate_to_tokens("word " * 1000, 1)
    assert factory.count_tokens(out) <= 1 if out else True
    assert len(client.calls) <= 12
    assert not factory.__dict__.get("_count_tokens_degraded"), "a tiny budget must not degrade CountTokens"


def test_max_tokens_one_with_message_overhead_never_fits():
    """Real CountTokens counts the message framing too, so a 1-token budget can never be met: the cut walks
    down to the empty prefix, Bedrock rejects the empty text, and the factory degrades CountTokens to the
    char estimate for the rest of the run."""
    client = FakeCountClient(overhead=7)
    factory = _factory(client)
    out = factory.truncate_to_tokens("word " * 1000, 1)
    assert 0 not in client.calls, "truncate sent an empty text to CountTokens"
    assert not factory.__dict__.get("_count_tokens_degraded")
    assert out == "" or factory.count_tokens(out) <= 1


def test_one_huge_word():
    client = FakeCountClient()
    factory = _factory(client)
    text = "x" * 200_000
    out = factory.truncate_to_tokens(text, 1000)
    assert 0 < len(out) <= 4000 and len(client.calls) <= 8


def test_count_tokens_raising_midway_degrades_and_still_returns_a_fitting_cut():
    client = FakeCountClient(fail_after=1)
    factory = _factory(client)
    text = "가나다라 " * 20_000
    out = factory.truncate_to_tokens(text, 500)
    assert factory.__dict__.get("_count_tokens_degraded")
    assert len(out) // 4 <= 500 and out
    calls = len(client.calls)
    factory.truncate_to_tokens("다른 " * 30_000, 500)
    assert len(client.calls) == calls  # no further CountTokens calls once degraded


def test_count_tokens_raising_on_the_first_call():
    client = FakeCountClient(fail_after=0)
    factory = _factory(client)
    out = factory.truncate_to_tokens("word " * 10_000, 100)
    assert len(out) // 4 <= 100 and out


@pytest.mark.parametrize("text", ["", " ", "a"])
def test_trivial_texts(text):
    factory = _factory(FakeCountClient())
    assert factory.truncate_to_tokens(text, 1) == text
