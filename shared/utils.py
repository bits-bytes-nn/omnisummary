import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .logger import logger


def extract_json_from_llm_output(raw: str) -> str:
    """Extract the first JSON value (object or array) from an LLM response.

    Tolerates leading/trailing prose and ```json fenced blocks, then returns the
    substring from the first opening brace/bracket to the matching last one. Callers
    pass the result to json.loads and handle the JSONDecodeError.
    """
    text = raw.strip()
    if "```" in text:
        fences = text.split("```")
        if len(fences) >= 3:
            text = fences[-2]
        text = re.sub(r"^\s*json\b", "", text, count=1).strip()

    candidates = [(text.find(opener), text.rfind(closer)) for opener, closer in (("{", "}"), ("[", "]"))]
    starts = [(start, end) for start, end in candidates if start != -1 and end > start]
    if not starts:
        return text
    start, end = min(starts, key=lambda pair: pair[0])
    return text[start : end + 1]


def parse_json_from_llm_output(raw: str) -> Any:
    """Extract and parse the first JSON value from an LLM response, tolerating raw control
    characters inside string values.

    LLMs routinely emit unescaped newlines/tabs inside JSON string literals (a real newline
    instead of `\\n`), which strict json.loads rejects with 'Invalid control character'.
    strict=False permits those control chars inside strings while keeping them intact, so
    multi-line prose survives without a lossy character-stripping pass. Callers still handle
    JSONDecodeError for genuinely malformed output.
    """
    text = extract_json_from_llm_output(raw)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        # A second JSON value or a braced note after the first one ("Extra data"): the extract spans
        # first brace to LAST brace, so take the first complete value, as documented.
        if not e.msg.startswith("Extra data"):
            raise
        return json.JSONDecoder(strict=False).raw_decode(text)[0]


def backoff_delay(backoff_sec: float, attempt: int, jitter_seed: str = "") -> float:
    """Delay before `attempt`'s retry: linear (backoff_sec * attempt), plus a deterministic
    0..backoff_sec jitter when a seed is given.

    Plain linear backoff re-synchronises concurrent retries into exactly the burst the upstream
    rate-limited — with 40 RSSHub account feeds, every retry came back at the same instant. The
    jitter is derived from the seed (the feed / subreddit name) rather than an RNG, so a retry
    schedule stays reproducible in tests and logs."""
    delay = backoff_sec * attempt
    if not jitter_seed:
        return delay
    frac = int(hashlib.sha256(f"{jitter_seed}:{attempt}".encode()).hexdigest(), 16) % 1000 / 1000.0
    return delay + backoff_sec * frac


async def retry_async(
    func: Callable[[], Awaitable[Any]],
    *,
    max_retries: int,
    backoff_sec: float,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    description: str = "operation",
    jitter_seed: str = "",
) -> Any:
    """Run an async callable with linear backoff on transient failures.

    Retries up to max_retries attempts, sleeping backoff_delay(backoff_sec, attempt, jitter_seed)
    between tries. Re-raises the last exception once attempts are exhausted. Pass jitter_seed
    (anything stable and per-caller, e.g. a feed URL) when many callers retry the same upstream at
    once, so their retries don't resynchronise into one burst.
    """
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await func()
        except retry_on as e:
            last_error = e
            if attempt < max_retries:
                delay = backoff_delay(backoff_sec, attempt, jitter_seed)
                logger.warning(
                    "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                    description,
                    attempt,
                    max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error
