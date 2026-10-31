"""The ports the application core reaches the outside world through.

The core (application/, domain/) depends only on these. Adapters (adapters/) implement them, and
the entrypoints build the adapters in entrypoints/bootstrap.py and hand them in, so a test hands
in a fake instead: nothing in the core can reach AWS, Threads, OpenAI or the web on its own."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import date, datetime, timedelta
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel

from domain.models import CollectedItem, ImageAsset, SourceStatus, ThreadsComment, ThreadsDelivery
from shared.constants import LanguageModelId
from shared.logger import logger


class StateReadError(RuntimeError):
    """The state blob could NOT be read — distinct from "there is no history yet".

    Both used to come back as None, so a throttled/denied S3 GET read as an empty ledger and the
    next read-modify-write persisted that emptiness: the published-URL ledger, the recent-leads log,
    the visual-format window and the Threads idempotency marker were all blanked by one failed read.

    Every consumer handles this the same way: log at ERROR, treat the history as UNKNOWN, and SKIP
    the write. It must never reach a publish path — a lost digest is strictly worse than a run
    without history.
    """


class StateStore(ABC):
    """Blob store for the structured trends memory (read-modify-write each run).

    Distinct from MemoryStore by design — and NOT replaceable by it.
    trends.json is a deliberately time-varying document with explicit code-managed
    merge/cooling/archive of topic threads. AgentCore's managed strategies extract STABLE
    records (semantic facts, user preferences) or per-session summaries; even
    customMemoryStrategy can only *append to* those built-in prompts, not implement
    trend-thread maintenance. So trends.json stays the system of record for trends;
    MemoryStore holds the digest snapshot for the follow-up agent.
    """

    @abstractmethod
    def read(self, key: str) -> str | None: ...

    @abstractmethod
    def write(self, key: str, content: str) -> None: ...

    def read_json(self, key: str, default: Any = None) -> Any:
        """Read and parse a JSON blob; return default on missing/corrupt content.

        Raises StateReadError when the blob could not be READ at all — corrupt content is a
        recoverable "start fresh", an unreadable store is not."""
        raw = self.read(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("State key '%s' held invalid JSON; using default", key)
            return default

    def write_json(self, key: str, value: Any) -> None:
        self.write(key, json.dumps(value, ensure_ascii=False))


class MemoryReadError(RuntimeError):
    """The digest snapshot could NOT be read — distinct from "there is no snapshot for that date".

    Both used to come back as None, so a throttled/denied/misconfigured AgentCore read was
    indistinguishable from a day that simply has no digest: the visual Lambda logged "nothing to
    publish", returned 200, and the day's digest — whose only delivery path that Lambda is — was
    never posted, with no alarm.

    A consumer that PUBLISHES must let this escape (a failed invoke fires the Errors alarm and
    lands in the DLQ; retry_attempts=0 means it cannot duplicate a post). A consumer that only
    ENRICHES — the cross-day dedup seed in run_pipeline — keeps catching it and degrades, because a
    lost dedup hint is harmless while a lost digest is not.
    """


class MemoryStore(ABC):
    """Persistence boundary for the structured digest snapshot, one per digest date.

    The visual Lambda reloads the day's snapshot to publish it, the pipeline reads recent ones to
    seed cross-day dedup, and the research CLI's recall_digest reads one back. Trend memory lives in
    the structured trends.json owned by the TrendTracker, not here.
    """

    @abstractmethod
    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_digest(self, digest_date: str) -> dict[str, Any] | None:
        """The snapshot stored FOR that ISO date, or None when the date has none.

        Keyed by the digest date rather than "the latest": digest_result.generated_at is UTC and
        would false-mismatch the KST digest date on every pre-09:00 KST run.

        Raises MemoryReadError when the snapshot could not be READ (as opposed to not existing),
        so a publish path never mistakes an unreadable store for an empty day.
        """

    def get_recent_digests(self, before: date, days: int) -> list[dict[str, Any]]:
        """The snapshots for the `days` dates preceding `before` (newest first), skipping dates that
        have none. Seeds cross-day dedup with the SAME window the published-URL ledger uses, and
        never includes `before` itself, so a same-day re-run keeps its own stories.

        Session ids are deterministic (`digest-YYYY-MM-DD`), so this reads exactly `days` sessions
        instead of listing every session ever stored. Best-effort per date: it only ENRICHES the
        dedup seed, so one unreadable day is logged and skipped rather than failing the read."""
        out: list[dict[str, Any]] = []
        for offset in range(1, max(0, days) + 1):
            day = (before - timedelta(days=offset)).isoformat()
            try:
                data = self.get_digest(day)
            except MemoryReadError as e:
                logger.warning("Skipping the digest snapshot for %s: %s", day, e)
                continue
            if data is not None:
                out.append(data)
        return out


class LanguageModels(Protocol):
    """Chat models by registry id, each tagged with the stage that spends it."""

    def get_model(self, model_id: LanguageModelId, *, stage: str = "", **kwargs: Any) -> BaseChatModel: ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str: ...


class ImageRenderer(Protocol):
    """Prompt -> image bytes (the daily comic)."""

    def ready(self) -> bool:
        """Whether a render can be attempted at all (its credential is readable and set). Logs why not."""
        ...

    def render(self, prompt: str, size: str, inputs: Sequence[bytes], label: str) -> bytes:
        """Blocking: the caller runs it off the event loop. `inputs` are reference images."""
        ...

    def is_moderation_error(self, exc: Exception) -> bool: ...


class PublishError(RuntimeError):
    """Threads refused or failed a post. `may_be_live`: the failure came back from the publish step in
    a way that can answer a post that went up, so posting the same text again may duplicate it.
    `transient`: another attempt may succeed."""

    def __init__(self, status: int, detail: str, *, may_be_live: bool, transient: bool) -> None:
        super().__init__(f"Threads publish failed ({status}): {detail}")
        self.status = status
        self.detail = detail
        self.may_be_live = may_be_live
        self.transient = transient


class ThreadsComments(Protocol):
    """One authenticated session over the account's posts and the comments under them."""

    async def fetch_own_posts(self, since: datetime) -> list[ThreadsComment]: ...

    async def fetch_conversation(self, media_id: str) -> list[ThreadsComment]: ...

    async def publish_reply(self, text: str, reply_to_id: str) -> str:
        """The reply's id. Raises PublishError when Threads refuses or fails it."""
        ...


class ThreadsPublisher(Protocol):
    # Seconds a post set may still need once its image is ready (the root's indexing wait plus the
    # reply chain's reserve), so a caller under a deadline stops the image work in time.
    publish_budget_sec: float

    async def post_thread(
        self,
        root_text: str,
        replies: list[str],
        *,
        image_bytes: bytes | None = None,
        image_content_type: str = "image/png",
        image_alt: str = "",
        deadline: float | None = None,
    ) -> ThreadsDelivery:
        """A root post (with the image when there is one) and one flat reply per entry of `replies`.
        Never raises for a Threads failure: the verdict says what landed."""
        ...

    def comments(self) -> AbstractAsyncContextManager[ThreadsComments]: ...


class ContentSource(Protocol):
    """One collector. After collect(), `health_issue` is (STALE or DEGRADED, why) when its items are
    usable but not healthy (a stalled sync, a fraction of its inputs answering), else None."""

    @property
    def health_issue(self) -> tuple[SourceStatus, str] | None: ...

    async def collect(self) -> list[CollectedItem]: ...


class ResearchSources(Protocol):
    """What the research agent reads. Each search answers formatted text, one "URL:" line per result."""

    async def web_search(
        self,
        query: str,
        *,
        topic: Literal["news"] | None = None,
        time_range: str | None = None,
        include_domains: list[str] | None = None,
    ) -> str: ...

    async def paper_search(self, query: str) -> str: ...

    async def read_page(self, url: str) -> str:
        """The page's text, or "" when it could not be read."""
        ...

    async def fetch_image(self, url: str) -> ImageAsset | None:
        """The page's representative image (og:image), or None."""
        ...
