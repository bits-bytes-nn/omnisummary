"""The ports the application core reaches the outside world through.

The core (application/, domain/) depends only on these. Adapters (adapters/) implement them, and
the entrypoints build the adapters in entrypoints/bootstrap.py and hand them in, so a test hands
in a fake instead: nothing in the core can reach AWS, Threads, OpenAI or the web on its own."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel

from shared.constants import LanguageModelId
from shared.models import CollectedItem, ImageAsset, ThreadsDelivery


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

    async def fetch_own_posts(self, since: datetime) -> list[dict[str, Any]]: ...

    async def fetch_conversation(self, media_id: str) -> list[dict[str, Any]]: ...

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
        image_key_prefix: str = "",
        topic_tag: str = "",
        deadline: float | None = None,
    ) -> ThreadsDelivery:
        """A root post (with the image when there is one) and one flat reply per entry of `replies`.
        Never raises for a Threads failure: the verdict says what landed."""
        ...

    def comments(self) -> AbstractAsyncContextManager[ThreadsComments]: ...


class ParkStatus(Protocol):
    degraded: bool
    detail: str


class ContentSource(Protocol):
    """One collector. After collect(), `park_status` says whether its items came from a stale or
    unreadable park file and `degraded_detail` whether only a fraction of its inputs answered."""

    park_status: ParkStatus | None
    degraded_detail: str

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
