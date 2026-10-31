"""In-memory stand-ins for the application's ports (application/ports.py), handed to the core the way
entrypoints/bootstrap.py hands it the real adapters. A test that uses them cannot reach Threads,
OpenAI or AWS whatever it does."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from adapters.threads import ThreadsGateway
from shared.models import ThreadsDelivery
from shared.state_store import StateStore


class MemoryStateStore(StateStore):
    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def read(self, key: str) -> str | None:
        return self.blobs.get(key)

    def write(self, key: str, content: str) -> None:
        self.blobs[key] = content


class FakeComments:
    def __init__(self, posts=None, conversation=None, publish=None) -> None:
        self.fetch_own_posts = AsyncMock(return_value=list(posts or []))
        self.fetch_conversation = AsyncMock(return_value=list(conversation or []))
        self.publish_reply = publish or AsyncMock(return_value="reply-id")


class FakeThreads:
    """`post_thread` is an AsyncMock answering `outcome`; `comments()` yields `session`."""

    publish_budget_sec = ThreadsGateway.publish_budget_sec

    def __init__(self, outcome: ThreadsDelivery | None = None, *, side_effect=None, session=None) -> None:
        self.post_thread = AsyncMock(return_value=outcome or ThreadsDelivery(2, 2), side_effect=side_effect)
        self.session = session or FakeComments()

    @asynccontextmanager
    async def comments(self):
        yield self.session


class FakeRenderer:
    def __init__(self, *, ready: bool = True, image: bytes = b"PNG") -> None:
        self.ready = MagicMock(return_value=ready)
        self.render = MagicMock(return_value=image)
        self.is_moderation_error = MagicMock(return_value=False)


def fake_models() -> MagicMock:
    """A LanguageModels whose truncator is a real no-op: a bare MagicMock would put its repr into a
    prompt while every assertion still passed."""
    models = MagicMock()
    models.truncate_to_tokens.side_effect = lambda text, max_tokens: text
    return models
