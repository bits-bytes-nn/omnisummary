import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

from shared import ParseResult, SummaryResult


class ToolStateManager:
    def __init__(self) -> None:
        self._parse_results: dict[str, ParseResult] = {}
        self._summary_results: dict[str, SummaryResult] = {}
        self._last_parse_hash: str | None = None
        self._last_summary_hash: str | None = None
        self._slack_channel_id: str | None = None
        self._message_sent: bool = False

    def clear(self) -> None:
        self._parse_results.clear()
        self._summary_results.clear()
        self._last_parse_hash = None
        self._last_summary_hash = None
        self._slack_channel_id = None
        self._message_sent = False

    def get_parse_result(self, parse_hash: str | None = None) -> ParseResult:
        effective_hash = parse_hash or self._last_parse_hash
        if effective_hash is None:
            raise ValueError("No parse result found (no hash specified and no recent parse)")

        result = self._parse_results.get(effective_hash)
        if result is None:
            raise ValueError(f"Parse result for hash '{effective_hash}' not found")
        return result

    def get_summary_result(self, summary_hash: str | None = None) -> SummaryResult:
        effective_hash = summary_hash or self._last_summary_hash
        if effective_hash is None:
            raise ValueError("No summary result found (no hash specified and no recent summary)")

        result = self._summary_results.get(effective_hash)
        if result is None:
            raise ValueError(f"Summary result for hash '{effective_hash}' not found")
        return result

    def store_parse_result(self, url: str, result: ParseResult) -> str:
        parse_hash = self._generate_hash(url)
        self._parse_results[parse_hash] = result
        self._last_parse_hash = parse_hash
        return parse_hash

    @staticmethod
    def _generate_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def store_summary_result(self, parse_hash: str, result: SummaryResult) -> str:
        summary_hash = f"{parse_hash}_sum"
        self._summary_results[summary_hash] = result
        self._last_summary_hash = summary_hash
        return summary_hash

    @property
    def slack_channel_id(self) -> str | None:
        return self._slack_channel_id

    @slack_channel_id.setter
    def slack_channel_id(self, channel_id: str | None) -> None:
        self._slack_channel_id = channel_id

    @property
    def message_sent(self) -> bool:
        return self._message_sent

    def mark_message_sent(self) -> None:
        self._message_sent = True


_state_context: ContextVar[ToolStateManager] = ContextVar("tool_state")


def get_state_manager() -> ToolStateManager:
    try:
        return _state_context.get()
    except LookupError:
        raise RuntimeError(
            "No active state context. Use 'with tool_state_context():' to create one."
        ) from None


@contextmanager
def tool_state_context(channel_id: str | None = None) -> Generator[ToolStateManager, None, None]:
    state = ToolStateManager()
    state.slack_channel_id = channel_id
    token = _state_context.set(state)
    try:
        yield state
    finally:
        _state_context.reset(token)


class _StateManagerProxy:
    def __getattr__(self, name: str):
        return getattr(get_state_manager(), name)

    def __setattr__(self, name: str, value):
        setattr(get_state_manager(), name, value)

    @staticmethod
    def clear() -> None:
        get_state_manager().clear()

    @property
    def slack_channel_id(self) -> str | None:
        return get_state_manager().slack_channel_id

    @slack_channel_id.setter
    def slack_channel_id(self, channel_id: str | None) -> None:
        get_state_manager().slack_channel_id = channel_id

    @property
    def message_sent(self) -> bool:
        return get_state_manager().message_sent

    @property
    def _last_parse_hash(self) -> str | None:
        return get_state_manager()._last_parse_hash

    @property
    def _summary_results(self) -> dict[str, SummaryResult]:
        return get_state_manager()._summary_results


state_manager = _StateManagerProxy()
