import hashlib

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


state_manager = ToolStateManager()
