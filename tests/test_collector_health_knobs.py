"""Every collector must hand the health helpers ALL the degradation knobs its config declares.

record_run_health/flag_degraded_park default empty_threshold to 100.0 and max_failed to 0 — both
disabled. A collector that passes only `threshold=` therefore silently ignores
`collectors.<source>.empty_rate_threshold` / `max_failed_inputs`: the strict config model accepts
`empty_rate_threshold: 90`, and 9 channels that all resolve but all return nothing still report OK.
The invariant is asserted over the call SITES so it also holds for collectors added later.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from collectors import RedditCollector, RSSCollector, RSSHubCollector, WebSearchCollector, YouTubeCollector
from collectors.base import PARK_META_ACCOUNTS_TOTAL, ParkedItems, ParkOutcome
from shared.config import YouTubeCollectorConfig

HEALTH_METHODS = ("record_run_health", "flag_degraded_park")
# kwarg name -> the config field it must be bound to.
REQUIRED_KNOBS = {
    "threshold": "self.config.error_rate_threshold",
    "empty_threshold": "self.config.empty_rate_threshold",
    "max_failed": "self.config.max_failed_inputs",
}
COLLECTORS = (RedditCollector, RSSCollector, RSSHubCollector, WebSearchCollector, YouTubeCollector)


def _health_calls(collector_cls: type) -> list[ast.Call]:
    source = Path(inspect.getsourcefile(collector_cls)).read_text()
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in HEALTH_METHODS
    ]


@pytest.mark.parametrize("collector_cls", COLLECTORS, ids=lambda c: c.__name__)
def test_every_health_call_forwards_every_knob(collector_cls):
    calls = _health_calls(collector_cls)
    assert calls, f"{collector_cls.__name__} reports no run health at all"
    for call in calls:
        passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}
        for name, expected in REQUIRED_KNOBS.items():
            assert (
                passed.get(name) == expected
            ), f"{collector_cls.__name__}.{call.func.attr} must pass {name}={expected}, got {passed.get(name)!r}"


class TestYouTubeForwardsTheKnobsAtRuntime:
    """The collector this regressed on: `collectors.youtube.empty_rate_threshold: 90` validated
    cleanly and was a no-op on both the live and the parked path."""

    @staticmethod
    def _config() -> YouTubeCollectorConfig:
        return YouTubeCollectorConfig(channels=["https://www.youtube.com/@example"], empty_rate_threshold=42.0)

    @pytest.mark.asyncio
    async def test_the_parked_path_judges_with_the_configured_knobs(self):
        collector = YouTubeCollector(self._config())
        parked = ParkedItems(outcome=ParkOutcome.FRESH, meta={PARK_META_ACCOUNTS_TOTAL: 1})
        with patch("collectors.youtube.load_items_from_s3", return_value=parked):
            with patch.object(collector, "flag_degraded_park") as flag:
                await collector.collect()
        assert flag.call_args.kwargs["empty_threshold"] == 42.0
        assert flag.call_args.kwargs["max_failed"] == 0

    @pytest.mark.asyncio
    async def test_the_live_path_judges_with_the_configured_knobs(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(self._config())
        with patch("collectors.youtube.load_items_from_s3", return_value=ParkedItems(outcome=ParkOutcome.ABSENT)):
            with patch.object(collector, "_collect_channel", new=AsyncMock(return_value=[])):
                with patch.object(collector, "record_run_health") as record:
                    await collector.collect()
        assert record.call_args.kwargs["empty_threshold"] == 42.0
        assert record.call_args.kwargs["max_failed"] == 0
