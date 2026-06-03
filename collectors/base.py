from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime, timedelta

from shared import CollectedItem, logger


class BaseCollector(ABC):
    @abstractmethod
    async def collect(self) -> list[CollectedItem]: ...


def cutoff_datetime(lookback_hours: int, reference_time: datetime | None = None) -> datetime:
    return (reference_time or datetime.now(UTC)) - timedelta(hours=lookback_hours)


async def gather_collector_results(
    tasks: Sequence[Awaitable[list[CollectedItem]]],
    labels: list[str] | None = None,
) -> list[CollectedItem]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            label = labels[i] if labels else f"task-{i}"
            logger.warning("Collector task '%s' failed: %s", label, result)
        else:
            items.extend(result)
    return items
