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
    raise_if_all_failed: bool = False,
) -> list[CollectedItem]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    failures: list[BaseException] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            label = labels[i] if labels else f"task-{i}"
            logger.warning("Collector task '%s' failed: %s", label, result)
            failures.append(result)
        else:
            items.extend(result)

    # When every task errored (and produced nothing), surface it so the health check
    # marks the source FAILED instead of reporting a silent empty result on an outage.
    if raise_if_all_failed and results and len(failures) == len(results):
        raise RuntimeError(f"All {len(failures)} collector tasks failed: {failures[0]}")

    return items
