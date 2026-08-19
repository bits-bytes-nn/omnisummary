from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from .constants import DEFAULT_PROJECT_NAME, METRIC_NAMESPACE, EnvVars


def metric_dimensions(project: str, stage: str) -> dict[str, str]:
    """The dimension map every EMF record carries, and the SAME map the CDK alarms are built with.

    Undimensioned records from different deployments collapse into one datapoint: a dev run of 5
    items made prod's Maximum<1 empty-digest alarm green on a day prod shipped nothing, and a dev
    agent failure paged on prod's error alarm. One function owns the dimension NAMES so the alarm
    can never read a different datapoint than the code publishes.

    An empty value is dropped rather than emitted: CloudWatch rejects a record with an empty
    dimension value outright, which would lose the metric instead of mis-filing it."""
    return {name: value for name, value in (("Project", project), ("Stage", stage)) if value}


def emit_emf(values: dict[str, int | float], extra: dict[str, Any] | None = None) -> None:
    """Emit ONE CloudWatch EMF record on stdout carrying `values` as metrics of the project
    namespace, dimensioned by project/stage. EMF is just a log line, so this needs no AWS resource
    and no PutMetricData call. `extra` holds non-metric properties (context an operator reads off
    the log record but does not alarm on).

    The timestamp is UTC: datetime.now() reads the naive LOCAL clock while EMF interprets Timestamp
    as epoch-UTC ms, so on a non-UTC runtime every datapoint would be filed at the wrong time (and
    far enough off, CloudWatch rejects it outright)."""
    dimensions = metric_dimensions(
        os.environ.get(EnvVars.PROJECT_NAME.value, DEFAULT_PROJECT_NAME),
        os.environ.get(EnvVars.STAGE.value, ""),
    )
    record: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(datetime.now(UTC).timestamp() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": METRIC_NAMESPACE,
                    "Dimensions": [list(dimensions)],
                    "Metrics": [{"Name": name} for name in values],
                }
            ],
        },
        **dimensions,
        **values,
    }
    if extra:
        record.update(extra)
    print(json.dumps(record))
