from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from shared.constants import DEFAULT_PROJECT_NAME, METRIC_NAMESPACE
from shared.naming import metric_dimensions


def emit_emf(values: dict[str, int | float], extra: dict[str, Any] | None = None) -> None:
    """Emit ONE CloudWatch EMF record on stdout carrying `values` as metrics of the project
    namespace, dimensioned by project/stage. EMF is just a log line, so this needs no AWS resource
    and no PutMetricData call. `extra` holds non-metric properties (context an operator reads off
    the log record but does not alarm on).

    The timestamp is UTC: datetime.now() reads the naive LOCAL clock while EMF interprets Timestamp
    as epoch-UTC ms, so on a non-UTC runtime every datapoint would be filed at the wrong time (and
    far enough off, CloudWatch rejects it outright)."""
    dimensions = metric_dimensions(
        os.environ.get("PROJECT_NAME", DEFAULT_PROJECT_NAME),
        os.environ.get("STAGE", ""),
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
