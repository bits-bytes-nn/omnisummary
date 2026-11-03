from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from shared.config import EDITION_ENV
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
        os.environ.get(EDITION_ENV, ""),
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


# One metric per kind of spend, summed per day by CloudWatch: the bill's own split (Bedrock, OpenAI,
# Tavily) rather than one number nobody could act on.
SPEND_METRICS = {"llm": "LLMCostUSD", "image": "ImageCostUSD", "search": "TavilyCredits"}


def emit_spend(kind: str, stage: str, model: str, amount: float | None, usage: dict[str, int]) -> None:
    """One EMF record per billed call: the amount as a metric, the stage, model and raw usage as
    properties a Logs Insights query groups by. Lambda only: locally stdout is a CLI's output (the
    research CLI prints one JSON object there), and the `usage` log line already carries the figure."""
    if amount is None or not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return
    emit_emf(
        {SPEND_METRICS[kind]: round(amount, 6)},
        {"spend_kind": kind, "spend_stage": stage, "spend_model": model, **usage},
    )
