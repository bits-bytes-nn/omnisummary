"""What every Lambda handler reads off its invocation: the time it has left and the event a DLQ replay
wraps."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any


def remaining_deadline(context: Any) -> float | None:
    """A plain monotonic deadline derived from the Lambda's own remaining time, or None when there is
    no context (local runs, tests): the deadline is optional everywhere downstream. The CONTEXT OBJECT
    never leaves this function; only a float is threaded through."""
    remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
    if remaining_ms is None:
        return None
    try:
        return time.monotonic() + float(remaining_ms()) / 1000.0
    except Exception:
        return None


def invoke_payload(event: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """The payload carrying any of `keys`. A DLQ replay hands back the failed invoke's envelope, whose
    original payload sits under `requestPayload`: read that too, so a replay is not silently re-dated
    or stripped of its correlation id."""
    if not (set(keys) & event.keys()) and isinstance(event.get("requestPayload"), dict):
        return event["requestPayload"]
    return event
