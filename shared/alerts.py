from __future__ import annotations

import os

import boto3

from .constants import DEFAULT_PROJECT_NAME, EnvVars
from .formatting import format_alarm
from .logger import get_correlation_id, logger


def publish_alert(event: str, status: str, fields: dict[str, str]) -> None:
    """Publish ONE SNS notice in the project's unified alarm format, or nothing when no topic is
    wired (local runs, un-wired stages).

    Never raises: an alerting failure must not fail the run it is reporting on. Both Lambda handlers
    carried a byte-identical copy of this routine, so a fix to the alert path could land in only one
    of them; they now own only their `fields` assembly.

    project/stage come from the function's own env, so a dev alert can't read as a prod one, and the
    correlation id lets the operator jump straight to this run's log lines."""
    topic_arn = os.environ.get("ALERT_SNS_TOPIC_ARN", "")
    if not topic_arn:
        return
    try:
        subject, message = format_alarm(
            event=event,
            status=status,
            fields=fields,
            project=os.environ.get(EnvVars.PROJECT_NAME.value, DEFAULT_PROJECT_NAME),
            stage=os.environ.get(EnvVars.STAGE.value, ""),
            correlation_id=get_correlation_id(),
        )
        boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject, Message=message)
        logger.warning("Published SNS alert (%s): %s", event, fields)
    except Exception as e:
        logger.error("Failed to publish SNS alert (%s): %s", event, e)
