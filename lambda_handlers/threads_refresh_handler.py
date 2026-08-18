from __future__ import annotations

import os
from typing import Any

import boto3
import httpx

from shared import logger, resolve_secret, set_correlation_id

THREADS_REFRESH_URL = "https://graph.threads.net/refresh_access_token"


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Refresh the long-lived Threads access token before its 60-day expiry and write
    the renewed value back to SSM. Scheduled well inside the window (e.g. every 50 days)
    so the token is effectively permanent.

    A failure is logged and RE-RAISED: a silent 500 body left the token quietly un-refreshed
    until Threads delivery broke 60 days later, because Lambda counts a returned error as a
    success (no Errors alarm, no DLQ message). The function runs with retry_attempts=0, so
    re-raising cannot loop.

    The token is resolved strictly: an SSM read failure raises instead of returning "", which this
    handler would otherwise report as "no token configured, nothing to refresh" — a 200 that leaves
    the real token to expire."""
    set_correlation_id(getattr(context, "aws_request_id", "") or None)
    token = resolve_secret("THREADS_ACCESS_TOKEN", "threads-access-token", strict=True)
    if not token:
        logger.info("No Threads access token configured, nothing to refresh")
        return {"statusCode": 200, "body": "no token"}

    try:
        resp = httpx.get(
            THREADS_REFRESH_URL,
            params={"grant_type": "th_refresh_token", "access_token": token},
            timeout=30,
        )
        resp.raise_for_status()
        new_token = resp.json()["access_token"]
    except Exception as e:
        logger.error("Failed to refresh Threads token: %s", e)
        raise

    project = os.environ.get("PROJECT_NAME", "omnisummary")
    stage = os.environ.get("STAGE", "dev")
    try:
        # No Type: the parameter is CREATED by CloudFormation, and AWS::SSM::Parameter cannot
        # create a SecureString — so it exists as a String. Passing Type=SecureString on an
        # overwrite is a type CHANGE, which SSM rejects with ValidationException, leaving the
        # token un-refreshed. Omitting Type keeps the existing type and only updates the value.
        boto3.client("ssm").put_parameter(
            Name=f"/{project}/{stage}/threads-access-token",
            Value=new_token,
            Overwrite=True,
        )
        logger.info("Refreshed Threads access token and updated SSM")
        return {"statusCode": 200, "body": "refreshed"}
    except Exception as e:
        logger.error("Failed to write refreshed Threads token to SSM: %s", e)
        raise
