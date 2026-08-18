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
    name = f"/{project}/{stage}/threads-access-token"
    ssm = boto3.client("ssm")
    try:
        # No Type: passing Type=SecureString on an overwrite is a type CHANGE, which SSM rejects
        # with ValidationException — and a rejected write leaves the token un-refreshed until it
        # expires. Omitting Type keeps whatever type the parameter has (SecureString once
        # scripts/put_secrets.py has migrated it) and only updates the value.
        ssm.put_parameter(Name=name, Value=new_token, Overwrite=True)
        logger.info("Refreshed Threads access token and updated SSM")
    except Exception as e:
        logger.error("Failed to write refreshed Threads token to SSM: %s", e)
        raise
    _warn_if_unencrypted(ssm, name)
    return {"statusCode": 200, "body": "refreshed"}


def _warn_if_unencrypted(ssm: Any, name: str) -> None:
    """Say so when the refreshed token is sitting in a plain String parameter. The write above
    deliberately omits Type, so it silently PRESERVES an unencrypted parameter — which is exactly
    the state scripts/put_secrets.py exists to remove. Best-effort: a failed check must never turn a
    successful refresh into an error."""
    try:
        param_type = ssm.get_parameter(Name=name)["Parameter"]["Type"]
    except Exception as e:
        logger.warning("Could not verify the type of '%s': %s", name, e)
        return
    if param_type != "SecureString":
        logger.error(
            "Threads access token is stored as a plain %s (unencrypted at rest) in '%s' — "
            "run scripts/put_secrets.py to migrate it to a SecureString",
            param_type,
            name,
        )
