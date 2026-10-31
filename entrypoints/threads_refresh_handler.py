from __future__ import annotations

from typing import Any

import boto3
import httpx

from adapters.aws import resolve_secret, secret_parameter_name
from shared import logger, set_correlation_id

THREADS_REFRESH_URL = "https://graph.threads.net/refresh_access_token"


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Refresh the long-lived Threads access token before its 60-day expiry and write
    the renewed value back to SSM. Scheduled weekly, so a failed refresh is retried long before
    the token expires and the token is effectively permanent.

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

    # The token rides in the query string, and httpx puts the request URL into its error text, so no
    # httpx error may reach the log or the traceback: each failure is re-raised `from None` without it.
    try:
        resp = httpx.get(
            THREADS_REFRESH_URL,
            params={"grant_type": "th_refresh_token", "access_token": token},
            timeout=30,
        )
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        logger.error("Threads token refresh request failed (%s)", type(e).__name__)
        raise RuntimeError(f"Threads token refresh request failed ({type(e).__name__})") from None
    if resp.is_error:
        # The error's code and type only, never the body: it is not ours to promise it never echoes the token.
        try:
            error = resp.json().get("error", {})
        except ValueError:
            error = {}
        logger.error(
            "Threads token refresh returned HTTP %d (code=%s, type=%s)",
            resp.status_code,
            error.get("code"),
            error.get("type"),
        )
        raise RuntimeError(f"Threads token refresh failed (HTTP {resp.status_code})")
    try:
        new_token = resp.json()["access_token"]
    except (ValueError, KeyError):
        logger.error("Threads token refresh response carried no access_token")
        raise RuntimeError("Threads token refresh response carried no access_token") from None

    name = secret_parameter_name("threads-access-token")
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
    return {"statusCode": 200, "body": "refreshed"}
