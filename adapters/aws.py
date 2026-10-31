"""AWS session and region resolution, and secrets: env first, then SSM Parameter Store."""

import os

import boto3

from shared.config import get_config
from shared.constants import SSM_PLACEHOLDER
from shared.logger import logger


def _is_parameter_not_found(exc: BaseException) -> bool:
    """True when SSM answered that the parameter does not exist (as opposed to failing to answer)."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code") == "ParameterNotFound"
    return False


def aws_region() -> str | None:
    """Region for a boto3 client built OUTSIDE a Session: the environment's, else the config's,
    else None so boto3 resolves it from the profile as usual.

    Four modules carried the literal
    ``os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))`` — one developer's
    region baked into the code, free to diverge from config.aws.region and silently sending an SSM or
    AgentCore call to the wrong region in any other deployment. Lambda and AgentCore always set
    AWS_REGION; the config covers a local run."""
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if region:
        return region
    try:
        return get_config().aws.region or None
    except Exception:
        logger.warning("Could not read aws.region from config; leaving the region to boto3", exc_info=True)
        return None


def available_boto_profile(profile: str) -> str | None:
    """`profile` when boto3 can actually resolve it here, else None — i.e. "use ambient credentials".

    Picking the credential source by PLATFORM SNIFF was wrong in both directions. is_running_in_aws()
    looks for AWS_EXECUTION_ENV / AWS_LAMBDA_FUNCTION_NAME / AWS_BATCH_JOB_ID /
    ECS_CONTAINER_METADATA_URI, and a managed runtime that sets none of them sent a container with no
    ~/.aws/config down the named-profile branch, where the session could not be built at all.

    Ambient credentials are what every managed runtime supplies, so they are the default; the
    configured profile is honoured exactly where it exists, which is the developer laptop it was
    added for. Any failure to enumerate profiles also means ambient — cost attribution and
    convenience must never be able to stop a run."""
    if not profile:
        return None
    try:
        if profile in boto3.session.Session().available_profiles:
            return profile
    except Exception as e:
        logger.debug("Could not enumerate AWS profiles (%s); using ambient credentials", e)
        return None
    logger.info("AWS profile '%s' is not configured here; using ambient credentials", profile)
    return None


def _ssm_client():
    """SSM on the configured profile and region where that profile exists (a developer laptop, whose
    shell may carry AWS_PROFILE/AWS_REGION for another account or the model region: the lookup then
    answered ParameterNotFound), else ambient credentials in the runtime's own region."""
    config = get_config()
    profile = available_boto_profile(config.aws.profile)
    if profile:
        return boto3.Session(profile_name=profile, region_name=config.aws.region or None).client("ssm")
    return boto3.client("ssm", region_name=aws_region())


class SecretUnavailableError(RuntimeError):
    """SSM could not be read (outage, throttle, denied) — as opposed to the secret being unset.
    Only raised for callers that opt into strict=True; every other caller keeps the ""-degrades
    contract it was written against."""


def resolve_secret(env_var: str, ssm_suffix: str, *, strict: bool = False) -> str:
    """Resolve a secret from env first, then SSM Parameter Store.

    SSM path is /{PROJECT_NAME}/{STAGE}/{ssm_suffix} (SecureString-decrypted).
    Returns "" if unavailable from either source (callers degrade gracefully).

    strict=True instead RAISES SecretUnavailableError when SSM itself could not be read, so a
    caller whose whole job is that secret (the Threads token refresh) fails loudly rather than
    reporting "nothing to refresh" — which is indistinguishable from "no token configured" and
    quietly let the token lapse. A parameter that is genuinely absent/placeholder still returns "".
    """
    value = os.getenv(env_var, "")
    if value:
        return value

    project = os.getenv("PROJECT_NAME", "omnisummary")
    stage = os.getenv("STAGE", "dev")
    try:
        ssm = _ssm_client()
        resolved = ssm.get_parameter(Name=f"/{project}/{stage}/{ssm_suffix}", WithDecryption=True)["Parameter"]["Value"]
    except Exception as e:
        if strict and not _is_parameter_not_found(e):
            raise SecretUnavailableError(f"Could not read secret '{ssm_suffix}' from SSM: {e}") from e
        logger.warning("Secret '%s' unavailable (env + SSM '%s'): %s", env_var, ssm_suffix, e)
        return ""
    # The CDK stack creates each parameter holding a placeholder (never a real secret in the
    # CloudFormation template); scripts/put_secrets.py writes the value. If that step was skipped,
    # treat it as unset so the caller takes its normal missing-credential path instead of sending
    # the placeholder to an API as a token.
    if resolved == SSM_PLACEHOLDER:
        logger.warning("Secret '%s' is still the SSM placeholder — run scripts/put_secrets.py", ssm_suffix)
        return ""
    return resolved
