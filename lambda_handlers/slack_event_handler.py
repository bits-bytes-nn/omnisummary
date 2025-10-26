import hashlib
import hmac
import json
import os
import time
from typing import Any

import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError

logger = Logger()

AGENTCORE_RUNTIME_ARN = os.getenv("AGENTCORE_RUNTIME_ARN")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME")
PROJECT_NAME = os.getenv("PROJECT_NAME", "omnisummary")
STAGE = os.getenv("STAGE", "dev")

EVENT_DEDUPLICATION_TTL_SEC: int = int(os.getenv("EVENT_DEDUPLICATION_TTL_SEC", "300"))
SLACK_SIGNATURE_EXPIRATION_SEC: int = int(os.getenv("SLACK_SIGNATURE_EXPIRATION_SEC", "300"))
SSM_PARAM_NAME_PREFIX: str = f"/{PROJECT_NAME}/{STAGE}"

boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
agentcore_client = boto_session.client("bedrock-agentcore")
dynamodb_client = boto_session.client("dynamodb")
lambda_client = boto_session.client("lambda")
ssm_client = boto_session.client("ssm")

_slack_signing_secret_cache: str | None = None


def get_ssm_param_value(boto_session: boto3.Session, param_name: str) -> str:
    try:
        ssm_client = boto_session.client("ssm")
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
        value = response["Parameter"]["Value"]
        if not isinstance(value, str):
            raise RuntimeError(f"SSM parameter '{param_name}' value is not a string")
        return value
    except ClientError as e:
        raise RuntimeError(f"Failed to get SSM parameter '{param_name}'") from e


def is_duplicate_event(event_id: str) -> bool:
    if not DDB_TABLE_NAME:
        logger.warning("DDB_TABLE_NAME not configured. Skipping deduplication.")
        return False

    try:
        response = dynamodb_client.get_item(
            TableName=DDB_TABLE_NAME, Key={"event_id": {"S": event_id}}, ConsistentRead=True
        )

        if "Item" in response:
            logger.info("Duplicate event detected via DynamoDB: '%s'. Skipping processing.", event_id)
            return True

        ttl_value = int(time.time()) + EVENT_DEDUPLICATION_TTL_SEC
        dynamodb_client.put_item(
            TableName=DDB_TABLE_NAME,
            Item={"event_id": {"S": event_id}, "ttl": {"N": str(ttl_value)}, "timestamp": {"N": str(int(time.time()))}},
        )
        logger.debug("Event '%s' recorded in DynamoDB for deduplication", event_id)
        return False

    except ClientError as e:
        logger.error("DynamoDB error during deduplication check: %s", e)
        return False


def verify_slack_request(event: dict[str, Any]) -> bool:
    global _slack_signing_secret_cache
    if _slack_signing_secret_cache is None:
        _slack_signing_secret_cache = get_ssm_param_value(boto_session, f"{SSM_PARAM_NAME_PREFIX}/slack-signing-secret")

    slack_signing_secret = _slack_signing_secret_cache
    if not slack_signing_secret:
        logger.warning("Slack signing secret is not configured. Skipping signature verification.")
        return True

    try:
        headers = event.get("headers", {})
        timestamp = headers.get("X-Slack-Request-Timestamp")
        signature = headers.get("X-Slack-Signature")
        body = event.get("body", "")

        if not timestamp or not signature:
            logger.warning("Missing Slack signature headers")
            return False

        if abs(time.time() - int(timestamp)) > SLACK_SIGNATURE_EXPIRATION_SEC:
            logger.warning("Slack request timestamp exceeds threshold")
            return False

        sig_basestring = f"v0:{timestamp}:{body}".encode()
        secret_bytes = slack_signing_secret.encode()
        my_signature = "v0=" + hmac.new(secret_bytes, sig_basestring, hashlib.sha256).hexdigest()

        if hmac.compare_digest(my_signature, signature):
            return True
        else:
            logger.warning("Slack signature mismatch")
            return False

    except (ValueError, TypeError, KeyError) as e:
        logger.warning("Error verifying Slack signature: %s", e)
        return False


def _handle_async_invocation(event: dict[str, Any]) -> dict[str, Any]:
    logger.info("Processing async AgentCore invocation")
    text = event.get("text")
    channel = event.get("channel")

    if not text or not channel:
        logger.warning("Missing required parameters: text='%s', channel='%s'", text, channel)
        return {"statusCode": 400, "body": json.dumps({"error": "Missing required parameters"})}

    invocation_id = hashlib.sha256(f"{text}:{channel}".encode()).hexdigest()
    if is_duplicate_event(f"invocation:{invocation_id}"):
        logger.info("Duplicate AgentCore invocation detected: '%s'. Skipping.", invocation_id)
        return {"statusCode": 200, "body": json.dumps({"ok": True})}

    invoke_agentcore_runtime(text, channel)
    return {"statusCode": 200, "body": json.dumps({"ok": True})}


def _handle_slack_event(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    if not verify_slack_request(event):
        logger.warning("Invalid Slack request signature")
        return {"statusCode": 401, "body": json.dumps({"error": "Unauthorized"})}

    body = json.loads(event.get("body", "{}"))
    event_type = body.get("type")

    if event_type == "url_verification":
        logger.info("URL verification request - responding with challenge")
        return {
            "statusCode": 200,
            "body": json.dumps({"challenge": body.get("challenge")}),
            "headers": {"Content-Type": "application/json"},
        }

    event_data = body.get("event", {})
    event_subtype = event_data.get("type")
    event_id = body.get("event_id")

    logger.info("Received event type: '%s', event_id: '%s'", event_subtype, event_id)

    if event_subtype == "app_mention":
        if event_id and is_duplicate_event(event_id):
            return {"statusCode": 200, "body": json.dumps({"ok": True})}

        text = event_data.get("text", "")
        channel = event_data.get("channel")

        if text and text.strip():
            logger.info("Received app_mention: '%s'", text[:100])
            try:
                lambda_client.invoke(
                    FunctionName=context.function_name,
                    InvocationType="Event",
                    Payload=json.dumps(
                        {
                            "action": "invoke_agentcore",
                            "text": text,
                            "channel": channel,
                        }
                    ),
                )
                logger.info("Lambda async invocation triggered for AgentCore")
            except Exception as e:
                logger.error("Failed to trigger async Lambda invocation: %s", e)
    else:
        logger.info("Ignoring event type: '%s'", event_subtype)

    return {"statusCode": 200, "body": json.dumps({"ok": True})}


@logger.inject_lambda_context
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    logger.info("Received event: '%s'", json.dumps(event))

    try:
        if event.get("action") == "invoke_agentcore":
            return _handle_async_invocation(event)
        else:
            return _handle_slack_event(event, context)

    except Exception as e:
        logger.error("Error processing event: %s", e, exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def invoke_agentcore_runtime(message: str, channel_id: str) -> None:
    if not AGENTCORE_RUNTIME_ARN:
        logger.error("'AGENTCORE_RUNTIME_ARN' environment variable is not configured")
        return

    try:
        payload = {
            "prompt": message,
            "channel_id": channel_id,
        }

        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=AGENTCORE_RUNTIME_ARN,
            qualifier="DEFAULT",
            payload=json.dumps(payload),
        )

        logger.info("AgentCore Runtime invoked successfully for message: '%s'", message[:100])
        logger.debug("Response: %s", response)

    except Exception as e:
        logger.error("Failed to invoke AgentCore Runtime for message '%s': %s", message[:100], e)
