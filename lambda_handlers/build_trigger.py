import json
import time
from typing import Any

import boto3
import urllib3
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError

FAILED_STATUSES: list[str] = ["FAILED", "FAULT", "TIMED_OUT"]
POLL_INTERVAL_SECONDS: int = 10
MAX_WAIT_TIME_SECONDS: int = 300

codebuild = boto3.client("codebuild")
http = urllib3.PoolManager()


def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    request_type = event.get("RequestType")
    project_name = event["ResourceProperties"]["ProjectName"]

    if request_type == "Create" or request_type == "Update":
        try:
            response = codebuild.start_build(projectName=project_name)
            build_id = response["build"]["id"]
            start_time = time.time()

            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > MAX_WAIT_TIME_SECONDS:
                    error_msg = f"Build timeout after {int(elapsed_time)} seconds"
                    print(error_msg)
                    return send_response(event, context, "FAILED", {"BuildId": build_id, "Error": error_msg})

                build_response = codebuild.batch_get_builds(ids=[build_id])
                build_status = build_response["builds"][0]["buildStatus"]

                if build_status == "SUCCEEDED":
                    return send_response(event, context, "SUCCESS", {"BuildId": build_id})
                elif build_status in FAILED_STATUSES:
                    return send_response(event, context, "FAILED", {"BuildId": build_id})

                time.sleep(POLL_INTERVAL_SECONDS)

        except ClientError as e:
            return send_response(event, context, "FAILED", {"Error": str(e)})

    return send_response(event, context, "SUCCESS", {})


def send_response(event: dict[str, Any], context: LambdaContext, status: str, data: dict[str, Any]) -> dict[str, Any]:
    response_body = json.dumps(
        {
            "Status": status,
            "Reason": f"See the details in CloudWatch Log Stream: {context.log_stream_name}",
            "PhysicalResourceId": context.log_stream_name,
            "StackId": event["StackId"],
            "RequestId": event["RequestId"],
            "LogicalResourceId": event["LogicalResourceId"],
            "Data": data,
        }
    )

    try:
        http.request(
            "PUT",
            event["ResponseURL"],
            body=response_body,
            headers={"Content-Type": "application/json"},
        )
    except urllib3.exceptions.HTTPError as e:
        print(f"Failed to send CFN response: {e}")

    return {"statusCode": 200, "body": json.dumps("SUCCESS")}
