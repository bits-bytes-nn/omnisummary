"""Write the deployment's secrets into SSM Parameter Store as SecureStrings.

The CDK stack creates each parameter holding `SSM_PLACEHOLDER` and never the real value: a
CloudFormation template cannot hold a SecureString, so passing the tokens through the stack wrote
them in PLAINTEXT into cdk.out/*.template.json, the CDK staging bucket and every
cloudformation:GetTemplate response. This script puts the real values in out-of-band.

Run it right after `cdk deploy`. Later deploys do NOT clobber the values: CloudFormation only
updates a resource whose template properties changed, and the placeholder never changes.

    uv run python scripts/put_secrets.py [--dry-run]

Values come from the environment (.env is loaded). A variable that is absent or empty is SKIPPED,
never blanked — so a partial .env cannot wipe a working parameter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import Config  # noqa: E402
from shared.constants import ALL_SSM_SECRET_ENV_VARS, SSM_PLACEHOLDER  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="also overwrite parameters that are ALREADY SecureStrings. Off by default because the "
        "live value can be newer than .env — the Threads token is rotated in place by the refresh "
        "Lambda, so reasserting the local copy would restore an expired token.",
    )
    args = parser.parse_args()

    config = Config.load()
    project, stage = config.aws.project_name, config.aws.stage
    ssm = boto3.client("ssm", region_name=config.aws.region)

    written, skipped, unchanged = [], [], []
    for name, env_var in ALL_SSM_SECRET_ENV_VARS.items():
        value = os.getenv(env_var, "").strip()
        path = f"/{project}/{stage}/{name}"
        if not value:
            skipped.append(f"{name} (${env_var} not set)")
            continue
        try:
            current = ssm.get_parameter(Name=path, WithDecryption=True)["Parameter"]
        except ssm.exceptions.ParameterNotFound:
            current = None
        if current and current["Type"] == "SecureString" and not args.force:
            # Already migrated, and the live value may be NEWER than .env: the Threads access token
            # is rotated in place by the refresh Lambda, so writing the local copy back would
            # restore an expired token. Leave it alone unless --force.
            unchanged.append(
                name + (" (matches .env)" if current["Value"] == value else " (live value differs from .env)")
            )
            continue
        if not current:
            was = "absent"
        else:
            was = current["Type"] + ("/placeholder" if current["Value"] == SSM_PLACEHOLDER else "")
        if args.dry_run:
            written.append(f"{name} (would write SecureString; currently {was})")
            continue
        ssm.put_parameter(Name=path, Value=value, Type="SecureString", Overwrite=True)
        written.append(f"{name} (SecureString; was {was})")

    for line in written:
        print(f"  written : {line}")
    for line in unchanged:
        print(f"  kept    : {line}")
    for line in skipped:
        print(f"  skipped : {line}")
    print(f"\n{len(written)} written, {len(unchanged)} kept, {len(skipped)} skipped")
    if skipped:
        print("A skipped parameter keeps whatever value it already had; the placeholder reads as unset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
