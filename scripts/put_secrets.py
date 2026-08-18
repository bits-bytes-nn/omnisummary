"""Write the deployment's secrets into SSM Parameter Store as SecureStrings.

The CDK stack creates each parameter holding `SSM_PLACEHOLDER` and never the real value: a
CloudFormation template cannot hold a SecureString, so passing the tokens through the stack wrote
them in PLAINTEXT into cdk.out/*.template.json, the CDK staging bucket and every
cloudformation:GetTemplate response. This script puts the real values in out-of-band.

Run it right after `cdk deploy`. Later deploys do NOT clobber the values: CloudFormation only
updates a resource whose template properties changed, and the placeholder never changes.

    uv run python scripts/put_secrets.py [--dry-run] [--verify] [--force]

Values come from the environment (.env is loaded). A variable that is absent or empty is SKIPPED,
never blanked — so a partial .env cannot wipe a working parameter.

One parameter that cannot be written no longer aborts the run: the loop continues and prints a loud
remediation line, so every later secret still lands (and the exit code is non-zero). Use --verify
(read-only) to see which parameters are real SecureStrings, which are set but PLAINTEXT, and which
still hold the placeholder.

SSM rejects a type change on an overwrite, so a parameter CloudFormation created as a String is
DELETED and re-created as a SecureString — but only in the exact state CloudFormation leaves behind
(Type == String AND Value == SSM_PLACEHOLDER). A String holding a real value is never deleted: the
value is written in place with a loud notice, because losing a live credential is worse than storing
it unencrypted. If a delete succeeds and the re-put then fails, that parameter is reported FAILED
with the exact command to restore it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="read-only: report each parameter's type and whether it still holds the placeholder",
    )
    args = parser.parse_args()

    config = Config.load()
    project, stage = config.aws.project_name, config.aws.stage
    ssm = boto3.client("ssm", region_name=config.aws.region)

    if args.verify:
        return _verify(ssm, project, stage)

    written, skipped, unchanged, failed = [], [], [], []
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
        except ClientError as e:
            failed.append(f"{name} (could not be read: {e.response['Error'].get('Code', 'ClientError')})")
            print(f"  !! {name}: could not be read — value NOT written. Re-run this script for it.")
            continue
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
        try:
            ssm.put_parameter(Name=path, Value=value, Type="SecureString", Overwrite=True)
            written.append(f"{name} (SecureString; was {was})")
        except ClientError as e:
            code = e.response["Error"].get("Code", "")
            if code != "ValidationException":
                # NEVER abort the loop: the exception used to kill the run at the first bad
                # parameter, leaving every later secret on its CloudFormation placeholder — which
                # reads as "unset" at runtime, so the affected feature silently stopped working.
                failed.append(f"{name} ({code or e})")
                print(f"  !! {name}: put_parameter failed ({code or e}) — this secret is NOT set.")
                continue
            # A type CHANGE (String -> SecureString) is rejected with ValidationException. The
            # parameter is a String only because CloudFormation created it holding SSM_PLACEHOLDER
            # (AWS::SSM::Parameter cannot create a SecureString), so in that exact state it can be
            # deleted and re-created encrypted — which is what makes this script's SecureString
            # claim true instead of aspirational.
            if _is_recreatable_placeholder(current):
                remediation = _recreate_as_secure_string(ssm, path, value)
                if not remediation:
                    written.append(f"{name} (SecureString, deleted and re-created; was {was})")
                    continue
                # A deleted-then-unwritten parameter is WORSE than a plaintext one: the secret is
                # gone, not merely unencrypted. Report it as FAILED with the exact command that
                # restores it.
                failed.append(f"{name} (DELETED but not re-created — restore it now)")
                print(f"  !! {name}: {remediation}")
                continue
            # Any other String parameter holds a REAL value (e.g. the Threads token the refresh
            # Lambda rotates in place). NEVER delete that to change its type — losing it is worse
            # than leaving it unencrypted. Write the value in place and say so loudly.
            try:
                ssm.put_parameter(Name=path, Value=value, Overwrite=True)
                written.append(f"{name} (value only, kept existing type; was {was})")
                print(
                    f"  !! {name}: SSM refused the SecureString type change and the parameter holds a "
                    "real value, so it was NOT deleted — the value is stored as a plain String. To "
                    f"encrypt it, delete it yourself once the value is safe elsewhere "
                    f"(aws ssm delete-parameter --name {path}) and re-run this script."
                )
            except ClientError as e2:
                failed.append(f"{name} ({e2.response['Error'].get('Code', 'ClientError')})")
                print(f"  !! {name}: could not be written at all — this secret is NOT set.")

    for line in written:
        print(f"  written : {line}")
    for line in unchanged:
        print(f"  kept    : {line}")
    for line in skipped:
        print(f"  skipped : {line}")
    for line in failed:
        print(f"  FAILED  : {line}")
    print(f"\n{len(written)} written, {len(unchanged)} kept, {len(skipped)} skipped, {len(failed)} failed")
    if skipped:
        print("A skipped parameter keeps whatever value it already had; the placeholder reads as unset.")
    if failed:
        print("A FAILED parameter still holds its old value (a placeholder reads as unset) — fix and re-run.")
        return 1
    return 0


def _is_recreatable_placeholder(current: dict | None) -> bool:
    """True ONLY for the exact state CloudFormation leaves behind: a plain String parameter still
    holding SSM_PLACEHOLDER. Nothing else may ever be deleted — a String carrying a real value is
    somebody's live credential, and a SecureString needs no migration."""
    return bool(current and current.get("Type") == "String" and current.get("Value") == SSM_PLACEHOLDER)


def _recreate_as_secure_string(ssm, path: str, value: str) -> str:
    """Delete the placeholder String parameter and re-create it as a SecureString.

    Returns "" on success, or a remediation line when the parameter was deleted and the re-put
    failed — the one state worse than a plaintext secret, so the caller reports it as FAILED. A
    failed DELETE leaves the parameter untouched and is reported the same way (nothing was lost,
    but the value did not land either)."""
    try:
        ssm.delete_parameter(Name=path)
    except ClientError as e:
        code = e.response["Error"].get("Code", "ClientError")
        return (
            f"could not delete the placeholder parameter ({code}); it still holds the placeholder, "
            f"which reads as UNSET at runtime. Fix the permission and re-run this script."
        )
    try:
        ssm.put_parameter(Name=path, Value=value, Type="SecureString", Overwrite=True)
    except ClientError as e:
        code = e.response["Error"].get("Code", "ClientError")
        return (
            f"the placeholder was DELETED but the SecureString could not be written ({code}) — the "
            f"parameter does NOT exist now. Restore it immediately with:\n"
            f"     aws ssm put-parameter --name {path} --type SecureString --value '<value>' --overwrite"
        )
    return ""


def _verify(ssm, project: str, stage: str) -> int:
    """Read-only report: which secret parameters are set, and which still hold the placeholder.
    Nothing is written, so it is safe to run against prod at any time."""
    placeholders, ok, plaintext, missing = [], [], [], []
    for name in ALL_SSM_SECRET_ENV_VARS:
        path = f"/{project}/{stage}/{name}"
        try:
            param = ssm.get_parameter(Name=path, WithDecryption=True)["Parameter"]
        except ssm.exceptions.ParameterNotFound:
            missing.append(name)
            continue
        except ClientError as e:
            missing.append(f"{name} (unreadable: {e.response['Error'].get('Code', 'ClientError')})")
            continue
        if param["Value"] == SSM_PLACEHOLDER:
            placeholders.append(f"{name} ({param['Type']})")
        elif param["Type"] != "SecureString":
            # Set, but NOT encrypted at rest: the state this script exists to eliminate. Reported
            # separately (and non-zero) so "the secrets are SecureStrings" is a checkable claim.
            plaintext.append(f"{name} ({param['Type']})")
        else:
            ok.append(f"{name} ({param['Type']})")
    for line in ok:
        print(f"  set         : {line}")
    for line in plaintext:
        print(f"  PLAINTEXT   : {line}")
    for line in placeholders:
        print(f"  PLACEHOLDER : {line}")
    for line in missing:
        print(f"  MISSING     : {line}")
    print(f"\n{len(ok)} set, {len(plaintext)} plaintext, {len(placeholders)} placeholder, {len(missing)} missing")
    if placeholders or missing:
        print("A placeholder/missing parameter reads as UNSET at runtime — run this script without --verify.")
    if plaintext:
        print("A PLAINTEXT parameter is set but unencrypted at rest — re-run this script to migrate it.")
    if placeholders or missing or plaintext:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
