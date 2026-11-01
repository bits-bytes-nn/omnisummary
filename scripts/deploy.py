#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_cdk import App, Environment
from dotenv import load_dotenv

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def main():
    # Not Config.load(): its template fallback (a WARNING) deployed a fresh worktree's placeholders,
    # an empty state bucket and source lists included.
    if not CONFIG_PATH.exists():
        raise SystemExit(f"{CONFIG_PATH} is missing: deploying without it ships the template's placeholders")
    config = Config.from_yaml(str(CONFIG_PATH))
    # Every other edition with a local config/<edition>.yaml gets its own digest and visual Lambdas.
    # Pair like with like: config.yaml with the local edition files, a template with the edition templates.
    suffix = "-template" if CONFIG_PATH.stem.endswith("-template") else ""
    editions = tuple(
        edition
        for path in sorted(CONFIG_PATH.parent.glob("*.yaml"))
        if path != CONFIG_PATH and path.stem.endswith("-template") == bool(suffix)
        if (edition := Config.from_yaml(str(path))).edition
    )
    for edition in editions:
        # The foundation (IAM scope, invoke ARNs, alarm dimensions) is built from the AI/ML config, and
        # each edition's Lambda loads config/<edition>.yaml: anything else would deploy functions that are
        # denied their own state or load another file.
        problems = [
            f"{key} differs from config.yaml"
            for key in ("project_name", "stage", "region", "state_bucket_name")
            if getattr(edition.aws, key) != getattr(config.aws, key)
        ]
        root = f"{config.aws.s3_prefix.strip('/')}/" if config.aws.s3_prefix.strip("/") else ""
        own = edition.aws.s3_prefix.strip("/")
        if not own.startswith(root) or own == root.rstrip("/") or own.split("/")[-1] == "digest_state":
            problems.append(f"s3_prefix '{edition.aws.s3_prefix}' is not its own prefix under '{config.aws.s3_prefix}'")
        if not (CONFIG_PATH.parent / f"{edition.edition}{suffix}.yaml").exists():
            problems.append(f"it is not saved as config/{edition.edition}{suffix}.yaml")
        if problems:
            raise SystemExit(f"Edition '{edition.edition}': " + "; ".join(problems))
    load_dotenv(CONFIG_PATH.parent.parent / ".env")
    # The alert email lives only in .env: a deploy without it REMOVES the SNS subscription, and every
    # alert (a failed Threads post included) then reaches nobody, with no error anywhere.
    if not os.getenv("ALERT_EMAIL"):
        raise SystemExit("ALERT_EMAIL is unset (.env): deploying would remove the alert email subscription")

    # The account the CDK CLI resolved from ITS credentials (the ones it deploys with), not a second
    # STS call through config.aws.profile, which can name a different account.
    account = os.environ.get("CDK_DEFAULT_ACCOUNT")
    if not account:
        raise SystemExit("CDK_DEFAULT_ACCOUNT is unset: run this through the CDK CLI (npx cdk deploy -a ...)")
    env = Environment(account=account, region=config.aws.region)

    app = App()

    foundation = OmniSummaryFoundationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-foundation",
        config=config,
        alert_email=os.getenv("ALERT_EMAIL", ""),
        editions=tuple(edition.edition for edition in editions),
        env=env,
    )

    image_ref = os.getenv("DIGEST_IMAGE_REF", "")
    if os.getenv("FOUNDATION_ONLY") == "1":
        # Bootstrap and a fresh account's first deploy, before any image exists. Never for an update of a
        # deployed account: the foundation alone drops the exports the live application stack imports.
        app.synth()
        return
    if not image_ref.startswith("sha256:"):
        # Unpinned, the Lambdas would reference `:latest`, and a deploy that changed anything else on a
        # function moved it to whatever was pushed last.
        raise SystemExit(
            f"DIGEST_IMAGE_REF is unset or not a sha256 digest ({image_ref!r}): export the pushed image's digest (README, Deploying to AWS), or set "
            "FOUNDATION_ONLY=1 for bootstrap and a fresh account's first foundation deploy."
        )

    OmniSummaryApplicationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-application",
        config=config,
        foundation=foundation,
        # No secrets are passed in: a CloudFormation template cannot hold a SecureString, so the
        # values would land in plaintext in cdk.out, the CDK staging bucket and GetTemplate. The
        # stack creates the parameters holding a placeholder; run scripts/put_secrets.py after the
        # deploy to write the real values as SecureStrings.
        digest_image_ref=image_ref,
        editions=editions,
        env=env,
    )

    app.synth()


if __name__ == "__main__":
    main()
