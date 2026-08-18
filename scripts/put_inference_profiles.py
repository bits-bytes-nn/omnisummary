"""Create this project's Bedrock APPLICATION inference profiles, tagged for cost allocation.

On-demand Bedrock has no taggable resource behind InvokeModel, so its token spend cannot be
attributed to a cost-allocation tag — in a shared account (this one bills several workloads against
the same Claude models) the Bedrock line is a single unattributable total. An application inference
profile IS taggable, and invoking through its ARN attributes the usage to those tags.

Not CloudFormation: the profiles must live in `aws.bedrock_region` (us-west-2) while the stacks
deploy to `aws.region` (ap-northeast-2), and a second regional stack plus its bootstrap is
disproportionate for a reporting concern. Same reasoning as scripts/put_secrets.py.

    uv run python scripts/put_inference_profiles.py [--dry-run] [--delete]

Idempotent: an existing profile with the expected name is left alone. The runtime looks profiles up
by that name and falls back to the system-defined inference profile when none exists, so running
this is optional — it only changes how the usage is BILLED, never whether a call works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import Config  # noqa: E402
from shared.constants import LanguageModelId  # noqa: E402
from shared.utils import BedrockCrossRegionModelHelper  # noqa: E402


def _configured_models(config: Config) -> dict[LanguageModelId, str]:
    """Every model this deployment can invoke, with the setting that selects it. A profile is only
    worth creating for models actually in use — the registry carries many that are not."""
    pipeline, agent, collectors = config.pipeline, config.agent, config.collectors
    wanted: dict[LanguageModelId, list[str]] = {}
    for setting, model in (
        ("pipeline.ranking_model", pipeline.ranking_model),
        ("pipeline.digest_model", pipeline.digest_model),
        ("pipeline.trend_model", pipeline.trend_model),
        ("agent.model_id", agent.model_id),
        ("collectors.web_search.refine_model", collectors.web_search.refine_model),
    ):
        wanted.setdefault(model, []).append(setting)
    return {model: ", ".join(settings) for model, settings in wanted.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--delete", action="store_true", help="delete this project's profiles instead")
    args = parser.parse_args()

    config = Config.load()
    region = config.aws.bedrock_region
    session = boto3.Session(region_name=region, profile_name=config.aws.profile or None)
    client = session.client("bedrock", region_name=region)

    existing: dict[str, str] = {}
    paginator = client.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="APPLICATION"):
        for summary in page.get("inferenceProfileSummaries", []):
            existing[summary["inferenceProfileName"]] = summary["inferenceProfileArn"]

    models = _configured_models(config)
    print(f"region {region} | {len(models)} configured model(s) | {len(existing)} existing application profile(s)\n")

    for model, settings in sorted(models.items(), key=lambda kv: kv[0].value):
        name = BedrockCrossRegionModelHelper.application_profile_name(model)
        arn = existing.get(name)
        if args.delete:
            if not arn:
                print(f"  absent  : {name}")
                continue
            if args.dry_run:
                print(f"  would delete: {name}")
                continue
            client.delete_inference_profile(inferenceProfileIdentifier=arn)
            print(f"  deleted : {name}")
            continue
        if arn:
            print(f"  exists  : {name}  ({settings})")
            continue
        # copyFrom takes the SYSTEM-DEFINED cross-region profile the runtime would otherwise use, so
        # the application profile inherits the same routing rather than pinning one region.
        source = BedrockCrossRegionModelHelper._resolve(session, model, region)
        source_arn = (
            source
            if source.startswith("arn:")
            else f"arn:aws:bedrock:{region}:{session.client('sts').get_caller_identity()['Account']}:inference-profile/{source}"
        )
        if args.dry_run:
            print(f"  would create: {name}  from {source}  ({settings})")
            continue
        created = client.create_inference_profile(
            inferenceProfileName=name,
            modelSource={"copyFrom": source_arn},
            tags=[
                {"key": "Project", "value": config.aws.project_name},
                {"key": "Stage", "value": config.aws.stage},
            ],
        )
        print(f"  created : {name}  -> {created['inferenceProfileArn']}  ({settings})")

    if not args.delete and not args.dry_run:
        print(
            "\nActivate the `Project` cost allocation tag in Billing for these to show up in Cost "
            "Explorer (up to 24h, not retroactive)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
