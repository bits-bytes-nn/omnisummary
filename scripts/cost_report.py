#!/usr/bin/env python3
"""Spend per day, Lambda and stage, read back from the deployed Lambdas' usage log lines (`LLM usage`,
`Image usage`, `Search usage`). Read-only: one CloudWatch Logs Insights query per day of the window
(billed per GB scanned, cents for these log groups).

Cost Explorer cannot answer this: both editions share the tagged inference profiles, and the bill is per
model while several stages share one. Bedrock prices are the global-profile list price; Tavily is
reported in credits because a credit's price depends on the plan.

Usage: uv run python scripts/cost_report.py [--days 7] [--by stage|function]
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.aws import available_boto_profile
from shared import Config

FUNCTIONS = ("digest", "visual", "invest-digest", "invest-visual", "comment-replies")
# Logs Insights' row cap: a longer window is queried in several pieces rather than silently cut.
ROW_LIMIT = 10000
QUERY = r"""
filter @message like /(LLM|Image|Search) usage stage=/
| parse @message /(?<s_kind>LLM|Image|Search) usage stage=(?<s_stage>[^ "]+)/
| parse @message / usd=(?<s_usd>[0-9.]+)/
| parse @message / credits=(?<s_credits>[0-9.]+)/
| display @timestamp, @log, s_kind, s_stage, s_usd, s_credits
| limit {limit}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--by", choices=("stage", "function"), default="function")
    args = parser.parse_args()

    import boto3

    config = Config.load()
    session = boto3.Session(profile_name=available_boto_profile(config.aws.profile), region_name=config.aws.region)
    logs = session.client("logs")
    prefix = f"/aws/lambda/{config.aws.project_name}-{config.aws.stage}-"
    end = datetime.now(UTC)
    existing = {g["logGroupName"] for g in logs.describe_log_groups(logGroupNamePrefix=prefix)["logGroups"]}
    groups = [prefix + name for name in FUNCTIONS if prefix + name in existing]
    if not groups:
        sys.exit(f"No log groups under {prefix}")
    rows: list = []
    for back in range(args.days):
        day_end = end - timedelta(days=back)
        query_id = logs.start_query(
            logGroupNames=groups,
            startTime=int((day_end - timedelta(days=1)).timestamp()),
            endTime=int(day_end.timestamp()),
            queryString=QUERY.format(limit=ROW_LIMIT),
        )["queryId"]
        while (result := logs.get_query_results(queryId=query_id))["status"] in ("Scheduled", "Running"):
            time.sleep(1)
        if result["status"] != "Complete":
            sys.exit(f"Logs Insights query ended {result['status']}")
        if len(result["results"]) >= ROW_LIMIT:
            sys.exit(f"A day ending {day_end:%Y-%m-%d %H:%M} UTC hit the {ROW_LIMIT}-row cap; totals would be low")
        rows += result["results"]

    tz = ZoneInfo(config.aws.timezone)
    usd: dict[tuple[str, str], float] = defaultdict(float)
    credits: dict[tuple[str, str], float] = defaultdict(float)
    unpriced: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        f = {c["field"]: c["value"] for c in row}
        day = datetime.fromisoformat(f["@timestamp"].replace(" ", "T") + "+00:00").astimezone(tz).date().isoformat()
        function = f["@log"].rsplit(prefix, 1)[-1]
        key = (day, function if args.by == "function" else f"{f['s_kind'].lower()}:{f['s_stage']}")
        if "s_credits" in f:
            credits[key] += float(f["s_credits"])
        elif "s_usd" in f:
            usd[key] += float(f["s_usd"])
        elif f["s_kind"] != "Search":
            unpriced[key] += 1

    keys = sorted(set(usd) | set(credits) | set(unpriced))
    total: dict[str, float] = defaultdict(float)
    for day, name in keys:
        note = f"  ({unpriced[(day, name)]} unpriced calls)" if unpriced[(day, name)] else ""
        tavily = f"  tavily {credits[(day, name)]:.0f} credits" if credits[(day, name)] else ""
        print(f"{day}  {name:28} ${usd[(day, name)]:8.3f}{tavily}{note}")
        total[day] += usd[(day, name)]
    print()
    for day, amount in sorted(total.items()):
        print(f"{day}  {'total (Bedrock + OpenAI)':28} ${amount:8.3f}")
    print(f"\n{args.days} days: ${sum(total.values()):.2f}; Tavily {sum(credits.values()):.0f} credits")


if __name__ == "__main__":
    main()
