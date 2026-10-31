"""Read back what the daily digest has already published, for OpenClaw and other local callers.

`digest [DATE]` prints that day's lead and stories; `trends` prints the tracked cross-day trends,
strongest first, each with its latest evidence only (all of it ran to ~70KB of JSON). Read-only
and free (no model call). Prints exactly ONE JSON object on stdout; logs go to stderr. Exit codes: 0 succeeded (also when nothing is stored), 1 failed, 2 rejected input.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# Same reason as research_cli: botocore at DEBUG logs the session token.
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from domain.digest_snapshot import DigestSnapshot  # noqa: E402
from domain.models import TrendMemory  # noqa: E402
from entrypoints import bootstrap  # noqa: E402
from shared import TRENDS_KEY, Config  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_REJECTED = 0, 1, 2


class _ArgumentError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        # argparse would print usage to stderr and exit 2 with nothing on stdout, breaking the
        # one-JSON-object contract; the caller gets a "rejected" result instead.
        raise _ArgumentError(message)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    # No -h: argparse prints help to stdout and exits, which is not the one JSON object callers parse.
    parser = _Parser(description="Read back the OmniSummary digest memory (prints one JSON object)", add_help=False)
    sub = parser.add_subparsers(dest="kind", required=True)
    digest = sub.add_parser("digest", add_help=False)
    digest.add_argument("date", nargs="?", default="", help="YYYY-MM-DD (default: today in the digest timezone)")
    sub.add_parser("trends", add_help=False)
    return parser.parse_args(argv)


def _digest(day: date) -> dict[str, Any]:
    data = bootstrap.memory_store().get_digest(day.isoformat())
    content = DigestSnapshot.from_dict(data).content if data else None
    if content is None or not content.items:
        return {"found": False, "lead": "", "stories": []}
    return {
        "found": True,
        "lead": content.lead,
        "stories": [
            {"title": it.title, "url": it.url, "body": it.body, "implication": it.implication} for it in content.items
        ],
    }


def _trends(config: Config, today: date) -> list[dict[str, Any]]:
    raw = bootstrap.state_store(config).read(TRENDS_KEY)
    memory = TrendMemory.model_validate_json(raw) if raw else TrendMemory()
    half_life = config.pipeline.trend_momentum_half_life_days
    trends = sorted((t for t in memory.trends if t.evidence), key=lambda t: t.momentum(today, half_life), reverse=True)
    return [
        {
            "title": t.title,
            "status": t.status.value,
            "first_seen": t.first_seen or t.evidence[0].date,
            "last_seen": t.last_seen or t.evidence[-1].date,
            "latest": t.evidence[-1].summary,
        }
        for t in trends
    ]


def run(argv: list[str] | None = None) -> tuple[dict[str, Any], int]:
    try:
        args = _parse_args(argv)
    except _ArgumentError as e:
        return {"status": "rejected", "kind": "", "error": str(e)}, EXIT_REJECTED
    result: dict[str, Any] = {"status": "failed", "kind": args.kind, "error": None}
    try:
        config = Config.load()
        today = datetime.now(ZoneInfo(config.aws.timezone)).date()
        if args.kind == "digest":
            try:
                day = date.fromisoformat(args.date) if args.date else today
            except ValueError:
                result.update(status="rejected", error=f'"{args.date}" is not a YYYY-MM-DD date.')
                return result, EXIT_REJECTED
            result.update(date=day.isoformat(), **_digest(day))
        else:
            result["trends"] = _trends(config, today)
    except Exception as e:
        # A store that could not be read is NOT an empty one: say so instead of reporting nothing stored.
        result["error"] = f"{type(e).__name__}: {e}"
        return result, EXIT_FAILED
    result["status"] = "succeeded"
    return result, EXIT_OK


def main() -> None:
    result, code = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
