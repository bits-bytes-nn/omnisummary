"""Deep research on one AI/ML topic, for OpenClaw and other local callers.

Prints exactly ONE JSON object on stdout; every log line goes to stderr. Exit codes: 0 succeeded,
1 failed, 2 rejected input. Publishes to Threads only with `--publish threads`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# Third-party loggers stay at WARNING whatever LOG_LEVEL says: at DEBUG, botocore logs request
# headers — the session token included — and the caller may keep this stderr. LOG_LEVEL still drives
# the app's own logger (shared.logger), which has its own stderr handler.
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from agent.research_agent import create_research_agent, research_run_limits  # noqa: E402
from agent.research_tools import DeliveryContext, request_context  # noqa: E402
from output.delivery import MARKDOWN, THREADS  # noqa: E402
from shared import Config, extract_urls, logger, normalize_citation_url  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports"

EXIT_OK, EXIT_FAILED, EXIT_REJECTED = 0, 1, 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniSummary deep research (prints one JSON object)")
    parser.add_argument("topic", help="The AI/ML topic or question to research")
    parser.add_argument(
        "--publish",
        choices=[THREADS],
        default="",
        help="Also publish a thread to the public Threads account (never happens without this flag)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where the Markdown report is saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def _usage(result: Any) -> dict[str, Any]:
    """Token usage the Strands run accumulated, in the same terms the pipeline logs (`LLM usage`)."""
    metrics = getattr(result, "metrics", None)
    usage = dict(getattr(metrics, "accumulated_usage", {}) or {})
    tool_metrics = getattr(metrics, "tool_metrics", {}) or {}
    return {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "cache_read_input_tokens": usage.get("cacheReadInputTokens", 0),
        "cache_write_input_tokens": usage.get("cacheWriteInputTokens", 0),
        "cycles": int(getattr(metrics, "cycle_count", 0) or 0),
        "tool_calls": {name: getattr(m, "call_count", 0) for name, m in tool_metrics.items()},
    }


def _title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()
    return ""


def _save(markdown: str, topic: str, output_dir: Path, timezone: str) -> Path:
    stamp = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d-%H%M%S")
    path = output_dir / f"{stamp}-{hashlib.sha256(topic.encode('utf-8')).hexdigest()[:8]}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _threads_result(delivery: DeliveryContext) -> dict[str, Any] | None:
    if not delivery.publish_threads:
        return None
    stats = delivery.threads_stats
    return {
        "published": THREADS in delivery.delivered_channels,
        "partial": THREADS in delivery.partial_channels,
        "posts_delivered": stats.delivered,
        "posts_rendered": stats.rendered,
        "dropped": stats.dropped,
        "trimmed": stats.trimmed,
    }


def run(argv: list[str] | None = None) -> tuple[dict[str, Any], int]:
    args = _parse_args(argv)
    topic = args.topic.strip()
    result: dict[str, Any] = {
        "status": "failed",
        "topic": topic,
        "title": "",
        "markdown_path": None,
        "markdown": "",
        "sources": [],
        "images": [],
        "threads": None,
        "model": "",
        "stop_reason": "",
        "usage": None,
        "warnings": [],
        "error": None,
    }
    if not topic:
        result.update(status="rejected", error="The topic is empty.")
        return result, EXIT_REJECTED

    try:
        config = Config.load()
    except Exception as e:
        # Still one JSON object on stdout: the caller (the OpenClaw skill) parses it either way.
        result["error"] = f"Config could not be loaded: {type(e).__name__}: {e}"
        return result, EXIT_FAILED
    result["model"] = config.agent.model_id.value
    delivery = DeliveryContext(publish_threads=args.publish == THREADS)
    allowed = "ALLOWED" if delivery.publish_threads else "NOT allowed"
    prompt = f"{topic}\n\n[DELIVERY] Threads publishing: {allowed}."

    warnings: list[str] = result["warnings"]
    with request_context(delivery):
        try:
            agent = create_research_agent()
            run_result = agent(prompt, limits=research_run_limits(config))
        except Exception as e:
            logger.error("Research run failed: %s", e, exc_info=True)
            result["error"] = f"{type(e).__name__}: {e}"
            return result, EXIT_FAILED

        result["usage"] = _usage(run_result)
        result["stop_reason"] = str(getattr(run_result, "stop_reason", "") or "")
        markdown = delivery.report
        if MARKDOWN not in delivery.delivered_channels:
            # The agent answered without submitting through deliver_report, so its citations were
            # never checked against what the tools returned. Use the answer, and say so.
            markdown = str(run_result).strip()
            warnings.append("The agent did not submit the report through deliver_report; its citations are unchecked.")
            unverified = [u for u in extract_urls(markdown) if normalize_citation_url(u) not in delivery.seen_urls]
            if unverified:
                warnings.append(f"URLs no tool returned: {', '.join(dict.fromkeys(unverified))}")

    if result["stop_reason"].startswith("limit_"):
        warnings.append(f"The run stopped at its budget cap ({result['stop_reason']}); the report may be partial.")
    result["threads"] = _threads_result(delivery)
    result["images"] = [
        {"image_url": a.image_url, "source_url": a.source_url, "content_type": a.content_type}
        for a in delivery.staged_images
    ]
    if not markdown:
        result["error"] = "The agent produced no report."
        return result, EXIT_FAILED

    result.update(
        title=_title(markdown),
        markdown=markdown,
        markdown_path=str(_save(markdown, topic, args.output_dir, config.aws.timezone)),
        sources=list(dict.fromkeys(extract_urls(markdown))),
    )
    threads = result["threads"]
    if threads is not None and not threads["published"]:
        result["error"] = "Threads publishing was requested but did not complete."
        return result, EXIT_FAILED
    if threads is not None and threads["posts_delivered"] < threads["posts_rendered"]:
        warnings.append("Some Threads posts did not land.")
    if threads is not None and (threads["dropped"] or threads["trimmed"]):
        warnings.append(f"Threads rendering dropped {threads['dropped']} and trimmed {threads['trimmed']} post(s).")
    result["status"] = "succeeded"
    return result, EXIT_OK


def main() -> None:
    result, code = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
