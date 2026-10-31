#!/usr/bin/env python3
"""Blind A/B evaluation of the daily comic and the research thread. Nothing is published.

    eval_harness.py visual <label> <date> [<date> ...]   # editor plan + art-director brief per stored day
    eval_harness.py research <label> <topic> [...]       # research run with a Threads PREVIEW (posts nothing)
    eval_harness.py judge <kind> <label_a> <label_b>     # Opus judges each case blind, A/B order shuffled

Runs land in evals/<kind>/<label>/<case>.json. Generate the baseline BEFORE changing a prompt and the
candidate after, on the same cases; keep a few cases aside and judge them once, at the end. Cost: a
visual case is two Sonnet calls (cents); a research case is a full research run (about $1.5); a judged
pair is one Opus call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from domain.digest_snapshot import DigestSnapshot  # noqa: E402
from entrypoints import bootstrap  # noqa: E402
from shared import Config, logger, parse_json_from_llm_output  # noqa: E402
from shared.constants import LanguageModelId  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "evals"
JUDGE_MODEL = LanguageModelId.CLAUDE_V5_5_OPUS

RUBRICS = {
    "visual": [
        (
            "understanding",
            "A reader outside the field comes away understanding what happened and why it matters, the hard part included.",
        ),
        ("wit", "It is genuinely funny or insightful, and the joke comes from the story itself."),
        (
            "freshness",
            "It avoids the formula: not the same beats, gag shape, props or framing as the recent comics listed.",
        ),
        ("cast", "The real people and companies of the story carry it, recognisably and fairly."),
        ("fidelity", "Nothing contradicts or overstates the source; no invented facts."),
        ("drawability", "An image model can render it legibly: clear frames, short Korean text, no clutter."),
    ],
    "research": [
        ("clarity", "A reader outside the field can follow it: terms glossed, background given, plain Korean."),
        ("depth", "Detailed and specific: mechanisms, numbers that matter, concrete examples."),
        (
            "perspectives",
            "It looks at the topic from several angles (technical, business, policy, community) where they apply.",
        ),
        ("insight", "It says something a reader could not get from the headlines; a clear take, argued."),
        ("connections", "It ties the topic to other trends or earlier events, where that genuinely illuminates."),
        ("sourcing", "Claims are cited and match their sources; nothing fabricated."),
    ],
}

_JUDGE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict, impartial editor comparing two outputs made from the same input. Judge only by "
            "the rubric. Length is not quality. Do not prefer an output for its position.",
        ),
        (
            "human",
            "<input>\n{context}\n</input>\n\n<output_1>\n{first}\n</output_1>\n\n<output_2>\n{second}\n</output_2>\n\n"
            "<rubric>\n{rubric}\n</rubric>\n\nScore each output 1-5 on every criterion, then name the better one "
            "overall. Answer ONLY with JSON: "
            '{{"scores_1": {{"<criterion>": n, ...}}, "scores_2": {{...}}, "winner": 1 or 2 or 0 (tie), '
            '"reason": "<two sentences>"}}',
        ),
    ]
)


def _case_dir(kind: str, label: str) -> Path:
    path = ROOT / kind / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(text: str) -> str:
    return re.sub(r"\W+", "-", text).strip("-")[:40] + "-" + hashlib.sha256(text.encode()).hexdigest()[:6]


async def _visual(label: str, dates: list[str]) -> None:
    config = Config.load()
    maker = bootstrap.daily_visual_maker(config)
    log = maker.comic_log.entries() if maker.comic_log else []
    for day in dates:
        data = bootstrap.memory_store().get_digest(day)
        if not data:
            logger.warning("No snapshot for %s; skipped", day)
            continue
        snapshot = DigestSnapshot.from_dict(data)
        recent = [e for e in log if str(e.get("date", "")) < day][-config.pipeline.recent_leads_window :]
        plan, instruction, source = await maker._compose(
            snapshot.ranked_items, snapshot.content, date.fromisoformat(day), recent_comics=recent
        )
        brief = await maker.generator.brief(instruction, source) if instruction else None
        out = {
            "case": day,
            "context": f"HEADLINE ARTICLE:\n{source[:6000]}\n\nRECENT COMICS:\n{json.dumps(recent, ensure_ascii=False)}",
            "artifact": json.dumps(
                {"plan": plan, "brief": brief.model_dump() if brief else None}, ensure_ascii=False, indent=1
            ),
        }
        (_case_dir("visual", label) / f"{day}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        logger.info("visual %s %s: %s", label, day, brief.title if brief else "(skipped by the editor)")


def _research(label: str, topics: list[str]) -> None:
    for topic in topics:
        with_tmp = _case_dir("research", label) / "reports"
        proc = subprocess.run(
            [sys.executable, "research_cli.py", topic, "--publish", "threads-preview", "--output-dir", str(with_tmp)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        result = json.loads(proc.stdout)
        posts = (result.get("threads") or {}).get("posts") or []
        out = {
            "case": topic,
            "context": f"TOPIC: {topic}",
            "artifact": "THREADS THREAD:\n"
            + "\n\n---\n\n".join(posts)
            + "\n\nFULL REPORT:\n"
            + result.get("markdown", ""),
            "status": result.get("status"),
            "usage": result.get("usage"),
        }
        (_case_dir("research", label) / f"{_slug(topic)}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1)
        )
        logger.info("research %s '%s': %s, %d posts", label, topic, result.get("status"), len(posts))


async def _judge(kind: str, label_a: str, label_b: str) -> None:
    config = Config.load()
    llm = bootstrap.language_models(config).get_model(JUDGE_MODEL, stage="eval-judge")
    chain = _JUDGE | llm | StrOutputParser()
    rubric = "\n".join(f"- {name}: {text}" for name, text in RUBRICS[kind])
    rows = []
    for path_a in sorted(_case_dir(kind, label_a).glob("*.json")):
        path_b = _case_dir(kind, label_b) / path_a.name
        if not path_b.exists():
            continue
        a, b = json.loads(path_a.read_text()), json.loads(path_b.read_text())
        # Deterministic shuffle per case, so a rerun judges the same order.
        a_first = int(hashlib.sha256(path_a.name.encode()).hexdigest(), 16) % 2 == 0
        first, second = (a, b) if a_first else (b, a)
        raw = await chain.ainvoke(
            {"context": a["context"], "first": first["artifact"], "second": second["artifact"], "rubric": rubric}
        )
        verdict = parse_json_from_llm_output(raw)
        scores_a, scores_b = (
            (verdict["scores_1"], verdict["scores_2"]) if a_first else (verdict["scores_2"], verdict["scores_1"])
        )
        winner = {0: "tie", 1: label_a if a_first else label_b, 2: label_b if a_first else label_a}[
            int(verdict["winner"])
        ]
        rows.append(
            {"case": a["case"], "winner": winner, label_a: scores_a, label_b: scores_b, "reason": verdict["reason"]}
        )
        logger.info("%s: %s", a["case"], winner)
    out = ROOT / kind / f"judge-{label_a}-vs-{label_b}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    wins = {label: sum(r["winner"] == label for r in rows) for label in (label_a, label_b, "tie")}
    means = {
        label: {
            name: round(sum(r[label].get(name, 0) for r in rows) / max(1, len(rows)), 2) for name, _ in RUBRICS[kind]
        }
        for label in (label_a, label_b)
    }
    print(
        json.dumps({"cases": len(rows), "wins": wins, "means": means, "file": str(out)}, ensure_ascii=False, indent=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    visual = sub.add_parser("visual")
    visual.add_argument("label")
    visual.add_argument("dates", nargs="+")
    research = sub.add_parser("research")
    research.add_argument("label")
    research.add_argument("topics", nargs="+")
    judge = sub.add_parser("judge")
    judge.add_argument("kind", choices=sorted(RUBRICS))
    judge.add_argument("label_a")
    judge.add_argument("label_b")
    args = parser.parse_args()
    if args.cmd == "visual":
        asyncio.run(_visual(args.label, args.dates))
    elif args.cmd == "research":
        _research(args.label, args.topics)
    else:
        asyncio.run(_judge(args.kind, args.label_a, args.label_b))


if __name__ == "__main__":
    main()
