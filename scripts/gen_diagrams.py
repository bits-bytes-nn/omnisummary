#!/usr/bin/env python3
"""Generate the Excalidraw concept diagram (hand-drawn style).

Render with:
  node scripts/render_excalidraw.mjs assets/concept-pipeline.excalidraw assets/concept-pipeline.png

(render_excalidraw.mjs screenshots the font-embedded SVG in a headless browser so the
hand-drawn Virgil font is preserved; the plain `excalidraw-to-png` PNG path rasterizes
via resvg, which drops the embedded font and falls back to sans-serif.)
"""

import json
import random

random.seed(101)


def _rid():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=16))


def _base(**kw):
    e = dict(
        id=_rid(),
        strokeColor="#1e1e1e",
        backgroundColor="transparent",
        fillStyle="solid",
        strokeWidth=2,
        strokeStyle="solid",
        roughness=1,
        opacity=100,
        angle=0,
        seed=random.randint(1, 10**9),
        versionNonce=random.randint(1, 10**9),
        version=1,
        isDeleted=False,
        boundElements=[],
        updated=1,
        link=None,
        locked=False,
        groupIds=[],
        frameId=None,
        roundness=None,
    )
    e.update(kw)
    return e


class Canvas:
    NW, NH = 184, 66

    def __init__(self):
        self.E = []

    def node(self, x, y, text, fill, stroke, fs=13, w=None, h=None):
        w, h = w or self.NW, h or self.NH
        cid, tid = _rid(), _rid()
        lines = text.split("\n")
        self.E.append(
            _base(
                type="rectangle",
                id=cid,
                x=x,
                y=y,
                width=w,
                height=h,
                backgroundColor=fill,
                strokeColor=stroke,
                roundness={"type": 3},
                boundElements=[{"type": "text", "id": tid}],
            )
        )
        self.E.append(
            _base(
                type="text",
                id=tid,
                x=x + 8,
                y=y + h / 2 - (fs * 1.25 * len(lines)) / 2,
                width=w - 16,
                height=fs * 1.25 * len(lines),
                text=text,
                fontSize=fs,
                fontFamily=1,
                textAlign="center",
                verticalAlign="middle",
                strokeColor=stroke,
                containerId=cid,
                originalText=text,
                lineHeight=1.25,
                baseline=fs,
            )
        )
        return (x, y, w, h)

    def zone(self, x, y, w, h, label, stroke, fill):
        cid, tid = _rid(), _rid()
        self.E.append(
            _base(
                type="rectangle",
                id=cid,
                x=x,
                y=y,
                width=w,
                height=h,
                backgroundColor=fill,
                strokeColor=stroke,
                roundness={"type": 3},
                strokeStyle="dashed",
                opacity=15,
                boundElements=[{"type": "text", "id": tid}],
            )
        )
        self.E.append(
            _base(
                type="text",
                id=tid,
                x=x + 14,
                y=y + 10,
                width=w - 28,
                height=20,
                text=label,
                fontSize=15,
                fontFamily=1,
                textAlign="center",
                verticalAlign="top",
                strokeColor=stroke,
                containerId=cid,
                originalText=label,
                lineHeight=1.25,
                baseline=15,
            )
        )

    def arrow(self, x1, y1, x2, y2, color="#1e1e1e", dashed=False, label=""):
        a = _base(
            type="arrow",
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            points=[[0, 0], [x2 - x1, y2 - y1]],
            endArrowhead="arrow",
            strokeColor=color,
            roundness={"type": 2},
        )
        if dashed:
            a["strokeStyle"] = "dashed"
        if label:
            tid = _rid()
            a["boundElements"] = [{"type": "text", "id": tid}]
            self.E.append(a)
            self.E.append(
                _base(
                    type="text",
                    id=tid,
                    x=(x1 + x2) / 2,
                    y=(y1 + y2) / 2 - 8,
                    width=len(label) * 7,
                    height=16,
                    text=label,
                    fontSize=11,
                    fontFamily=1,
                    textAlign="center",
                    verticalAlign="middle",
                    strokeColor=color,
                    containerId=a["id"],
                    originalText=label,
                    lineHeight=1.25,
                    baseline=11,
                )
            )
        else:
            self.E.append(a)

    def title(self, x, y, t, fs=26, color="#1e1e1e"):
        self.E.append(
            _base(
                type="text",
                x=x,
                y=y,
                width=int(len(t) * fs * 0.55),
                height=int(fs * 1.25),
                text=t,
                fontSize=fs,
                fontFamily=1,
                textAlign="left",
                verticalAlign="top",
                strokeColor=color,
                originalText=t,
                lineHeight=1.25,
                baseline=fs,
            )
        )

    def save(self, path):
        scene = {
            "type": "excalidraw",
            "version": 2,
            "source": "omnisummary",
            "elements": self.E,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
            "files": {},
        }
        with open(path, "w") as f:
            json.dump(scene, f)
        return len(self.E)


# palette
ORANGE, ORANGE_S = "#ffd8a8", "#e8590c"  # compute
PURPLE, PURPLE_S = "#d0bfff", "#7048e8"  # bedrock / agentcore
PINK, PINK_S = "#fcc2d7", "#c2255c"  # messaging / ops
BLUE, BLUE_S = "#a5d8ff", "#1971c2"  # data / network
GREEN, GREEN_S = "#b2f2bb", "#2f9e44"  # external / slack
GREY, GREY_S = "#e9ecef", "#868e96"  # internal logic


def build_architecture():
    c = Canvas()
    NW, NH = c.NW, c.NH
    c.title(40, 20, "OmniSummary — AWS Architecture")

    # left main column (x=90) + side column (x=330); rows every 112
    LX, SX = 90, 330
    lcol, scol = LX + NW / 2, SX + NW / 2
    rows = [118, 230, 342, 454, 566]
    c.zone(40, 70, 480, 660, "Scheduled digest path", BLUE_S, BLUE)
    c.node(LX, rows[0], "EventBridge\ndaily cron", PINK, PINK_S)
    c.node(LX, rows[1], "Digest Lambda\n(Docker, VPC)", ORANGE, ORANGE_S)
    c.node(LX, rows[2], "Collectors (async)\nRSS·Reddit·YouTube\nWeb·X (RSSHub)", GREY, GREY_S, 12)
    c.node(LX, rows[3], "Pipeline\nRank (Opus 4.8) →\nDigest (Sonnet 4.6)", GREY, GREY_S, 12)
    c.node(LX, rows[4], "Slack channel\n(daily digest)", GREEN, GREEN_S)
    for a, b in zip(rows, rows[1:], strict=False):
        c.arrow(lcol, a + NH, lcol, b, GREY_S)
    # side-cars (each at its own row -> no arrow overlap)
    c.node(SX, rows[0], "SNS alerts\nemail on FAIL", PINK, PINK_S)
    c.arrow(LX + NW, rows[0] + NH / 2, SX, rows[0] + NH / 2, PINK_S, dashed=True)
    c.node(SX, rows[1], "CloudWatch\nlogs + alarms", PINK, PINK_S)
    c.arrow(LX + NW, rows[1] + NH / 2, SX, rows[1] + NH / 2, PINK_S, dashed=True)
    c.node(SX, rows[2], "ECS Fargate\nRSSHub", ORANGE, ORANGE_S)
    c.arrow(LX + NW, rows[2] + NH / 2, SX, rows[2] + NH / 2, GREY_S, dashed=True)
    c.node(SX, rows[3], "Bedrock\nOpus / Sonnet", PURPLE, PURPLE_S)
    c.arrow(LX + NW, rows[3] + NH / 2, SX, rows[3] + NH / 2, PURPLE_S)
    c.node(SX, rows[4], "AgentCore Memory\n(snapshot + trends)", PURPLE, PURPLE_S, 12)
    c.arrow(scol, rows[3] + NH, scol, rows[4], PURPLE_S)  # bedrock area -> memory

    # right main column (x=620) + side column (x=850)
    RX, RSX = 620, 850
    rcol = RX + NW / 2
    c.zone(560, 70, 470, 660, "Interactive follow-up path", GREEN_S, GREEN)
    c.node(RX, rows[0], "Slack user\n@mention", GREEN, GREEN_S)
    c.node(RX, rows[1], "AWS WAF\nmanaged rules", PINK, PINK_S)
    c.node(RX, rows[2], "API Gateway\n(throttled)", PURPLE, PURPLE_S)
    c.node(RX, rows[3], "Slack-events Lambda\nverify + dedup", ORANGE, ORANGE_S, 12)
    c.node(RX, rows[4], "AgentCore Runtime\nStrands agent", PURPLE, PURPLE_S, 12)
    colors = [GREEN_S, PINK_S, PURPLE_S, ORANGE_S]
    for (a, b), col in zip(zip(rows, rows[1:], strict=False), colors, strict=False):
        c.arrow(rcol, a + NH, rcol, b, col)
    c.node(RSX, rows[3], "DynamoDB\ndedup (TTL)", BLUE, BLUE_S)
    c.arrow(RX + NW, rows[3] + NH / 2, RSX, rows[3] + NH / 2, BLUE_S, dashed=True)
    c.node(RSX, rows[4], "Tools → OpenAI\ngpt-image-2 → Slack", GREY, GREY_S, 12)
    c.arrow(RX + NW, rows[4] + NH / 2, RSX, rows[4] + NH / 2, GREY_S)
    # runtime reads memory: clean horizontal across the gap at the memory row
    c.arrow(RX, rows[4] + NH / 2, SX + NW, rows[4] + NH / 2, PURPLE_S, dashed=True, label="reads state")
    return c


if __name__ == "__main__":
    n = build_architecture().save("assets/architecture.excalidraw")
    print(f"architecture.excalidraw: {n} elements")
