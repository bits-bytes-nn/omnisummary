from __future__ import annotations

from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from slack_sdk.web import WebClient

from agent import create_research_agent, research_run_limits
from agent.research_tools import DeliveryContext, request_context
from output.renderers import render_agent_blocks
from shared import emit_emf, logger, resolve_secret, sanitize_slack_mrkdwn, set_correlation_id

app = BedrockAgentCoreApp()

LIMIT_NOTICE = "\n\n_This report was cut short: the research run reached its per-request budget cap._"


def _emit_agent_error_metric() -> None:
    """Emit a CloudWatch EMF error metric so a systemic agent break is alarmable — the runtime
    catches its own exceptions and replies with text, so nothing else would record a failure."""
    emit_emf({"AgentErrors": 1})


def _emit_agent_limit_metric(stop_reason: str) -> None:
    """Emit a CloudWatch EMF counter when the loop was stopped by a budget cap. A capped run still
    returns text, so without this a topic that systematically fails to converge (and burns the whole
    budget every time) is indistinguishable from a normal day. The stop reason rides along as a
    non-metric property: it is context an operator reads, not something to alarm on."""
    emit_emf({"AgentLimitStops": 1}, {"StopReason": stop_reason})


def _emit_agent_run_metrics(usage: dict[str, Any], cycles: int, tool_calls: int) -> None:
    """Emit the run's token usage as CloudWatch EMF, next to the error metric. EMF is just a log
    line, so this needs no new AWS resource — and it is the only way the agent's spend (the most
    expensive component) is attributable at all, since a research turn re-sends the whole
    conversation and the runtime is billed per token like every pipeline stage."""
    emit_emf(
        {
            "AgentInputTokens": int(usage.get("inputTokens", 0) or 0),
            "AgentOutputTokens": int(usage.get("outputTokens", 0) or 0),
            "AgentCycles": cycles,
            "AgentToolCalls": tool_calls,
        }
    )


def _log_agent_run(result: Any) -> None:
    """Log what the run cost in the SAME shape every pipeline stage logs (`LLM usage stage=...`),
    plus the cycle count and per-tool call counts.

    The entrypoint used to do `str(agent(prompt))` and throw the AgentResult — and with it the
    accumulated usage — away, so the single most expensive component in the system was the one
    stage whose spend nothing recorded. Telemetry must never break a completed run, hence the
    blanket except."""
    try:
        metrics = getattr(result, "metrics", None)
        if metrics is None:
            return
        usage = dict(getattr(metrics, "accumulated_usage", {}) or {})
        tool_metrics = getattr(metrics, "tool_metrics", {}) or {}
        tool_calls = {name: getattr(m, "call_count", 0) for name, m in tool_metrics.items()}
        cycles = int(getattr(metrics, "cycle_count", 0) or 0)
        logger.info(
            "LLM usage stage=research input=%s output=%s cache_read=%s cache_write=%s cycles=%d tools=%s",
            usage.get("inputTokens"),
            usage.get("outputTokens"),
            usage.get("cacheReadInputTokens"),
            usage.get("cacheWriteInputTokens"),
            cycles,
            tool_calls or "{}",
        )
        _emit_agent_run_metrics(usage, cycles, sum(tool_calls.values()))
    except Exception:  # pragma: no cover - telemetry must never break a completed run
        logger.debug("Could not read agent run metrics", exc_info=True)


def _send_slack_message(channel: str, text: str, thread_ts: str = "") -> None:
    """Fallback delivery: post the agent's final text to Slack when the agent finished without
    calling deliver_report. The happy path delivers through the deliver_report tool instead."""
    # resolve_secret is THE env-then-SSM ladder (it also treats the put_secrets placeholder as
    # unset and resolves the region from the environment/config rather than a baked-in default).
    # This module already imports from shared, so a 15-line copy of that ladder was pure duplication.
    bot_token = resolve_secret("SLACK_BOT_TOKEN", "slack-bot-token")
    if not bot_token:
        return
    client = WebClient(token=bot_token)
    for blocks in render_agent_blocks(text):
        kwargs: dict[str, Any] = {"channel": channel, "blocks": blocks, "text": text[:200]}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        client.chat_postMessage(**kwargs)


@app.entrypoint
def invoke(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt", "")
    channel_id = payload.get("channel_id", "")
    thread_ts = payload.get("thread_ts", "")

    set_correlation_id(payload.get("correlation_id") or None)
    logger.info("AgentCore invoked: prompt='%s', channel='%s'", prompt[:100], channel_id)

    # The channels this invocation may publish to, as the caller stated them. Empty means
    # unconstrained; the Slack ingress sends both, because whether a Slack request also wants Threads
    # is in the requester's own words and only the model can read those — but deliver_report now
    # enforces an explicit allow-list rather than trusting a phrase match with no floor under it.
    delivery = DeliveryContext(
        channel_id=channel_id,
        thread_ts=thread_ts,
        requested_channels={str(c).lower().strip() for c in (payload.get("requested_channels") or ()) if c},
    )

    # contextvar-scoped per-invocation delivery: a warm container handling concurrent
    # invocations can't leak one request's channel into another.
    notice = ""
    with request_context(delivery):
        try:
            # CONSTRUCTED INSIDE the guard: it resolves credentials, an inference profile and the
            # model registry, so it can raise. Built outside, that raise escaped the entrypoint —
            # the user got the Slack ack and then permanent silence, with no AgentErrors metric and
            # no fallback post, which is the one failure shape this whole block exists to prevent.
            agent = create_research_agent()
            result = agent(prompt, limits=research_run_limits())
            _log_agent_run(result)
            response = sanitize_slack_mrkdwn(str(result))
            stop_reason = str(getattr(result, "stop_reason", "") or "")
            if stop_reason.startswith("limit_"):
                # The loop stops at a turn boundary, so the last message is whatever the agent had
                # written by then. Say so rather than passing a partial report off as finished.
                logger.warning("Research run stopped by a budget cap (%s); the report is partial", stop_reason)
                _emit_agent_limit_metric(stop_reason)
                notice = LIMIT_NOTICE
        except Exception as e:
            logger.error("Agent execution failed: %s", e, exc_info=True)
            _emit_agent_error_metric()
            # Don't leak the raw exception (model IDs, ARNs, backend error bodies) into Slack.
            response = "Sorry — the research run failed. Please try again; details are in the logs."

        # Fallback ONLY when the agent delivered to NO channel at all (it never called
        # deliver_report, or every delivery failed) — so the user always gets something. Do NOT
        # fall back to Slack just because Slack wasn't a target: a Threads-only request that
        # succeeded on Threads must not also dump the (Threads-formatted) report into Slack.
        # Prefer the actual report the agent produced over its terminal one-line confirmation.
        if channel_id and not delivery.delivered_channels:
            fallback_text = (delivery.last_report or response) + notice
            # This is the last-resort "always give the user something" path — a raise here (rate
            # limit, bad channel) must not turn the whole invocation into a hard error.
            try:
                _send_slack_message(channel_id, sanitize_slack_mrkdwn(fallback_text), thread_ts)
            except Exception as e:
                # Nothing reached the user on ANY channel: the report is lost. Emit the error metric
                # so the alarm fires — the entrypoint still returns text, so a failure here is
                # otherwise invisible.
                logger.error("Fallback Slack post failed: %s", e, exc_info=True)
                _emit_agent_error_metric()

    return response + notice


if __name__ == "__main__":
    app.run()
