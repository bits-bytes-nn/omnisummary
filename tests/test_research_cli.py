import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import research_cli

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(argv, agent_response="done"):
    captured = {}

    def fake_create():
        agent = MagicMock(return_value=agent_response)

        def call(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return agent_response

        agent.side_effect = call
        return agent

    with patch.object(sys, "argv", ["research_cli.py", *argv]):
        with patch.object(research_cli, "create_research_agent", side_effect=fake_create):
            research_cli.main()
    return captured


def test_threads_hint_singular():
    cap = _run(["주제", "--channel", "threads"])
    assert "쓰레드에 올려줘" in cap["prompt"]
    assert "에도" not in cap["prompt"]


def test_both_hint_uses_edo():
    cap = _run(["주제", "--channel", "both"])
    assert "쓰레드에도 올려줘" in cap["prompt"]


def test_slack_default_no_hint():
    cap = _run(["주제"])
    assert "쓰레드" not in cap["prompt"]
    assert cap["prompt"] == "주제"


class TestEveryInvocationIsBudgeted:
    """`limits=` appeared exactly ONCE in the repo, in the AgentCore entrypoint. research_cli called
    `agent(prompt)` bare, so the caps AgentConfig documents as the reason 'a loop that never converges
    is unbounded in both cost and wall time' did not apply to the local path — the path used for prompt
    iteration, where non-convergence is most likely and one run can reach the full token budget."""

    # Every module that may invoke the research agent. A new entrypoint belongs here.
    _ENTRYPOINTS = ("research_cli.py", "agent_runtime/app.py")

    def test_the_cli_passes_the_shared_limits(self):
        from agent import research_run_limits

        cap = _run(["주제"])
        assert cap["kwargs"]["limits"] == research_run_limits()

    def test_the_limits_come_from_config(self):
        from agent import research_run_limits
        from shared import Config

        config = Config.load()
        limits = research_run_limits(config)
        assert limits["turns"] == config.agent.research_max_turns
        assert limits["total_tokens"] == config.agent.research_max_total_tokens
        assert limits["output_tokens"] == config.agent.research_max_output_tokens

    def test_every_agent_invocation_carries_limits(self):
        # Source-level (over the AST, so a docstring mentioning a bare call doesn't count): an
        # unbounded invocation is not an error the SDK reports, it just runs until it converges.
        import ast

        for name in self._ENTRYPOINTS:
            tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "agent"
            ]
            assert calls, f"{name} invokes no agent; drop it from _ENTRYPOINTS"
            for call in calls:
                keywords = {kw.arg for kw in call.keywords}
                assert "limits" in keywords, f"{name} line {call.lineno} invokes the agent without limits"

    def test_the_limits_helper_has_one_definition(self):
        # It used to live in the entrypoint that happened to need it, which is why the other one never
        # got it. Beside the agent, there is nothing for a new caller to re-derive. (Source-level:
        # agent_runtime.app imports the bedrock_agentcore runtime, which is not a dev dependency.)
        source = (REPO_ROOT / "agent_runtime" / "app.py").read_text(encoding="utf-8")
        assert "def _run_limits" not in source
        assert "research_run_limits" in source
        for name in self._ENTRYPOINTS:
            assert "Limits(" not in (REPO_ROOT / name).read_text(encoding="utf-8")
