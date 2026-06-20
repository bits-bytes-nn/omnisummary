import sys
from unittest.mock import MagicMock, patch

import research_cli


def _run(argv, agent_response="done"):
    captured = {}

    def fake_create():
        agent = MagicMock(return_value=agent_response)

        def call(prompt):
            captured["prompt"] = prompt
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
