import json
import os
import shutil

import pytest

import agent
from agent_core.llm import LLMError, LLMResponse
from agent_core.loop import AgentError, AuditAgent
from helpers import FakeLLM, tool_reply
from tools import RepoTools

VIOLATION = {
    "rule_id": "SEC-001", "category": "security", "severity": "critical", "file": "config.py", "line": 1,
    "description": "Hardcoded API credential", "evidence": "Credential assigned in source code",
    "recommendation": "Use an environment variable",
}
REPORT = tool_reply(("submit_report", {"violations": [VIOLATION]}))
INSPECTION = [
    tool_reply(("list_files", {})),
    tool_reply(("read_file", {"path": "config.py"}), ("search_code", {"pattern": "api_key"})),
]


def make_agent(llm, repo, **kwargs):
    return AuditAgent(llm, RepoTools(repo), "https://github.com/a/b", min_interval=0, log=lambda text: None, **kwargs)


def test_agent_investigates_then_reports(sample_repo):
    audit = make_agent(FakeLLM(INSPECTION + [REPORT]), sample_repo)
    assert audit.run() == [VIOLATION]
    assert [step["tool"] for step in audit.trace] == ["list_files", "read_file", "search_code", "submit_report"]


def test_tool_results_are_fed_back_to_the_model(sample_repo):
    llm = FakeLLM(INSPECTION + [REPORT])
    make_agent(llm, sample_repo).run()
    last_messages = llm.calls[-1]["messages"]
    tool_outputs = [m["content"] for m in last_messages if m["role"] == "tool"]
    assert any("[REDACTED]" in output for output in tool_outputs)
    assert not any("abcd1234efgh5678" in output for output in tool_outputs)


def test_report_before_enough_inspection_is_rejected(sample_repo):
    llm = FakeLLM([tool_reply(("list_files", {})), REPORT, tool_reply(("git_log", {})),
                   tool_reply(("read_file", {"path": "README.md"})), tool_reply(("submit_report", {"violations": []}))])
    audit = make_agent(llm, sample_repo)
    assert audit.run() == []
    rejection = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "tool"][-1]
    assert rejection.startswith("Not accepted")


def test_model_that_never_uses_tools_fails_cleanly(sample_repo):
    llm = FakeLLM([LLMResponse("thinking"), LLMResponse("still thinking"), LLMResponse("nothing")])
    with pytest.raises(AgentError):
        make_agent(llm, sample_repo).run()


def test_step_limit_forces_a_final_report(sample_repo):
    llm = FakeLLM([tool_reply(("list_files", {}))] * 3 + [REPORT])
    assert make_agent(llm, sample_repo, max_steps=3).run() == [VIOLATION]
    assert llm.calls[-1]["tools"] == ["submit_report"]


def test_time_budget_forces_a_final_report(sample_repo):
    llm = FakeLLM([REPORT])
    assert make_agent(llm, sample_repo, max_seconds=-1).run() == [VIOLATION]
    assert llm.calls[0]["tools"] == ["submit_report"]


def test_unknown_tools_do_not_crash_the_loop(sample_repo):
    llm = FakeLLM([tool_reply(("list_files", {}), ("nonsense", {})), INSPECTION[1], REPORT])
    assert make_agent(llm, sample_repo).run() == [VIOLATION]


def test_old_tool_outputs_are_shortened():
    messages = [{"role": "user", "content": "x"}]
    for number in range(10):
        messages.append({"role": "tool", "tool_call_id": str(number), "name": "read_file", "content": "A" * 3000})
    lengths = [len(m["content"]) for m in AuditAgent._compact(messages) if m["role"] == "tool"]
    assert lengths[:6] == [lengths[0]] * 6 and lengths[0] < 600
    assert lengths[6:] == [3000] * 4
    assert len(messages[1]["content"]) == 3000


def test_trace_never_contains_secrets(sample_repo):
    audit = make_agent(FakeLLM([]), sample_repo)
    audit._record(1, "read_file", {"path": "config.py"}, 'API_KEY = "supersecretvalue123"')
    assert "supersecretvalue123" not in json.dumps(audit.trace)


@pytest.fixture
def cli(monkeypatch, sample_repo, tmp_path):
    """Run agent.main() with a fake clone and a fake LLM (no network, no API key)."""
    clones = []

    def fake_clone(url):
        destination = str(tmp_path / f"clone{len(clones)}")
        shutil.copytree(sample_repo, destination)
        clones.append(destination)
        return destination

    monkeypatch.setattr(agent, "clone_repo", fake_clone)
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0")

    def run(replies, *extra, repo="https://github.com/a/b.git"):
        monkeypatch.setattr(agent, "create_client", lambda: FakeLLM(replies))
        return agent.main(["--repo", repo, "--output", str(tmp_path / "out" / "report.json"), *extra])

    run.clones = clones
    run.report = tmp_path / "out" / "report.json"
    return run


def test_cli_writes_a_valid_report_and_trace(cli):
    assert cli(INSPECTION + [REPORT]) == 0
    report = json.loads(cli.report.read_text())
    assert report["repository"] == "https://github.com/a/b"
    assert report["violations"][0]["rule_id"] == "SEC-001"
    trace = json.loads((cli.report.parent / "report.trace.json").read_text())
    assert [call["tool"] for call in trace["tool_calls"]][0] == "list_files"
    assert trace["model"] == "fake/scripted"


def test_cli_can_skip_the_trace(cli):
    assert cli(INSPECTION + [REPORT], "--no-trace") == 0
    assert not (cli.report.parent / "report.trace.json").exists()


def test_cli_removes_the_clone_afterwards(cli):
    cli(INSPECTION + [REPORT])
    assert not os.path.exists(cli.clones[-1])


def test_cli_rejects_invalid_input(cli, capsys):
    assert cli([], repo="not-a-url") == 2
    assert "Invalid repository URL" in capsys.readouterr().err
    assert cli([], "--max-steps", "0") == 2
    assert not cli.report.exists()


def test_cli_exits_nonzero_without_a_report_when_the_llm_fails(cli):
    assert cli([LLMError("quota reached")]) == 1
    assert not cli.report.exists()
    assert not os.path.exists(cli.clones[-1])


def test_cli_exits_nonzero_when_clone_fails(cli, monkeypatch):
    def broken_clone(url):
        raise RuntimeError("Could not clone: repository not found")

    monkeypatch.setattr(agent, "clone_repo", broken_clone)
    assert cli([]) == 1
    assert not cli.report.exists()
