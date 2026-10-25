import json

import pytest

from agent_core.report import build_report, normalize_violations, validate_report, write_report


def finding(**overrides):
    base = {
        "rule_id": "CI-1", "category": "github_actions_ci", "severity": "high", "file": "ci.yml", "line": 2,
        "description": "Tests are skipped", "evidence": "continue-on-error on line 2", "recommendation": "Remove it",
    }
    base.update(overrides)
    return base


@pytest.fixture
def repo_root(tmp_path):
    (tmp_path / "ci.yml").write_text("one\ntwo\nthree\n")
    return tmp_path


def test_valid_finding_is_kept_and_renumbered(repo_root):
    result = normalize_violations([finding(rule_id="whatever")], repo_root)
    assert result[0]["rule_id"] == "CI-001"
    assert list(result[0]) == [
        "rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation",
    ]


def test_duplicates_are_merged(repo_root):
    result = normalize_violations([finding(), finding(description="same thing reworded")], repo_root)
    assert len(result) == 1


def test_invalid_severity_becomes_medium(repo_root):
    assert normalize_violations([finding(severity="catastrophic")], repo_root)[0]["severity"] == "medium"


def test_unknown_file_and_bad_line_become_null(repo_root):
    assert normalize_violations([finding(file="ghost.py")], repo_root)[0]["file"] is None
    result = normalize_violations([finding(line=999)], repo_root)[0]
    assert result["file"] == "ci.yml" and result["line"] is None


def test_paths_cannot_escape_the_repository(repo_root):
    assert normalize_violations([finding(file="../../etc/passwd")], repo_root)[0]["file"] is None


def test_finding_without_evidence_is_dropped(repo_root):
    assert normalize_violations([finding(evidence="")], repo_root) == []


def test_garbage_entries_are_ignored(repo_root):
    assert normalize_violations(["junk", None, 5, {}], repo_root) == []
    assert normalize_violations("not a list", repo_root) == []


def test_secrets_never_reach_the_report(repo_root):
    result = normalize_violations([finding(evidence='password = "hunter2hunter2" in config')], repo_root)
    assert "hunter2" not in json.dumps(result)


def test_missing_recommendation_gets_a_default(repo_root):
    assert normalize_violations([finding(recommendation="")], repo_root)[0]["recommendation"]


def test_sorted_by_severity(repo_root):
    result = normalize_violations([finding(severity="low", line=1), finding(severity="critical", line=2)], repo_root)
    assert [v["severity"] for v in result] == ["critical", "low"]


def test_empty_report_is_valid():
    report = build_report("https://github.com/a/b", [])
    validate_report(report)
    assert report["violations"] == []


@pytest.mark.parametrize("report", [
    {"repository": "u"},
    {"repository": "u", "violations": [{"rule_id": "A"}]},
    {"repository": "u", "violations": [finding(severity="bad")]},
    {"repository": "u", "violations": [finding(line="3")]},
])
def test_validate_report_rejects_bad_reports(report):
    with pytest.raises(ValueError):
        validate_report(report)


def test_write_report_creates_valid_json(tmp_path, repo_root):
    target = tmp_path / "out" / "report.json"
    write_report(target, build_report("https://github.com/a/b", [finding()], repo_root))
    data = json.loads(target.read_text())
    assert data["repository"] == "https://github.com/a/b"
    assert data["violations"][0]["file"] == "ci.yml"
    assert not list(target.parent.glob("*.tmp"))
