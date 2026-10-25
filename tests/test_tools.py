import subprocess

import pytest

import tools
from tools import RepoTools, clone_repo, redact_secrets, validate_repo_url


def test_redact_masks_credentials():
    text = 'API_KEY = "abcd1234efgh5678"\ntoken: ghp_' + "a" * 36 + "\nkey=AKIA" + "B" * 16
    cleaned = redact_secrets(text)
    assert "abcd1234efgh5678" not in cleaned
    assert "ghp_" not in cleaned
    assert "AKIA" not in cleaned
    assert cleaned.startswith('API_KEY = "[REDACTED]"')


def test_redact_private_key_keeps_line_count():
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJB\n-----END RSA PRIVATE KEY-----\nafter"
    cleaned = redact_secrets(key)
    assert "MIIBOg" not in cleaned
    assert cleaned.count("\n") == key.count("\n")


def test_redact_leaves_workflow_secret_references_alone():
    line = "token: ${{ secrets.GITHUB_TOKEN }}"
    assert redact_secrets(line) == line


@pytest.mark.parametrize("url", [
    "https://github.com/pallets/click",
    "https://github.com/pallets/click.git",
    "https://github.com/pallets/click/",
])
def test_validate_repo_url_accepts_github_urls(url):
    assert validate_repo_url(url) == "https://github.com/pallets/click"


@pytest.mark.parametrize("url", [
    "", "not-a-url", "http://github.com/a/b", "https://gitlab.com/a/b",
    "https://github.com/a/b; rm -rf /", "https://github.com/a",
])
def test_validate_repo_url_rejects_everything_else(url):
    with pytest.raises(ValueError):
        validate_repo_url(url)


def test_clone_repo_rejects_bad_url_before_running_git():
    with pytest.raises(ValueError):
        clone_repo("nope")


def test_list_files(sample_repo):
    output = RepoTools(sample_repo).list_files()
    assert output.startswith("4 files")
    assert "config.py" in output
    assert ".github/workflows/ci.yml" in output


def test_list_files_directory_filter(sample_repo):
    output = RepoTools(sample_repo).list_files(directory=".github")
    assert "ci.yml" in output
    assert "config.py" not in output


def test_read_file_numbers_lines_and_masks_secrets(sample_repo):
    output = RepoTools(sample_repo).read_file("config.py")
    assert '   1 | API_KEY = "[REDACTED]"' in output
    assert "abcd1234efgh5678" not in output
    assert "   2 | DEBUG = True" in RepoTools(sample_repo).read_file("config.py", start_line=2)


def test_read_file_blocks_path_traversal(sample_repo):
    assert RepoTools(sample_repo).read_file("../../etc/passwd").startswith("ERROR")


def test_read_file_handles_missing_directory_and_binary(sample_repo):
    repo_tools = RepoTools(sample_repo)
    assert "file not found" in repo_tools.read_file("nope.py")
    assert "directory" in repo_tools.read_file(".github")
    assert "binary" in repo_tools.read_file("logo.bin")


def test_search_code_finds_matches_and_masks_values(sample_repo):
    output = RepoTools(sample_repo).search_code("api_key")
    assert output.startswith("config.py:1:")
    assert "abcd1234efgh5678" not in output


def test_search_code_can_be_limited_to_a_path(sample_repo):
    repo_tools = RepoTools(sample_repo)
    assert "ci.yml:2" in repo_tools.search_code("push", path=".github")
    assert "No matches" in repo_tools.search_code("push", path="docs")


def test_search_code_invalid_regex_and_no_match(sample_repo):
    repo_tools = RepoTools(sample_repo)
    assert repo_tools.search_code("(").startswith("ERROR: invalid regex")
    assert "No matches" in repo_tools.search_code("does-not-exist-anywhere")


def test_git_log(sample_repo):
    output = RepoTools(sample_repo).git_log()
    assert "first commit" in output
    assert "Tester" in output


def test_dispatch_handles_unknown_tools_and_bad_arguments(sample_repo):
    repo_tools = RepoTools(sample_repo)
    assert "unknown tool" in repo_tools.dispatch("delete_everything", {})
    assert "bad arguments" in repo_tools.dispatch("read_file", {"nope": 1})


def test_dispatch_truncates_long_output(sample_repo, monkeypatch):
    monkeypatch.setattr(tools, "MAX_OUTPUT_CHARS", 20)
    assert "(output truncated)" in RepoTools(sample_repo).dispatch("read_file", {"path": "config.py"})


def test_run_ruff_reports_when_unavailable(sample_repo, monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    assert RepoTools(sample_repo).run_ruff().startswith("UNAVAILABLE")


def test_tools_leave_the_repository_unchanged(sample_repo):
    repo_tools = RepoTools(sample_repo)
    repo_tools.list_files()
    repo_tools.read_file("config.py")
    repo_tools.search_code("DEBUG")
    repo_tools.git_log()
    repo_tools.run_ruff()
    status = subprocess.run(
        ["git", "-C", str(sample_repo), "status", "--porcelain"], capture_output=True, text=True
    ).stdout
    assert status == ""
