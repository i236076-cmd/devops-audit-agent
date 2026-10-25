import subprocess

import pytest


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def sample_repo(tmp_path):
    """A tiny git repository with a hardcoded credential and a CI file."""
    repo = tmp_path / "repo"
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "config.py").write_text('API_KEY = "abcd1234efgh5678"\nDEBUG = True\n')
    (repo / "README.md").write_text("# demo\n")
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: ci\non: push\n")
    (repo / "logo.bin").write_bytes(b"\x00\x01\x02binary")
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "tester@example.com")
    git(repo, "config", "user.name", "Tester")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first commit")
    return repo
