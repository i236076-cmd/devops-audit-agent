# DevOps Repository Auditing Agent

![CI](https://github.com/i236076-cmd/devops-audit-agent/actions/workflows/ci.yml/badge.svg)

An AI agent that audits a public GitHub repository for DevOps best-practice violations and writes an
evidence-based JSON report. An LLM decides which inspection tool to call next, reads the result, follows
up on anything suspicious, and finally reports only what it verified.

Built for *MLOps & Production AI Engineering, Assignment 2*. The scope is software engineering and
DevOps only: ML training, datasets, model validation and experiment tracking are deliberately ignored.

## Quick start

```bash
sudo apt install -y git python3 python3-venv python3-pip   # Ubuntu / Debian
git clone https://github.com/i236076-cmd/devops-audit-agent.git
cd devops-audit-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your API key into .env (see "Configuration")
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

Requirements: Python 3.10 or newer, `git`, internet access, and one LLM API key.

## Configuration

Copy `.env.example` to `.env` and fill in **one** API key. The provider is detected from the key.
`.env` is git-ignored, so keys are never committed.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI key, or the key of any OpenAI-compatible service |
| `ANTHROPIC_API_KEY` | Anthropic key |
| `LLM_PROVIDER` | optional, `openai` or `anthropic` (auto-detected when empty) |
| `LLM_MODEL` | optional model name (defaults: `gpt-4o-mini`, `claude-sonnet-5-5`) |
| `LLM_BASE_URL` | optional, for OpenAI-compatible services |
| `LLM_REASONING_EFFORT` | optional, `low`, `medium` or `high` for models that support it |
| `LLM_MIN_INTERVAL` | optional, minimum seconds between LLM requests (default 1) |

Examples:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Groq (used during development; free tier)
OPENAI_API_KEY=<groq key>
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
LLM_REASONING_EFFORT=low

# Gemini through its OpenAI-compatible endpoint
OPENAI_API_KEY=<gemini key>
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=<a current Gemini model name>
```

The LLM provider or model does not change how the agent works. The Groq and Gemini routes were run
live during development. The Anthropic backend is covered by unit tests with fake clients only.

## Usage

```bash
python agent.py --repo https://github.com/owner/project --output report.json
```

| Option | Meaning |
|---|---|
| `--repo URL` | public GitHub repository (required) |
| `--output PATH` | where to write the JSON report (default `report.json`) |
| `--trace PATH` | where to write the tool-call trace (default `report.trace.json` next to the report) |
| `--no-trace` | do not write a trace file |
| `--max-steps N` | maximum LLM turns (default 20) |
| `--max-seconds N` | time budget before the agent must report (default 240) |

The agent runs unattended and never asks questions. Exit codes: `0` report written, `1` fatal error
(clone failed, LLM unavailable, ...), `2` invalid command line input, `130` interrupted. On a fatal
error no report is written and a message is printed to stderr.

## How it works

```
python agent.py --repo URL --output report.json
        |
        v
  shallow, read-only clone  ------>  RepoTools: list_files, read_file, search_code, git_log, run_ruff
        |                                      ^
        v                                      | tool results (secrets masked)
   agent loop (agent_core/loop.py)  <--->  LLM chooses the next tool, or calls submit_report
        |
        v
  report builder (agent_core/report.py): validate, de-duplicate, verify file/line, mask secrets
        |
        v
  report.json  (+ report.trace.json with every tool call)
```

The agent cycle, implemented in plain Python with no agent framework:

1. **Observe**: the first message gives the model the repository URL; every tool result is fed back.
2. **Reason and select a tool**: the model chooses which tool to call, with which arguments, and may
   request several calls at once. Follow-up calls depend on what earlier results showed.
3. **Execute**: `RepoTools.dispatch` runs the tool. It never raises, so a bad tool call becomes an
   `ERROR:` message the model can recover from.
4. **Interpret**: the model reads the output and decides whether to dig deeper.
5. **Report**: the model calls `submit_report` with its findings. `report.py` then cleans them.

### Tools (`tools.py`)

| Tool | Purpose |
|---|---|
| `list_files(directory, max_files)` | files tracked by git, optionally under one directory |
| `read_file(path, start_line, max_lines)` | file contents with line numbers |
| `search_code(pattern, path)` | regex search, returns `file:line: snippet` |
| `git_log(max_count, path)` | commit history |
| `run_ruff()` | optional static lint; says `UNAVAILABLE` when ruff is missing |

### Guard rails

- The model must inspect the repository at least three times before a report is accepted.
- A step limit and a time budget force the model to submit what it has found so far.
- Old tool outputs are shortened so the context stays small.
- Rate limits wait as long as the server asks (up to 60 s). A daily quota fails immediately with a
  clear message. Server errors and timeouts use exponential backoff. Auth errors never echo the key.
- The prompt contains the evidence rules from the assignment: an absent optional file is not a defect
  when an equivalent exists, remote branch protection is never inferred, healthy repositories get an
  empty list, and ML-specific topics are out of scope.
- Reports are post-processed in code: findings without evidence are dropped, duplicates are merged,
  severities are validated, a cited file must exist and a cited line must be inside it (otherwise they
  become `null`), and rule ids are renumbered per category.

## Safety

- The inspected repository is never modified and none of its code is executed (no pytest, no scripts,
  no installs). The agent only reads files and runs `git` and `ruff check` (static analysis).
- Cloning is shallow (last 200 commits) with hooks disabled and symlinks turned off, into a temporary
  directory that is deleted afterwards.
- Only `https://github.com/<owner>/<repo>` URLs are accepted. File paths are confined to the clone.
- Secrets are masked in every tool output, in the trace and in the report. The agent reports that a
  credential exists, never its value.
- Nothing about any specific repository is hard-coded.

## Output format

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "Credential assigned directly in source code",
      "recommendation": "Use a secrets manager or environment variable"
    }
  ]
}
```

`severity` is one of `critical`, `high`, `medium`, `low`. `file` and `line` are `null` when they do not
apply. `violations` is an empty list for a healthy repository. Categories: `repository_hygiene`,
`dependency_management`, `code_quality`, `testing`, `github_actions_ci`, `security`, `configuration`,
`collaboration`, `build_release` (rule id prefixes `HYG`, `DEP`, `QUAL`, `TEST`, `CI`, `SEC`, `CFG`,
`COL`, `BLD`). `sample_report.json` is real output produced by this agent.

## Project structure

```
agent.py              CLI entry point (--repo, --output)
tools.py              inspection tools, safe cloning, secret masking
agent_core/llm.py     provider-neutral LLM client (OpenAI-compatible and Anthropic) with retries
agent_core/loop.py    the agent loop, system prompt and guard rails
agent_core/report.py  report cleaning, validation and writing
tests/                pytest suite (no network and no API key needed)
.github/workflows/    CI: ruff and pytest on every push and pull request
requirements.txt      dependencies
.env.example          required variable names, no real credentials
sample_report.json    real example output
```

## Tests

```bash
pip install -r requirements.txt
pytest -q          # unit tests for the tools, report builder, LLM client, agent loop and CLI
ruff check .
```

The tests use a scripted fake LLM and temporary git repositories, so they run offline. They also check
that the tools leave a repository untouched, that secrets never reach the report, and that the CLI
exits non-zero without writing a report when something fails.

## Limitations

- Findings depend on the model. A stronger model usually finds more subtle problems.
- Free LLM tiers have small per-minute and per-day limits, which makes audits slower. The time budget
  keeps a run bounded, and `--max-seconds` can raise it for faster providers.
- Remote settings such as branch protection cannot be seen from repository files and are not reported.
- Only the last 200 commits are inspected, and files over 200 KB are read partially.
- `run_ruff` only helps for Python projects.
