"""The agent loop: Observe -> Reason -> Select tool -> Execute -> Interpret -> Report.

The LLM decides which inspection tool to call next, we execute it, feed the
result back, and repeat until the model calls submit_report. All control flow
and tool dispatch is plain Python (no agent frameworks).
"""

import json
import os
import re
import shutil
import sys
import time

from tools import TOOL_SCHEMAS, RepoTools, clone_repo, redact_secrets

MAX_STEPS = 20               # LLM turns before we force a final report
MAX_SECONDS = 240            # wall-clock budget before we force a final report
MIN_INVESTIGATION_CALLS = 3  # refuse a report before this many inspections
MAX_NUDGES = 2               # times we remind the model to use a tool
MAX_COVERAGE_REVIEWS = 1     # times a report is sent back to close inspection gaps
KEEP_FULL_OUTPUTS = 4        # newest tool outputs kept in full in the context
OLD_OUTPUT_CHARS = 400       # older tool outputs are shortened to this

CATEGORIES = [
    "repository_hygiene", "dependency_management", "code_quality", "testing",
    "github_actions_ci", "security", "configuration", "collaboration",
    "build_release",
]
SEVERITIES = ["critical", "high", "medium", "low"]

# File kinds a thorough audit should have looked at (used by the completeness check).
SECRET_WORDS = re.compile(r"(?i)secret|passw|token|api[_-]?key|credential")
MANIFESTS = re.compile(
    r"(?i)(^|/)(requirements[^/]*\.txt|pyproject\.toml|setup\.py|setup\.cfg|pipfile|package\.json|"
    r"go\.mod|pom\.xml|build\.gradle|gemfile|cargo\.toml|composer\.json)$"
)
CONFIG_FILES = re.compile(
    r"(?i)(^|/)([^/]*config[^/]*\.(py|ya?ml|json|toml|ini|js|ts)|settings[^/]*\.(py|ya?ml|json|toml|ini)|"
    r"\.env[^/]*|application[^/]*\.(properties|ya?ml)|docker-compose[^/]*\.ya?ml|dockerfile[^/]*|makefile)$"
)
WORKFLOWS = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")


def _norm_path(path):
    path = str(path or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


class AgentError(Exception):
    """Raised when the agent cannot finish an audit."""


SUBMIT_SCHEMA = {
    "name": "submit_report",
    "description": (
        "Finish the audit. Call exactly once, when the investigation is complete, "
        "with every violation you verified. Use an empty list if the repository is healthy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "description": "e.g. SEC-001"},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "severity": {"type": "string", "enum": SEVERITIES},
                        "file": {"type": "string", "description": "Path, omit if not applicable"},
                        "line": {"type": "integer", "description": "Line number, omit if not applicable"},
                        "description": {"type": "string"},
                        "evidence": {"type": "string", "description": "What you saw. Never include secret values."},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["rule_id", "category", "severity", "description",
                                 "evidence", "recommendation"],
                },
            }
        },
        "required": ["violations"],
    },
}

SYSTEM_PROMPT = """You are a senior DevOps engineer auditing a GitHub repository for DevOps best-practice violations. You work autonomously: call a tool, read what it returns, decide what to inspect next. The repository is already cloned and is read-only.

TOOLS
- list_files(directory, max_files): files tracked by git. Start here.
- read_file(path, start_line, max_lines): file contents with line numbers.
- search_code(pattern, path): regex search, returns file:line: snippet. path optionally limits it to one directory.
- git_log(max_count, path): commit history.
- run_ruff(): static lint for Python code.
- submit_report(violations): finish. Call it exactly once when the investigation is complete.

HOW TO INVESTIGATE
Form a hypothesis from what you observe, pick the tool that tests it, read the result, then follow up. Adapt to the project type (Python, Node, Docker, ...). Request several independent tool calls in the same turn instead of one at a time. Read configuration, CI, dependency and build files in full, but do not read every source file: use search_code to look for specific problems instead. Areas to cover:
1. repository_hygiene: README present and not empty, .gitignore suitable for the language, tracked build/temp/generated files (__pycache__, .pyc, node_modules, .DS_Store, *.log, dist/, committed .env), large binaries.
2. dependency_management: manifests vs lockfiles, conflicting or duplicate specifications, unpinned or very loose version constraints.
3. code_quality: linter/formatter setup (ruff, flake8, black, eslint, prettier, pre-commit) and obvious maintainability problems you actually saw.
4. testing: are there tests? Does CI run them? Do they look broken?
5. github_actions_ci: read every file in .github/workflows. Check triggers (wrong branches, missing pull_request), steps that skip or neutralise checks (continue-on-error on tests or lint, "|| true", "if: false", commented-out steps), jobs that never run tests, actions pinned to @master/@main.
6. security: hardcoded credentials (search for key/token/password/secret patterns, private keys, committed .env files), workflow permissions (write-all, unneeded contents: write), pull_request_target that checks out untrusted PR code, secrets printed to logs, "curl | bash".
7. configuration: machine-specific absolute paths (/home/<user>, C:\\Users), committed local settings (.vscode, .idea), no separation of dev and prod configuration.
8. collaboration: unsafe PR automation, missing review guidance (PR template, CODEOWNERS) only where relevant.
9. build_release: Dockerfile problems (latest tags, running as root), release or deploy automation that skips tests, no build validation in CI.

EVIDENCE RULES (precision matters: a wrong finding is penalised)
- Report a violation only if you saw the evidence yourself in a tool result. Give the file and the exact line number from read_file or search_code output.
- An absent optional file is not a defect if an equivalent exists (pyproject.toml instead of requirements.txt, ruff configured in pyproject, flake8 instead of ruff, pre-commit instead of a CI lint job). Check for equivalents before claiming something is missing.
- Never infer remote settings such as branch protection from repository files.
- Do not report the same underlying problem twice, and do not report a consequence of another finding as a separate violation. Report each distinct defect separately, even when several are in the same file: for example a dangerous workflow trigger and an over-broad permission are two entries.
- Never report something about a file you have not read. File names alone are not evidence.
- The line must be the line where the problem is. Unpinned action or image tags are low severity.
- Always give file and line when the evidence is in a specific file. Omit them only for problems about something missing.
- run_ruff output is style-level. Report it only for clear defects such as syntax errors or undefined names, never for rule codes that only express style preferences.
- An empty file (0 lines) where content is expected, such as a README or test file, counts as missing. Empty or missing tests belong in category testing.
- Example or template files such as .env.example are good practice, not a risk. A linter that is installed but has no config file is not a defect, because tools have sensible defaults.
- If the repository looks healthy, report an empty list. Do not invent problems.
- Scope is software engineering and DevOps only. Ignore ML training code, datasets, model quality and experiment tracking.
- Never write a secret's value anywhere. State that a credential is hardcoded, not what it is.
- If a tool returns UNAVAILABLE or ERROR, do not guess what it would have shown.

SEVERITY
critical = exposed credential or exploitable CI/workflow hole; high = broken or bypassed quality gate, risky deploy; medium = missing key practice (no lockfile, no tests, wrong .gitignore); low = minor hygiene.

RULE IDS
PREFIX-NNN where PREFIX is one of HYG, DEP, QUAL, TEST, CI, SEC, CFG, COL, BLD and NNN counts up from 001 per prefix.

When finished, call submit_report once. Each violation needs rule_id, category, severity, description, evidence, recommendation, plus file and line when they apply (omit them when they do not)."""


class AuditAgent:
    def __init__(self, llm, tools, repo_url, max_steps=MAX_STEPS, min_interval=None, log=None,
                 max_seconds=MAX_SECONDS, coverage_reviews=MAX_COVERAGE_REVIEWS):
        self.llm = llm
        self.tools = tools
        self.repo_url = repo_url
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.coverage_reviews = coverage_reviews
        if min_interval is None:
            try:
                min_interval = float(os.getenv("LLM_MIN_INTERVAL") or "1.0")
            except ValueError:
                min_interval = 1.0
        self.min_interval = min_interval
        self.log = log or (lambda text: print(text, file=sys.stderr))
        self.trace = []
        self._t0 = time.monotonic()
        self._last_call = 0.0

    # ---- helpers ---------------------------------------------------------
    def _ask(self, messages, tool_defs):
        """One LLM request, spaced out to respect provider rate limits."""
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.llm.chat(SYSTEM_PROMPT, self._compact(messages), tool_defs)
        finally:
            self._last_call = time.monotonic()

    @staticmethod
    def _compact(messages):
        """Shorten old tool outputs so the context does not grow without limit."""
        tool_positions = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        old = set(tool_positions[:-KEEP_FULL_OUTPUTS]) if len(tool_positions) > KEEP_FULL_OUTPUTS else set()
        compacted = []
        for i, msg in enumerate(messages):
            if i in old and len(msg["content"]) > OLD_OUTPUT_CHARS:
                msg = dict(msg, content=msg["content"][:OLD_OUTPUT_CHARS] + "\n... [older output shortened]")
            compacted.append(msg)
        return compacted

    def _record(self, step, name, arguments, output):
        self.trace.append({
            "step": step,
            "tool": name,
            "arguments": arguments,
            "output_preview": redact_secrets(output)[:300],
        })
        elapsed = time.monotonic() - self._t0
        self.log(f"[step {step} +{elapsed:.0f}s] {name}({json.dumps(arguments)[:120]})")

    @staticmethod
    def _check_submission(arguments, inspections):
        """Return (message for the model, violations or None)."""
        violations = arguments.get("violations")
        if not isinstance(violations, list):
            return "ERROR: 'violations' must be a list. Call submit_report again.", None
        if inspections < MIN_INVESTIGATION_CALLS:
            return ("Not accepted: you have inspected too little of the repository. "
                    "Look at the structure, CI workflows, dependency files and .gitignore first."), None
        return "Report accepted.", violations

    def _coverage_gaps(self, violations):
        """Inspection areas the model skipped. The model decides what to do about them."""
        files = self.tools.tracked_files()
        read = {_norm_path(c["arguments"].get("path")) for c in self.trace if c["tool"] == "read_file"}
        gaps = []
        searched_secrets = any(
            c["tool"] == "search_code" and SECRET_WORDS.search(str(c["arguments"].get("pattern", "")))
            for c in self.trace
        )
        if not searched_secrets:
            gaps.append("You never searched the code for hardcoded credentials. Use search_code "
                        "with patterns such as api_key, secret, token, password.")
        for label, pattern in (("dependency files", MANIFESTS), ("configuration files", CONFIG_FILES),
                               ("CI workflow files", WORKFLOWS)):
            unread = [name for name in files if pattern.search(name) and name not in read]
            if unread:
                gaps.append(f"These {label} were never read: {', '.join(unread[:6])}.")
        has_readme = any(re.match(r"(?i)^readme(\.\w+)?$", name) for name in files)
        mentions_readme = any("readme" in str(v).lower() for v in violations if isinstance(v, dict))
        if not has_readme and not mentions_readme:
            gaps.append("The file list has no README at the repository root. Check for an equivalent "
                        "(for example under docs/); if there is none, report the missing README.")
        return gaps

    # ---- main loop -------------------------------------------------------
    def run(self):
        """Investigate the repository and return the raw list of violations."""
        messages = [{
            "role": "user",
            "content": f"Audit this repository: {self.repo_url}\n"
                       "It is already cloned. Begin by inspecting its structure.",
        }]
        tool_defs = TOOL_SCHEMAS + [SUBMIT_SCHEMA]
        inspections = 0
        nudges = 0
        reviews_left = self.coverage_reviews
        self._t0 = time.monotonic()

        for step in range(1, self.max_steps + 1):
            if time.monotonic() - self._t0 > self.max_seconds:
                self.log("[agent] time budget used up")
                break
            response = self._ask(messages, tool_defs)

            if not response.tool_calls:
                nudges += 1
                if nudges > MAX_NUDGES:
                    raise AgentError("The model stopped using tools without submitting a report")
                messages.append({"role": "assistant", "content": response.text or "(no output)"})
                messages.append({
                    "role": "user",
                    "content": "Continue the audit by calling a tool, or call submit_report if you are done.",
                })
                continue

            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments, "extra": c.extra}
                    for c in response.tool_calls
                ],
            })
            accepted = None
            for call in response.tool_calls:
                if call.name == "submit_report":
                    output, violations = self._check_submission(call.arguments, inspections)
                    if violations is not None:
                        gaps = self._coverage_gaps(violations) if reviews_left > 0 else []
                        if gaps:
                            reviews_left -= 1
                            output = ("Not accepted yet. Before finishing, close these gaps:\n- "
                                      + "\n- ".join(gaps)
                                      + "\nInspect them with your tools, add any real findings, "
                                        "then call submit_report again.")
                        else:
                            accepted = violations
                else:
                    output = self.tools.dispatch(call.name, call.arguments)
                    inspections += 1
                self._record(step, call.name, call.arguments if call.name != "submit_report" else {}, output)
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name, "content": output,
                })
            if accepted is not None:
                return accepted

        return self._force_report(messages)

    def _force_report(self, messages):
        """Step limit reached: make the model submit what it has found."""
        self.log("[agent] step limit reached, requesting final report")
        messages.append({
            "role": "user",
            "content": "Step limit reached. Call submit_report now with the violations you have verified.",
        })
        response = self._ask(messages, [SUBMIT_SCHEMA])
        for call in response.tool_calls:
            if call.name == "submit_report" and isinstance(call.arguments.get("violations"), list):
                self._record(self.max_steps + 1, "submit_report", {}, "forced report")
                return call.arguments["violations"]
        raise AgentError("The model did not submit a report before the step limit")


if __name__ == "__main__":
    # Quick manual test:  python -m agent_core.loop https://github.com/owner/repo
    from agent_core.llm import create_client

    if len(sys.argv) != 2:
        sys.exit("usage: python -m agent_core.loop <github_repo_url>")
    path = clone_repo(sys.argv[1])
    try:
        agent = AuditAgent(create_client(), RepoTools(path), sys.argv[1])
        found = agent.run()
        print(json.dumps(found, indent=2))
        print(f"\n{len(found)} violations, {len(agent.trace)} tool calls", file=sys.stderr)
    finally:
        shutil.rmtree(path, ignore_errors=True)
