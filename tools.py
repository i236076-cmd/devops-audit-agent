"""Inspection tools for the DevOps auditing agent.

Every tool here is read-only. Nothing from the inspected repository is ever
executed (no pytest, no scripts, no installs); we only read files and run
git / ruff as static inspection. All output is passed through redact_secrets()
so credentials never reach the LLM, the logs or the report.
"""

import codecs
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

GITHUB_URL_RE = re.compile(
    r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)

CLONE_DEPTH = 200
CLONE_TIMEOUT = 120      # seconds
CMD_TIMEOUT = 30         # seconds
SEARCH_TIMEOUT = 20      # seconds
MAX_FILE_BYTES = 200_000
MAX_OUTPUT_CHARS = 6000

# Patterns that look like credentials. Used to mask values before output.
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    # key = "value" style assignments (keeps the key name, hides the value)
    re.compile(
        r"""(?i)((?:api[_-]?key|secret|token|passw(?:or)?d|pwd|access[_-]?key)[\w-]*\s*[:=]\s*)(["'])[^"'\n]{4,}\2"""
    ),
]


# Unquoted env-style lines such as DB_PASSWORD=hunter2 (typical for .env files).
ENV_SECRET = re.compile(
    r"(?m)^([ \t]*(?:export[ \t]+)?[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD)[A-Z0-9_]*[ \t]*=[ \t]*)"
    r"(?![\"'$#{(\[])(?![^\s#]*[(\[{])([^\s#]+)"
)
# Obvious placeholders stay readable so example files are not mistaken for leaks.
PLACEHOLDER = re.compile(r"(?i)^(your[_-].*|<.*|change[_-]?me|x{3,}|\*{3,}|example.*|placeholder.*|.*[_-]here|todo|none|null)$")


def _mask_env(match):
    if PLACEHOLDER.match(match.group(2)):
        return match.group(0)
    return match.group(1) + "[REDACTED]"


def redact_secrets(text):
    """Mask anything that looks like a credential, keeping line numbers intact."""

    def _mask(match):
        if match.lastindex:  # assignment pattern: keep name and quotes
            return match.group(1) + match.group(2) + "[REDACTED]" + match.group(2)
        # keep the same number of lines so line numbers don't shift
        return "[REDACTED]" + "\n" * match.group(0).count("\n")

    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_mask, text)
    return ENV_SECRET.sub(_mask_env, text)


# --------------------------------------------------------------------------
# Cloning
# --------------------------------------------------------------------------
def validate_repo_url(url):
    """Return a clean https://github.com/owner/name URL or raise ValueError."""
    match = GITHUB_URL_RE.match((url or "").strip())
    if not match:
        raise ValueError(
            f"Invalid repository URL: {url!r}. Expected https://github.com/<owner>/<repo>"
        )
    return f"https://github.com/{match.group(1)}/{match.group(2)}"


def clone_repo(url, dest=None, depth=CLONE_DEPTH):
    """Shallow-clone a public GitHub repo into dest (a temp dir by default)."""
    clean_url = validate_repo_url(url)
    dest = dest or tempfile.mkdtemp(prefix="audit_")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    cmd = [
        "git",
        "-c", "core.symlinks=false",
        "-c", "core.hooksPath=/dev/null",
        "clone", "--quiet", "--no-tags", "--depth", str(depth),
        clean_url + ".git", dest,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=CLONE_TIMEOUT, env=env
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Cloning {clean_url} timed out after {CLONE_TIMEOUT}s")
    except FileNotFoundError:
        raise RuntimeError("git is not installed or not on PATH")
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not clone {clean_url}: {redact_secrets(result.stderr.strip())}"
        )
    return dest


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
class RepoTools:
    """Read-only inspection tools bound to one cloned repository."""

    def __init__(self, repo_path):
        self.root = Path(repo_path).resolve()
        self._tracked = None

    # ---- helpers ---------------------------------------------------------
    def _git(self, *args):
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True, text=True, errors="replace",
            timeout=CMD_TIMEOUT, env=env,
        )

    def _tracked_files(self):
        """Files tracked by git (falls back to walking the folder)."""
        if self._tracked is not None:
            return self._tracked
        files = []
        try:
            res = self._git("ls-files", "-z")
            if res.returncode == 0 and res.stdout:
                files = [f for f in res.stdout.split("\0") if f]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if not files:
            for folder, dirs, names in os.walk(self.root):
                dirs[:] = [d for d in dirs if d != ".git"]
                for name in names:
                    full = Path(folder) / name
                    files.append(str(full.relative_to(self.root)))
        self._tracked = sorted(files)
        return self._tracked

    def _safe_path(self, rel_path):
        path = (self.root / rel_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("path is outside the repository")
        return path

    @staticmethod
    def _looks_binary(chunk):
        """True for NUL bytes or data that is not valid UTF-8 text."""
        if b"\0" in chunk:
            return True
        try:
            codecs.getincrementaldecoder("utf-8")().decode(chunk, final=False)
        except UnicodeDecodeError:
            return True
        return False

    # ---- required tools --------------------------------------------------
    def list_files(self, directory="", max_files=300):
        """List files tracked in the repo, optionally under one directory."""
        prefix = directory.strip().strip("/")
        files = self._tracked_files()
        if prefix:
            files = [f for f in files if f == prefix or f.startswith(prefix + "/")]
        total = len(files)
        shown = files[: int(max_files)]
        header = f"{total} files" + (f" under '{prefix}'" if prefix else "")
        if total > len(shown):
            header += f" (showing first {len(shown)}; pass 'directory' to narrow down)"
        return header + ":\n" + "\n".join(shown)

    def read_file(self, path, start_line=1, max_lines=250):
        """Read a text file with line numbers (secrets are masked)."""
        try:
            full = self._safe_path(path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not full.exists():
            return f"ERROR: file not found: {path}"
        if full.is_dir():
            return f"ERROR: {path} is a directory, use list_files instead"
        try:
            with open(full, "rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            return f"ERROR: could not read {path}: {exc}"
        if self._looks_binary(raw[:2048]):
            return f"ERROR: {path} looks like a binary file"
        truncated = len(raw) > MAX_FILE_BYTES
        text = redact_secrets(raw[:MAX_FILE_BYTES].decode("utf-8", errors="replace"))
        lines = text.splitlines()
        start = max(int(start_line), 1)
        chunk = lines[start - 1: start - 1 + int(max_lines)]
        if not chunk:
            return f"{path} has {len(lines)} lines; nothing at line {start}"
        numbered = "\n".join(f"{start + i:>4} | {line}" for i, line in enumerate(chunk))
        note = ""
        if start - 1 + len(chunk) < len(lines):
            note = f"\n... ({len(lines)} lines total, use start_line to read more)"
        if truncated:
            note += "\n... (file larger than 200KB, only the start was read)"
        return f"{path}:\n{numbered}{note}"

    def search_code(self, pattern, path="", ignore_case=True, max_results=40):
        """Regex search across tracked text files. Returns file:line: snippet."""
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return f"ERROR: invalid regex {pattern!r}: {exc}"
        results = []
        deadline = time.monotonic() + SEARCH_TIMEOUT
        prefix = (path or "").strip().strip("/")
        for rel in self._tracked_files():
            if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
                continue
            if time.monotonic() > deadline:
                results.append("... search stopped early (time limit)")
                break
            full = self.root / rel
            try:
                if not full.is_file() or full.stat().st_size > MAX_FILE_BYTES:
                    continue
                raw = full.read_bytes()
            except OSError:
                continue
            if self._looks_binary(raw[:2048]):
                continue
            for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    snippet = redact_secrets(line.strip())[:200]
                    results.append(f"{rel}:{number}: {snippet}")
                    if len(results) >= int(max_results):
                        results.append("... more matches truncated")
                        return "\n".join(results)
        if not results:
            return f"No matches for pattern {pattern!r}"
        return "\n".join(results)

    def git_log(self, max_count=20, path=None):
        """Recent commit history: hash | author | date | subject."""
        count = min(max(int(max_count), 1), 100)
        args = ["log", f"--max-count={count}", "--date=short",
                "--pretty=format:%h | %an | %ad | %s"]
        if path:
            try:
                self._safe_path(path)
            except ValueError as exc:
                return f"ERROR: {exc}"
            args += ["--", path]
        try:
            res = self._git(*args)
        except subprocess.TimeoutExpired:
            return "ERROR: git log timed out"
        except FileNotFoundError:
            return "ERROR: git is not installed"
        if res.returncode != 0:
            return f"ERROR: git log failed: {redact_secrets(res.stderr.strip())}"
        if not res.stdout.strip():
            return "No commits found"
        out = redact_secrets(res.stdout.strip())
        shallow = self._git("rev-parse", "--is-shallow-repository").stdout.strip()
        if shallow == "true":
            out += f"\n(history limited to the last {CLONE_DEPTH} commits)"
        return out

    # ---- optional tool ---------------------------------------------------
    def run_ruff(self):
        """Static lint with ruff (reads code, never runs it, never fixes)."""
        exe = shutil.which("ruff")
        if not exe:
            return "UNAVAILABLE: ruff is not installed, so no lint result exists"
        try:
            res = subprocess.run(
                [exe, "check", "--no-cache", "--exit-zero", "--output-format=concise", "."],
                cwd=self.root, capture_output=True, text=True, errors="replace",
                timeout=CMD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: ruff timed out"
        lines = (res.stdout + res.stderr).strip().splitlines()
        if not lines:
            return "ruff produced no output"
        if len(lines) > 60:
            lines = lines[:60] + [f"... {len(lines) - 60} more lines"]
        return redact_secrets("\n".join(lines))

    # ---- dispatch --------------------------------------------------------
    def dispatch(self, name, arguments):
        """Run a tool by name (used by the agent loop). Never raises."""
        tools = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
            "git_log": self.git_log,
            "run_ruff": self.run_ruff,
        }
        func = tools.get(name)
        if func is None:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(tools)}"
        try:
            output = func(**(arguments or {}))
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # keep the agent loop alive
            return f"ERROR: {name} failed: {exc}"
        output = redact_secrets(str(output))
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"
        return output


# Tool descriptions the LLM sees (used in Step 3/4).
TOOL_SCHEMAS = [
    {
        "name": "list_files",
        "description": "List files tracked in the repository. Use first to learn the structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Optional folder to list, e.g. .github/workflows"},
                "max_files": {"type": "integer", "description": "Max files to return (default 300)"},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file with line numbers. Secrets are masked.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the repo root"},
                "start_line": {"type": "integer"},
                "max_lines": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Regex search over tracked files. Returns file:line: snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "path": {"type": "string", "description": "Optional directory or file to restrict the search to"},
                "ignore_case": {"type": "boolean"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "git_log",
        "description": "Show recent commits (hash | author | date | subject), optionally for one path.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_count": {"type": "integer"},
                "path": {"type": "string"},
            },
        },
    },
    {
        "name": "run_ruff",
        "description": "Run ruff lint (static, no code execution). Returns UNAVAILABLE if ruff is missing.",
        "parameters": {"type": "object", "properties": {}},
    },
]
