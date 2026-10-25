"""Turn the agent's raw findings into the final, validated JSON report.

The LLM can be sloppy (wrong severity, a file that does not exist, the same
issue twice, a secret pasted into the evidence). Everything is cleaned here in
plain code so the report always matches the required output contract.
"""

import json
import os
import re
import tempfile
from pathlib import Path

from tools import redact_secrets

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
PREFIXES = {
    "repository_hygiene": "HYG",
    "dependency_management": "DEP",
    "code_quality": "QUAL",
    "testing": "TEST",
    "github_actions_ci": "CI",
    "security": "SEC",
    "configuration": "CFG",
    "collaboration": "COL",
    "build_release": "BLD",
}
REQUIRED_FIELDS = [
    "rule_id", "category", "severity", "file", "line",
    "description", "evidence", "recommendation",
]
MAX_TEXT = 600
DEFAULT_RECOMMENDATION = "Review this finding and fix it following the project's DevOps practices."


def _clean_text(value, limit=MAX_TEXT):
    """Single-line, secret-free text of bounded length."""
    if not isinstance(value, str):
        return ""
    return " ".join(redact_secrets(value).split())[:limit]


def _clean_path(value, repo_root):
    """Repo-relative path that really exists in the repository, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = redact_secrets(value.strip()).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    if not path:
        return None
    if repo_root is not None:
        root = Path(repo_root).resolve()
        full = (root / path).resolve()
        if root not in full.parents or not full.is_file():
            return None
    return path


def _clean_line(value, path, repo_root):
    """Positive line number inside the file, else None."""
    if path is None or isinstance(value, bool):
        return None
    try:
        line = int(value)
    except (TypeError, ValueError):
        return None
    if line < 1:
        return None
    if repo_root is not None:
        with open(Path(repo_root) / path, "rb") as handle:
            total = sum(1 for _ in handle)
        if line > total:
            return None
    return line


def normalize_violations(raw, repo_root=None):
    """Validate, de-duplicate, sort and renumber the agent's findings."""
    candidates = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        description = _clean_text(item.get("description"))
        evidence = _clean_text(item.get("evidence"))
        if not description or not evidence:
            continue  # a claim without evidence is not reported
        category = re.sub(r"[^a-z0-9_]+", "_", str(item.get("category", "")).strip().lower()).strip("_")
        category = category or "uncategorized"
        severity = str(item.get("severity", "")).strip().lower()
        if severity not in SEVERITY_ORDER:
            severity = "medium"
        path = _clean_path(item.get("file"), repo_root)
        line = _clean_line(item.get("line"), path, repo_root)

        match = re.match(r"^([A-Z]{2,5})-\d+$", str(item.get("rule_id", "")).strip().upper())
        prefix = PREFIXES.get(category) or (match.group(1) if match else "GEN")
        # two findings on the same file and line describe the same defect
        key = (path, line) if line else (category, path, description[:50].lower())
        candidates.append((key, {
            "rule_id": prefix,  # numbered below
            "category": category,
            "severity": severity,
            "file": path,
            "line": line,
            "description": description,
            "evidence": evidence,
            "recommendation": _clean_text(item.get("recommendation")) or DEFAULT_RECOMMENDATION,
        }))

    # most severe first, so the highest severity survives when findings are merged
    candidates.sort(key=lambda c: (SEVERITY_ORDER[c[1]["severity"]], c[1]["file"] or "", c[1]["line"] or 0))
    cleaned, seen = [], set()
    for key, violation in candidates:
        if key not in seen:
            seen.add(key)
            cleaned.append(violation)

    counters = {}
    for violation in cleaned:
        prefix = violation["rule_id"]
        counters[prefix] = counters.get(prefix, 0) + 1
        violation["rule_id"] = f"{prefix}-{counters[prefix]:03d}"
    return cleaned


def build_report(repo_url, raw_violations, repo_root=None):
    return {"repository": repo_url, "violations": normalize_violations(raw_violations, repo_root)}


def validate_report(report):
    """Raise ValueError if the report does not follow the output contract."""
    if not isinstance(report, dict) or not isinstance(report.get("repository"), str):
        raise ValueError("report needs a string 'repository'")
    violations = report.get("violations")
    if not isinstance(violations, list):
        raise ValueError("report needs a 'violations' list")
    for number, violation in enumerate(violations, 1):
        if not isinstance(violation, dict):
            raise ValueError(f"violation {number} is not an object")
        missing = [name for name in REQUIRED_FIELDS if name not in violation]
        if missing:
            raise ValueError(f"violation {number} is missing {missing}")
        if violation["severity"] not in SEVERITY_ORDER:
            raise ValueError(f"violation {number} has invalid severity {violation['severity']!r}")
        if violation["file"] is not None and not isinstance(violation["file"], str):
            raise ValueError(f"violation {number}: file must be a string or null")
        if violation["line"] is not None and (
            isinstance(violation["line"], bool) or not isinstance(violation["line"], int)
        ):
            raise ValueError(f"violation {number}: line must be an integer or null")
        for name in ("rule_id", "category", "description", "evidence", "recommendation"):
            if not isinstance(violation[name], str) or not violation[name]:
                raise ValueError(f"violation {number}: {name} must be a non-empty string")


def write_json(path, data):
    """Write JSON atomically so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def write_report(path, report):
    validate_report(report)
    write_json(path, report)
