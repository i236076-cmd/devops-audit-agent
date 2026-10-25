"""DevOps repository auditing agent.

Usage:
    python agent.py --repo https://github.com/owner/project --output report.json

Exit codes: 0 report written, 1 fatal error during the audit,
2 invalid command line input, 130 interrupted.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

from agent_core.llm import LLMError, create_client
from agent_core.loop import MAX_SECONDS, MAX_STEPS, AgentError, AuditAgent
from agent_core.report import build_report, write_json, write_report
from tools import RepoTools, clone_repo, redact_secrets, validate_repo_url


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit a public GitHub repository for DevOps best-practice violations."
    )
    parser.add_argument("--repo", required=True, help="public GitHub repository URL")
    parser.add_argument("--output", default="report.json", help="where to write the JSON report")
    parser.add_argument("--trace", help="where to write the tool-call trace "
                                        "(default: next to the report, e.g. report.trace.json)")
    parser.add_argument("--no-trace", action="store_true", help="do not write a trace file")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help=f"maximum LLM turns (default {MAX_STEPS})")
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS,
                        help=f"time budget before the agent must report (default {MAX_SECONDS})")
    return parser.parse_args(argv)


def fail(message, code=1):
    print(f"error: {redact_secrets(str(message))}", file=sys.stderr)
    return code


def main(argv=None):
    args = parse_args(argv)
    try:
        repo_url = validate_repo_url(args.repo)
    except ValueError as exc:
        return fail(exc, 2)
    if args.max_steps < 1 or args.max_seconds < 1:
        return fail("--max-steps and --max-seconds must be positive", 2)

    trace_path = None if args.no_trace else (args.trace or str(Path(args.output).with_suffix(".trace.json")))
    for stale in (args.output, trace_path):  # never leave the results of an earlier run behind
        if stale:
            try:
                Path(stale).unlink(missing_ok=True)
            except OSError:
                pass

    clone_path = None
    started = time.monotonic()
    try:
        print(f"Cloning {repo_url} ...", file=sys.stderr)
        clone_path = clone_repo(repo_url)
        llm = create_client()
        agent = AuditAgent(
            llm, RepoTools(clone_path), repo_url,
            max_steps=args.max_steps, max_seconds=args.max_seconds,
        )
        raw_violations = agent.run()
        report = build_report(repo_url, raw_violations, clone_path)
        write_report(args.output, report)
    except KeyboardInterrupt:
        return fail("interrupted", 130)
    except (ValueError, RuntimeError, LLMError, AgentError, OSError) as exc:
        return fail(exc)
    except Exception as exc:  # last resort: still exit non-zero with a message
        return fail(f"unexpected {type(exc).__name__}: {exc}")
    finally:
        if clone_path:
            shutil.rmtree(clone_path, ignore_errors=True)

    if trace_path:
        try:
            write_json(trace_path, {
                "repository": repo_url,
                "model": f"{llm.provider}/{llm.model}",
                "tool_calls": agent.trace,
            })
        except OSError as exc:
            print(f"warning: could not write trace: {exc}", file=sys.stderr)

    elapsed = time.monotonic() - started
    print(
        f"Done: {len(report['violations'])} violations, {len(agent.trace)} tool calls, "
        f"{elapsed:.0f}s -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
