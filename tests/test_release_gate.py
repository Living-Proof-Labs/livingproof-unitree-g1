# SPDX-License-Identifier: Apache-2.0
"""Releases only run after Test succeeds on the same main commit.

Issue #69: an ungated push-triggered release.yml cuts a release even when
Test fails on the same commit. Gating on workflow_run(workflows=[Test],
types=[completed], branches=[main]) requires conclusion == 'success' and
head_sha == github.sha so semantic-release never tags a failed, cancelled,
skipped, PR, or newer unverified commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
TEST = WORKFLOWS / "test.yml"

UNGATED_PUSH_RELEASE = """\
# SPDX-License-Identifier: Apache-2.0
name: Release
on:
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  release:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: write
    uses: Living-Proof-Labs/collaboration-guidelines/.github/workflows/semantic-release.yml@v9.6.10
"""

_TOKEN = re.compile(r"\s*(?:(&&|\|\||==|!=|[()!])|'([^']*)'|([A-Za-z_][A-Za-z0-9_.-]*))")
_SHA_A = "a" * 40
_SHA_B = "b" * 40
_REPO = "Living-Proof-Labs/livingproof-unitree-g1"


def _tokenize_expr(expr: str) -> list[tuple[str, str]]:
    text = expr.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN.match(text, pos)
        if not match:
            raise ValueError(f"unsupported token in if expression near {text[pos:]!r}")
        op, quoted, ident = match.groups()
        if op is not None:
            tokens.append(("OP", op))
        elif quoted is not None:
            tokens.append(("STR", quoted))
        else:
            tokens.append(("ID", ident))
        pos = match.end()
    return tokens


def _lookup(path: str, ctx: dict) -> object:
    cur: object = ctx
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return cur


def _eval_expr(expr: str, ctx: dict) -> bool:
    """Evaluate a GitHub Actions boolean if-expression against a context dict."""
    tokens = _tokenize_expr(expr)
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> tuple[str, str]:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_or() -> object:
        val = parse_and()
        while peek() == ("OP", "||"):
            take()
            rhs = parse_and()
            val = bool(val) or bool(rhs)
        return val

    def parse_and() -> object:
        val = parse_eq()
        while peek() == ("OP", "&&"):
            take()
            rhs = parse_eq()
            val = bool(val) and bool(rhs)
        return val

    def parse_eq() -> object:
        val = parse_unary()
        while peek() in {("OP", "=="), ("OP", "!=")}:
            _, op = take()
            rhs = parse_unary()
            val = (val == rhs) if op == "==" else (val != rhs)
        return val

    def parse_unary() -> object:
        if peek() == ("OP", "!"):
            take()
            return not bool(parse_unary())
        return parse_atom()

    def parse_atom() -> object:
        tok = peek()
        if tok is None:
            raise ValueError("unexpected end of expression")
        kind, text = take()
        if kind == "OP" and text == "(":
            val = parse_or()
            if take() != ("OP", ")"):
                raise ValueError("unclosed parenthesis")
            return val
        if kind == "STR":
            return text
        if kind == "ID":
            if text == "true":
                return True
            if text == "false":
                return False
            return _lookup(text, ctx)
        raise ValueError(f"unexpected token {(kind, text)!r}")

    result = bool(parse_or())
    if pos != len(tokens):
        raise ValueError(f"trailing tokens: {tokens[pos:]!r}")
    return result


def _context(
    *,
    event_name: str = "workflow_run",
    trigger_event: str = "push",
    head_branch: str = "main",
    head_repo: str = _REPO,
    conclusion: str = "success",
    head_sha: str = _SHA_A,
    sha: str = _SHA_A,
) -> dict:
    return {
        "github": {
            "event_name": event_name,
            "ref": "refs/heads/main",
            "sha": sha,
            "repository": _REPO,
            "event": {
                "workflow_run": {
                    "event": trigger_event,
                    "head_branch": head_branch,
                    "head_repository": {"full_name": head_repo},
                    "conclusion": conclusion,
                    "head_sha": head_sha,
                }
            },
        }
    }


def _evaluate_release_gate_cases(expr: str) -> list[str]:
    problems: list[str] = []
    if not _eval_expr(expr, _context()):
        problems.append("release job rejects a passing main push Test run on matching SHA")
    bad_cases = [
        ("failed Test", _context(conclusion="failure")),
        ("cancelled Test", _context(conclusion="cancelled")),
        ("skipped Test", _context(conclusion="skipped")),
        ("timed_out Test", _context(conclusion="timed_out")),
        ("neutral Test", _context(conclusion="neutral")),
        ("empty conclusion", _context(conclusion="")),
        ("branch tip moved past head_sha", _context(head_sha=_SHA_A, sha=_SHA_B)),
        ("pull_request event on branch named main", _context(trigger_event="pull_request")),
        (
            "merge_group event",
            _context(trigger_event="merge_group", head_branch="gh-readonly-queue/main/1"),
        ),
        ("workflow_dispatch event", _context(trigger_event="workflow_dispatch")),
        ("non-main head_branch", _context(head_branch="feature")),
        ("fork head_repository", _context(head_repo="someone-else/livingproof-unitree-g1")),
        ("direct push event_name", _context(event_name="push")),
    ]
    for label, ctx in bad_cases:
        if _eval_expr(expr, ctx):
            problems.append(f"release job would still run on {label}")
    return problems


def release_gate_problems(release_text: str, test_text: str) -> list[str]:
    """Return problems if release.yml is not strictly gated on Test passing for the same SHA."""
    problems: list[str] = []
    release_doc = yaml.safe_load(release_text)
    test_doc = yaml.safe_load(test_text)
    if not isinstance(release_doc, dict) or not isinstance(test_doc, dict):
        return ["workflow YAML did not parse as a mapping"]

    test_name = test_doc.get("name")
    if not isinstance(test_name, str) or not test_name:
        problems.append("test.yml must define a non-empty workflow name")

    trigger = release_doc.get("on", release_doc.get(True))
    if not isinstance(trigger, dict):
        return [*problems, "release.yml must define an `on:` mapping"]

    if sorted(trigger.keys()) != ["workflow_run"]:
        problems.append(
            f"release.yml must trigger only on workflow_run, got {sorted(trigger.keys())}"
        )
    wf_run = trigger.get("workflow_run")
    if not isinstance(wf_run, dict):
        problems.append("release.yml workflow_run trigger must be a mapping")
    else:
        if wf_run.get("workflows") != [test_name]:
            problems.append(
                f"release.yml workflow_run.workflows must be [{test_name!r}], "
                f"got {wf_run.get('workflows')!r}"
            )
        if wf_run.get("types") != ["completed"]:
            problems.append(
                f"release.yml workflow_run.types must be ['completed'], got {wf_run.get('types')!r}"
            )
        if wf_run.get("branches") != ["main"]:
            problems.append(
                f"release.yml workflow_run.branches must be ['main'], got {wf_run.get('branches')!r}"
            )

    jobs = release_doc.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        return [*problems, "release.yml must define jobs"]

    release_jobs = [
        (name, job)
        for name, job in jobs.items()
        if isinstance(job, dict) and "semantic-release.yml" in str(job.get("uses", ""))
    ]
    if not release_jobs:
        problems.append("release.yml must define a job calling semantic-release.yml")

    for name, job in release_jobs:
        cond = job.get("if")
        if not isinstance(cond, str) or not cond.strip():
            problems.append(f"release job {name!r} is missing an if: gate")
            continue
        try:
            problems.extend(_evaluate_release_gate_cases(cond))
        except ValueError as exc:
            problems.append(f"release job {name!r} has unparseable if: ({exc})")

    return problems


def test_release_workflow_is_gated_on_passing_test_for_same_commit():
    problems = release_gate_problems(
        RELEASE.read_text(encoding="utf-8"),
        TEST.read_text(encoding="utf-8"),
    )
    assert problems == [], "\n".join(problems)


def test_ungated_push_release_workflow_fails():
    problems = release_gate_problems(
        UNGATED_PUSH_RELEASE,
        TEST.read_text(encoding="utf-8"),
    )
    assert problems, "expected ungated push release workflow to fail"


@pytest.mark.parametrize(
    "removed_clause",
    [
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.head_sha == github.sha",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
    ],
)
def test_weakening_any_release_gate_clause_fails(removed_clause: str):
    text = RELEASE.read_text(encoding="utf-8")
    assert removed_clause in text
    mutated = text.replace(removed_clause, "true")
    problems = release_gate_problems(mutated, TEST.read_text(encoding="utf-8"))
    assert problems, f"expected replacing {removed_clause!r} with 'true' to fail"


def test_renaming_test_workflow_without_updating_release_fails():
    test_text = TEST.read_text(encoding="utf-8").replace("name: Test\n", "name: CI\n", 1)
    problems = release_gate_problems(RELEASE.read_text(encoding="utf-8"), test_text)
    assert any("workflow_run.workflows" in p for p in problems), problems
