# SPDX-License-Identifier: Apache-2.0
"""Release publication must wait for a passing Test run on the same commit."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
TEST = WORKFLOWS / "test.yml"
RELEASE_CALLEE_PREFIX = (
    "Living-Proof-Labs/collaboration-guidelines/.github/workflows/semantic-release.yml@"
)

TOKEN_RE = re.compile(r"\s*(?:(\&\&|\|\||==|!=|[()!])|('(?:[^']|'')*')|([A-Za-z_][A-Za-z0-9_.-]*))")


def _triggers(document: dict) -> dict:
    # PyYAML implements YAML 1.1, which parses bare `on` as boolean True.
    triggers = document.get("on") if "on" in document else document.get(True)
    assert isinstance(triggers, dict), "workflow must declare a mapping of triggers"
    return triggers


def _eval_if(expr: str, context: dict[str, str]) -> bool:
    """Evaluate a restricted GitHub Actions boolean `if:` expression."""
    cleaned = expr.strip()
    if cleaned.startswith("${{") and cleaned.endswith("}}"):
        cleaned = cleaned[3:-2].strip()

    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(cleaned):
        match = TOKEN_RE.match(cleaned, pos)
        if not match or match.end() == pos:
            raise AssertionError(f"unsupported token in if expression near: {cleaned[pos:]!r}")
        op, quoted, ident = match.groups()
        if op is not None:
            tokens.append(("OP", op))
        elif quoted is not None:
            tokens.append(("STR", quoted[1:-1].replace("''", "'")))
        else:
            tokens.append(("IDENT", ident))
        pos = match.end()

    index = 0

    def peek() -> tuple[str, str] | None:
        return tokens[index] if index < len(tokens) else None

    def consume(expected: tuple[str, str] | None = None) -> tuple[str, str]:
        nonlocal index
        tok = peek()
        if tok is None or (expected is not None and tok != expected):
            raise AssertionError(f"unexpected token {tok!r}, expected {expected!r}")
        index += 1
        return tok

    def parse_or() -> bool:
        value = parse_and()
        while peek() == ("OP", "||"):
            consume()
            rhs = parse_and()
            value = value or rhs
        return value

    def parse_and() -> bool:
        value = parse_not()
        while peek() == ("OP", "&&"):
            consume()
            rhs = parse_not()
            value = value and rhs
        return value

    def parse_not() -> bool:
        if peek() == ("OP", "!"):
            consume()
            return not parse_not()
        return parse_cmp()

    def parse_atom() -> object:
        tok = peek()
        if tok == ("OP", "("):
            consume()
            value = parse_or()
            consume(("OP", ")"))
            return value
        kind, raw = consume()
        if kind == "STR":
            return raw
        if kind == "IDENT":
            if peek() == ("OP", "("):
                raise AssertionError(f"disallowed function call in release gate if: {raw}()")
            if raw == "true":
                return True
            if raw == "false":
                return False
            if raw not in context:
                raise AssertionError(f"unknown identifier in release gate if: {raw}")
            return context[raw]
        raise AssertionError(f"unexpected token {kind}:{raw!r}")

    def parse_cmp() -> bool:
        lhs = parse_atom()
        tok = peek()
        if tok in (("OP", "=="), ("OP", "!=")):
            _, op = consume()
            rhs = parse_atom()
            return (lhs == rhs) if op == "==" else (lhs != rhs)
        return bool(lhs)

    result = parse_or()
    if index != len(tokens):
        raise AssertionError(f"trailing tokens in if expression: {tokens[index:]}")
    return bool(result)


def assert_release_gated_on_test(release_doc: dict, test_doc: dict) -> None:
    triggers = _triggers(release_doc)
    assert set(triggers) == {"push"}, set(triggers)
    assert triggers["push"] == {"branches": ["main"]}
    assert release_doc.get("permissions") == {"contents": "read"}

    test_triggers = _triggers(test_doc)
    assert {"pull_request", "merge_group", "push", "workflow_call"} <= set(test_triggers)
    test_jobs = test_doc.get("jobs") or {}
    gate_job = test_jobs.get("verification-gate")
    assert isinstance(gate_job, dict)
    assert gate_job.get("needs") in ("test", ["test"])
    assert gate_job.get("if") == "always()"

    jobs = release_doc.get("jobs") or {}
    assert isinstance(jobs, dict)

    release_jobs = [
        (name, job)
        for name, job in jobs.items()
        if isinstance(job, dict) and str(job.get("uses", "")).startswith(RELEASE_CALLEE_PREFIX)
    ]
    assert len(release_jobs) == 1, f"expected one semantic-release job, found {release_jobs!r}"
    release_name, release_job = release_jobs[0]
    assert not release_job.get("continue-on-error", False)

    test_callers = [
        (name, job)
        for name, job in jobs.items()
        if isinstance(job, dict) and job.get("uses") == "./.github/workflows/test.yml"
    ]
    assert len(test_callers) == 1, f"expected one test.yml caller job, found {test_callers!r}"
    test_job_name, test_job = test_callers[0]
    assert not test_job.get("continue-on-error", False)
    assert "if" not in test_job, "the gating test job must run unconditionally on push to main"
    if "permissions" in test_job:
        assert test_job["permissions"] == {"contents": "read"}

    needs = release_job.get("needs")
    if isinstance(needs, str):
        needs_list = [needs]
    elif isinstance(needs, list):
        needs_list = list(needs)
    else:
        needs_list = []
    assert test_job_name in needs_list, (
        f"release job {release_name!r} must depend on {test_job_name!r} via needs"
    )

    if_expr = release_job.get("if")
    assert isinstance(if_expr, str) and if_expr.strip(), "release job must declare an explicit if:"
    lowered = if_expr.lower()
    for banned in ("always(", "failure(", "cancelled("):
        assert banned not in lowered, f"release job condition must not contain {banned}"

    result_key = f"needs.{test_job_name}.result"
    for result in ("success", "failure", "cancelled", "skipped", "neutral", "", "unknown"):
        for event_name in ("push", "pull_request", "merge_group", "workflow_dispatch"):
            for ref in ("refs/heads/main", "refs/heads/feature", "refs/tags/v1.0.0"):
                allowed = _eval_if(
                    if_expr,
                    {
                        result_key: result,
                        "github.event_name": event_name,
                        "github.ref": ref,
                    },
                )
                expected = result == "success" and event_name == "push" and ref == "refs/heads/main"
                assert allowed == expected, (
                    f"if={if_expr!r} evaluated to {allowed!r} for "
                    f"({result!r}, {event_name!r}, {ref!r}), expected {expected!r}"
                )


def test_release_is_gated_on_test_for_the_same_commit() -> None:
    release_doc = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    test_doc = yaml.safe_load(TEST.read_text(encoding="utf-8"))
    assert_release_gated_on_test(release_doc, test_doc)


def test_release_gate_rejects_ungated_and_permissive_mutations() -> None:
    base_release = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    base_test = yaml.safe_load(TEST.read_text(encoding="utf-8"))

    no_needs = copy.deepcopy(base_release)
    no_needs["jobs"]["release"].pop("needs")
    with pytest.raises(AssertionError):
        assert_release_gated_on_test(no_needs, base_test)

    no_result_check = copy.deepcopy(base_release)
    no_result_check["jobs"]["release"]["if"] = (
        "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    with pytest.raises(AssertionError):
        assert_release_gated_on_test(no_result_check, base_test)

    always_runs = copy.deepcopy(base_release)
    always_runs["jobs"]["release"]["if"] = (
        "always() && github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    with pytest.raises(AssertionError):
        assert_release_gated_on_test(always_runs, base_test)

    continue_on_error = copy.deepcopy(base_release)
    continue_on_error["jobs"]["test"]["continue-on-error"] = True
    with pytest.raises(AssertionError):
        assert_release_gated_on_test(continue_on_error, base_test)

    no_test_caller = copy.deepcopy(base_release)
    no_test_caller["jobs"].pop("test")
    with pytest.raises(AssertionError):
        assert_release_gated_on_test(no_test_caller, base_test)

    no_workflow_call = copy.deepcopy(base_test)
    _triggers(no_workflow_call).pop("workflow_call")
    with pytest.raises(AssertionError):
        assert_release_gated_on_test(base_release, no_workflow_call)
