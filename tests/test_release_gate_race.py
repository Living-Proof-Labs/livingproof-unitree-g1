# SPDX-License-Identifier: Apache-2.0
"""Release gate, race model and workflow contract for the workflow_run release.

Modelled on booster-t1's test_release_gate_race.py (booster-t1#63), adapted for
livingproof-unitree-g1#73 (issue #69). Under `workflow_run`, GITHUB_SHA is the
*default-branch tip* when the Test run completes, not the commit Test ran on
(GitHub docs, "Events that trigger workflows", workflow_run: GITHUB_SHA = "Last
commit on default branch"). The release `if:` compares the two, so a Test run
for a commit that is no longer the tip is skipped, and the Test run for the
newer tip releases everything since the last tag in one version.

Offline:
- every workflow parses and the Test workflow keeps the push-on-main run that the gate needs;
- the privileged `workflow_run` workflow stays a single call to the shared release workflow
  (no steps, no secrets, no cancel-in-progress, minimal permissions);
- the real `if:` is evaluated over a timeline of pushes and Test completions.
LIVINGPROOF_NETWORK=1 also checks the pinned shared workflow still checks out GITHUB_SHA
and still refuses to run when HEAD is not GITHUB_SHA, which is what makes "release the
commit that Test passed on" hold without an explicit `ref:`.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
TEST = WORKFLOWS / "test.yml"
GUIDELINES = "Living-Proof-Labs/collaboration-guidelines"
REPO = "Living-Proof-Labs/livingproof-unitree-g1"


def _load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name} did not parse as a mapping"
    return doc


def _triggers(doc: dict) -> dict:
    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    triggers = doc["on"] if "on" in doc else doc[True]
    assert isinstance(triggers, dict)
    return triggers


# --- a second, independent evaluator for the release job's `if:` -----------------------

_TOKEN = re.compile(r"'(?:[^']|'')*'|[A-Za-z_][\w.-]*|&&|\|\||==|!=|[()!]|\s+")


def gate_allows(expr: str, context: dict) -> bool:
    """Evaluate a restricted GitHub `if:` expression by translating it to Python."""
    text = expr.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()

    def lookup(path: str) -> object:
        cur: object = context
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return ""  # GitHub: a missing property is null, which == '' is true for
            cur = cur[part]
        return cur

    out: list[str] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        assert match, f"unsupported expression syntax near {text[pos:]!r}"
        tok = match.group(0)
        pos = match.end()
        if tok.isspace():
            continue
        if tok.startswith("'"):
            out.append(repr(tok[1:-1].replace("''", "'")))
        elif tok == "&&":
            out.append(" and ")
        elif tok == "||":
            out.append(" or ")
        elif tok == "!":
            out.append(" not ")
        elif tok in ("(", ")", "==", "!="):
            out.append(tok)
        elif tok in ("true", "false"):
            out.append(tok.capitalize())
        else:
            assert not text[pos:].lstrip().startswith("("), (
                f"function calls are not allowed: {tok}()"
            )
            out.append(f"lookup({tok!r})")
    return bool(eval("".join(out), {"__builtins__": {}}, {"lookup": lookup}))  # noqa: S307


def release_if() -> str:
    jobs = _load(RELEASE)["jobs"]
    (job,) = jobs.values()
    cond = job.get("if")
    assert isinstance(cond, str) and cond.strip(), "the release job needs an explicit if:"
    return cond


def completion(*, tip: str, tested: str, conclusion: str = "success", **overrides: str) -> dict:
    """The github context of the Release run that a completed Test run starts."""
    run = {
        "event": "push",
        "head_branch": "main",
        "head_repository": {"full_name": REPO},
        "conclusion": conclusion,
        "head_sha": tested,
    }
    run.update(overrides)
    return {
        "github": {
            "event_name": "workflow_run",
            "ref": "refs/heads/main",
            "sha": tip,  # workflow_run: last commit on the default branch
            "repository": REPO,
            "event": {"workflow_run": run},
        }
    }


# --- timeline model: which Test completions release, and what each release contains ---------


def simulate(events: list[tuple]) -> list[tuple[str, list[str]]]:
    """Replay pushes and Test completions. Returns (released commit, commits in the version)."""
    expr = release_if()
    pushed: list[str] = []
    released_upto = 0
    releases: list[tuple[str, list[str]]] = []
    for event in events:
        if event[0] == "push":
            pushed.append(event[1])
            continue
        _, tested, conclusion = event
        tip = pushed[-1]
        if gate_allows(expr, completion(tip=tip, tested=tested, conclusion=conclusion)):
            # The shared workflow checks out github.sha (== tip == tested) and
            # semantic-release releases everything since the last tag.
            assert tested == tip
            releases.append((tip, pushed[released_upto:]))
            released_upto = len(pushed)
    return releases


@pytest.mark.parametrize(
    ("name", "events", "expected"),
    [
        (
            "main moves while Test runs: the older run skips, the newer run batches both",
            [("push", "A"), ("push", "B"), ("done", "A", "success"), ("done", "B", "success")],
            [("B", ["A", "B"])],
        ),
        (
            "the newer Test finishes first: it releases both, the older one is then skipped",
            [("push", "A"), ("push", "B"), ("done", "B", "success"), ("done", "A", "success")],
            [("B", ["A", "B"])],
        ),
        (
            "Test finishes before the next push: two releases, one per commit",
            [("push", "A"), ("done", "A", "success"), ("push", "B"), ("done", "B", "success")],
            [("A", ["A"]), ("B", ["B"])],
        ),
        (
            "a failed Test on the tip never releases",
            [("push", "A"), ("done", "A", "failure")],
            [],
        ),
        (
            "the newer Test fails: nothing releases until a later green run batches all three",
            [
                ("push", "A"),
                ("push", "B"),
                ("done", "A", "success"),
                ("done", "B", "failure"),
                ("push", "C"),
                ("done", "C", "success"),
            ],
            [("C", ["A", "B", "C"])],
        ),
        (
            "a cancelled or skipped Test never releases",
            [("push", "A"), ("done", "A", "cancelled"), ("done", "A", "skipped")],
            [],
        ),
        (
            "a re-run of an old green Test cannot release once main moved on",
            [("push", "A"), ("done", "A", "success"), ("push", "B"), ("done", "A", "success")],
            [("A", ["A"])],
        ),
    ],
)
def test_release_race_model(name: str, events: list[tuple], expected: list) -> None:
    assert simulate(events) == expected, name


@pytest.mark.parametrize(
    "overrides",
    [
        {"event": "pull_request"},  # a fork or branch named main
        {"event": "merge_group", "head_branch": "gh-readonly-queue/main/pr-1-abc"},
        {"event": "workflow_dispatch"},
        {"head_branch": "feature"},
        {"head_repository": {"full_name": "someone-else/livingproof-unitree-g1"}},
    ],
)
def test_only_same_repository_push_runs_on_main_can_release(overrides: dict) -> None:
    ctx = completion(tip="a" * 40, tested="a" * 40, **overrides)
    assert not gate_allows(release_if(), ctx)
    assert gate_allows(release_if(), completion(tip="a" * 40, tested="a" * 40))


# --- workflow contract ---------------------------------------------------------------------


def test_every_workflow_parses_and_declares_name_triggers_and_jobs() -> None:
    files = sorted(WORKFLOWS.glob("*.y*ml"))
    assert files
    names = []
    for path in files:
        doc = _load(path)
        assert isinstance(doc.get("name"), str) and doc["name"], path.name
        assert _triggers(doc), path.name
        assert isinstance(doc.get("jobs"), dict) and doc["jobs"], path.name
        names.append(doc["name"])
    assert len(names) == len(set(names)), f"duplicate workflow names: {names}"


def test_test_workflow_still_runs_on_push_to_main() -> None:
    """The gate needs a Test run with event == push on main, or nothing ever releases."""
    triggers = _triggers(_load(TEST))
    assert triggers["push"] == {"branches": ["main"]}
    assert "workflow_run" not in triggers


def test_release_workflow_is_one_privileged_call_and_runs_no_repository_code() -> None:
    doc = _load(RELEASE)
    assert doc["permissions"] == {"contents": "read"}
    jobs = doc["jobs"]
    assert len(jobs) == 1, "a workflow_run workflow holds a write token: keep it to one job"
    ((name, job),) = jobs.items()
    assert "steps" not in job, "no steps: nothing from the repository runs with contents: write"
    assert job["permissions"] == {"contents": "write"}
    assert job["uses"].startswith(f"{GUIDELINES}/.github/workflows/semantic-release.yml@v")
    assert not {"secrets", "with", "env", "continue-on-error", "needs"} & set(job), name
    # A new Test run must never cancel a release that is already tagging.
    assert "concurrency" not in doc
    assert "cancel-in-progress" not in RELEASE.read_text(encoding="utf-8")


def test_release_workflow_listens_only_to_the_test_workflow_by_its_exact_name() -> None:
    trigger = _triggers(_load(RELEASE))
    assert set(trigger) == {"workflow_run"}
    assert trigger["workflow_run"] == {
        "workflows": [_load(TEST)["name"]],
        "types": ["completed"],
        "branches": ["main"],
    }


# --- the shared workflow (network) ---------------------------------------------------------


def _pinned_ref() -> str:
    match = re.search(
        rf"uses:\s*{re.escape(GUIDELINES)}/\.github/workflows/semantic-release\.yml@(\S+)",
        RELEASE.read_text(encoding="utf-8"),
    )
    assert match
    return match.group(1)


def _raw(path: str) -> str:
    url = f"https://raw.githubusercontent.com/{GUIDELINES}/{_pinned_ref()}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
def test_pinned_shared_workflow_releases_exactly_github_sha() -> None:
    """With head_sha == github.sha required by the gate, these two facts mean: tag the tested commit.

    A Renovate bump of the guidelines tag that changes either one must be reviewed together
    with this gate.
    """
    workflow = yaml.safe_load(_raw(".github/workflows/semantic-release.yml"))
    steps = workflow["jobs"]["release"]["steps"]
    (checkout,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert "ref" not in checkout.get("with", {}), (
        "checkout must default to GITHUB_REF at GITHUB_SHA"
    )
    assert checkout["with"].get("persist-credentials") is False
    assert workflow["jobs"]["release"]["if"] == "github.ref == 'refs/heads/main'"
    runner = _raw(".github/actions/semantic-release/run.mjs")
    assert "process.env.GITHUB_SHA" in runner
    assert "Checked-out HEAD must match the triggering commit" in runner
    assert "git fetch" not in runner and "git pull" not in runner
