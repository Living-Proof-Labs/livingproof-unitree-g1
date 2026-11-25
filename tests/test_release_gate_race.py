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
- only the shared release call holds contents: write; the dependent manual delivery
  has its own exact read/Pages permissions and cannot run inline repository code;
- the real `if:` is evaluated over a timeline of pushes and Test completions.
LIVINGPROOF_NETWORK=1 also checks the pinned shared workflow still checks out GITHUB_SHA
and that plan.mjs still refuses when HEAD is not the expected commit. run.mjs passes
GITHUB_SHA as that expected commit for a non-candidate release. A fetch there may
update only the same tag ref and must throw unless that tag is the planned commit,
which is what makes "release the commit that Test passed on" hold without an
explicit `ref:`. The same three checks run offline on fixture copies of those scripts.
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
    job = jobs["release"]
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


def _assert_release_job_boundaries(doc):
    assert doc["permissions"] == {"contents": "read"}
    jobs = doc["jobs"]
    assert set(jobs) == {"release", "deploy-manual"}
    job = jobs["release"]
    assert "steps" not in job, "repository code must not run with contents: write"
    assert job["permissions"] == {"contents": "write"}
    assert job["uses"].startswith(f"{GUIDELINES}/.github/workflows/semantic-release.yml@v")
    assert not {"secrets", "with", "env", "continue-on-error", "needs"} & set(job)
    delivery = jobs["deploy-manual"]
    assert delivery == {
        "needs": "release",
        "if": "needs.release.outputs.published == 'true'",
        "permissions": {"contents": "read", "pages": "write", "id-token": "write"},
        "uses": "./.github/workflows/manual.yml",
        "with": {
            "release_tag": "${{ needs.release.outputs.git-tag }}",
            "release_version": "${{ needs.release.outputs.version }}",
        },
    }
    assert "concurrency" not in doc


def test_release_workflow_keeps_one_contents_writer_and_gated_manual_delivery() -> None:
    _assert_release_job_boundaries(_load(RELEASE))
    assert "cancel-in-progress" not in RELEASE.read_text(encoding="utf-8")
    manual = _load(WORKFLOWS / "manual.yml")
    assert manual["permissions"] == {"contents": "read"}
    assert set(manual["jobs"]) == {"build", "deploy"}
    assert "permissions" not in manual["jobs"]["build"]
    deploy = manual["jobs"]["deploy"]
    assert deploy["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    for step in deploy["steps"]:
        assert "checkout@" not in step.get("uses", "")
        assert "tools/" not in step.get("run", "")
    assert "GH_TOKEN" not in manual["jobs"]["build"].get("env", {})


@pytest.mark.parametrize(
    "mutation",
    ["extra-job", "writer-steps", "delivery-write", "ungated", "no-dependency", "delivery-steps"],
)
def test_release_job_boundary_mutations_are_rejected(mutation):
    doc = _load(RELEASE)
    if mutation == "extra-job":
        doc["jobs"]["extra"] = {"run": "unreviewed"}
    elif mutation == "writer-steps":
        doc["jobs"]["release"]["steps"] = [{"run": "python untrusted.py"}]
    elif mutation == "delivery-write":
        doc["jobs"]["deploy-manual"]["permissions"]["contents"] = "write"
    elif mutation == "ungated":
        doc["jobs"]["deploy-manual"]["if"] = "always()"
    elif mutation == "no-dependency":
        del doc["jobs"]["deploy-manual"]["needs"]
    else:
        doc["jobs"]["deploy-manual"]["steps"] = [{"run": "python untrusted.py"}]
    with pytest.raises(AssertionError):
        _assert_release_job_boundaries(doc)


@pytest.mark.parametrize("published", ["true", "false", "", "missing", "failed", "skipped"])
def test_manual_delivery_requires_successfully_published_release(published):
    job = _load(RELEASE)["jobs"]["deploy-manual"]
    assert job["needs"] == "release"
    assert gate_allows(
        job["if"], {"needs": {"release": {"outputs": {"published": published}}}}
    ) == (published == "true")


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


_HEAD_MISMATCH = "Checked-out HEAD must match the triggering commit"


def _strip_js_comments(source: str) -> str:
    """Drop // and /* */ comments. Strings, including the refusal text, stay."""
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch in "'\"`":
            start = i
            quote = ch
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                i += 1
            out.append(source[start:i])
            continue
        if ch == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and source[i : i + 2] != "*/":
                i += 1
            i = min(n, i + 2)
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _matching_closer(source: str, open_at: int) -> int:
    """Index of the bracket that closes source[open_at]. Brackets inside strings do not count."""
    closer = {"(": ")", "{": "}", "[": "]"}[source[open_at]]
    opener = source[open_at]
    depth = 0
    i = open_at
    n = len(source)
    quote: str | None = None
    while i < n:
        ch = source[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            i += 1
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise AssertionError("unbalanced bracket in pinned release script")


def _plan_version_body(source: str) -> str:
    code = _strip_js_comments(source)
    match = re.search(r"\bfunction\s+planVersion\s*\(", code)
    assert match, "plan.mjs must define function planVersion"
    open_paren = match.end() - 1
    close_paren = _matching_closer(code, open_paren)
    open_brace = code.find("{", close_paren)
    assert open_brace != -1, "planVersion is missing a body"
    close_brace = _matching_closer(code, open_brace)
    return code[open_brace + 1 : close_brace]


def _assignment_reads_rev_parse_head(text: str, expr: str, end: int, seen: set[str]) -> bool:
    """True when expr is git rev-parse HEAD, or a name still bound to that call."""
    expr = _strip_wrapping_parens(expr)
    if re.search(r"""['"]rev-parse['"]\s*,\s*['"]HEAD['"]""", expr):
        return True
    if re.fullmatch(r"[A-Za-z_$][\w$]*", expr) and expr not in seen:
        bound = _initializer_before(text, expr, end)
        if bound is None:
            return False
        return _assignment_reads_rev_parse_head(text, bound, end, seen | {expr})
    return False


def _assert_plan_refuses_head_mismatch(source: str) -> None:
    """planVersion throws the HEAD refusal when rev-parsed HEAD !== expected.

    The name in the comparison has to be the one still bound to ``git rev-parse HEAD``.
    ``commit = expected`` keeps the inequality and the message, and does not count.
    Whitespace and either quote style may change.
    """
    body = re.sub(r"\s+", " ", _plan_version_body(source))
    refusal = (
        r"throw\s+new\s+Error\(\s*(?P<q>['\"`])" + re.escape(_HEAD_MISMATCH) + r"(?P=q)\s*,?\s*\)"
    )
    guard = re.search(
        r"(?:\b(?P<left>[A-Za-z_$][\w$]*)\s*!==\s*expected\b"
        r"|\bexpected\s*!==\s*(?P<right>[A-Za-z_$][\w$]*))"
        r".{0,600}?" + refusal,
        body,
    )
    assert guard, (
        "plan.mjs planVersion must refuse when the checked-out commit is not "
        f"expected ({_HEAD_MISMATCH})"
    )
    name = guard.group("left") or guard.group("right")
    bound = _initializer_before(body, name, guard.start())
    assert bound is not None and _assignment_reads_rev_parse_head(
        body, bound, guard.start(), set()
    ), "plan.mjs must compare expected with the commit from git rev-parse HEAD"


def _index_in_string(source: str, index: int) -> bool:
    quote: str | None = None
    i = 0
    while i < index:
        ch = source[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
        i += 1
    return quote is not None


def _strip_wrapping_parens(expr: str) -> str:
    text = expr.strip()
    while text.startswith("("):
        try:
            closes_all = _matching_closer(text, 0) == len(text) - 1
        except AssertionError:
            break
        if not closes_all:
            break
        text = text[1:-1].strip()
    return text


def _read_expression(source: str, start: int) -> str:
    """Read one JS expression. Newlines are fine when the next token continues it."""
    i = start
    n = len(source)
    depth = 0
    quote: str | None = None
    while i < n:
        ch = source[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            i += 1
            continue
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            if depth == 0:
                break
            depth -= 1
        elif ch in ";," and depth == 0:
            break
        elif ch == "\n" and depth == 0:
            nxt = i + 1
            while nxt < n and source[nxt] in " \t\r\n":
                nxt += 1
            if nxt >= n or source[nxt] not in "?:+-*/%&|^<>=.":
                break
        i += 1
    return source[start:i].strip()


def _initializer_before(source: str, name: str, pos: int) -> str | None:
    pattern = re.compile(rf"(?<![\w$.])(?:(?:const|let|var)\s+)?{re.escape(name)}\s*=(?!=)")
    found: str | None = None
    for match in pattern.finditer(source, 0, pos):
        if _index_in_string(source, match.start()):
            continue
        found = _read_expression(source, match.end())
    return found


def _split_ternary(expr: str) -> tuple[str, str, str] | None:
    """Split `cond ? yes : no` at top level. `??` and `?.` are not ternaries."""
    depth = 0
    nested = 0
    question: int | None = None
    colon: int | None = None
    quote: str | None = None
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0 and ch == "?" and i + 1 < n and expr[i + 1] in ".?":
            i += 2
            continue
        elif depth == 0 and ch == "?":
            if question is None and nested == 0:
                question = i
            nested += 1
        elif depth == 0 and ch == ":" and nested:
            nested -= 1
            if nested == 0 and colon is None:
                colon = i
                break
        i += 1
    if question is None or colon is None:
        return None
    return expr[:question], expr[question + 1 : colon], expr[colon + 1 :]


def _false_candidate_branch(expr: str) -> str | None:
    """The expression used when `candidate` is false, if expr selects on it."""
    parts = _split_ternary(_strip_wrapping_parens(expr))
    if parts is None:
        return None
    cond, yes, no = (part.strip() for part in parts)
    condition = re.sub(r"\s+", "", _strip_wrapping_parens(cond))
    if condition in {"candidate", "candidate===true", "candidate==true"}:
        return no
    if condition in {
        "!candidate",
        "candidate===false",
        "candidate==false",
        "candidate!==true",
        "candidate!=true",
    }:
        return yes
    return None


def _if_else_non_candidate(source: str, name: str, pos: int) -> str | None:
    """Value assigned to name when candidate is false, for an if/else before pos.

    A ternary is handled separately. This is the if/else form: the non-candidate arm
    may itself be a name bound to process.env.GITHUB_SHA.
    """
    ident = re.escape(name)
    binding = rf"(?:(?:const|let|var)\s+)?{ident}\s*=\s*"
    forms = (
        (
            rf"if\s*\(\s*candidate\s*\)\s*\{{?\s*{binding}(?P<yes>[^;]+);\s*\}}?\s*"
            rf"else\s*\{{?\s*{binding}(?P<no>[^;]+);"
        ),
        (
            rf"if\s*\(\s*!\s*candidate\s*\)\s*\{{?\s*{binding}(?P<no>[^;]+);\s*\}}?\s*"
            rf"else\s*\{{?\s*{binding}(?P<yes>[^;]+);"
        ),
    )
    found: str | None = None
    window = source[:pos]
    for pattern in forms:
        for match in re.finditer(pattern, window):
            found = match.group("no").strip()
    return found


def _resolve_non_candidate(source: str, expr: str, pos: int, seen: set[str]) -> str | None:
    text = _strip_wrapping_parens(expr)
    if not text:
        return None
    branch = _false_candidate_branch(text)
    if branch is not None:
        return _resolve_non_candidate(source, branch, pos, seen)
    if re.fullmatch(r"[A-Za-z_$][\w$]*", text):
        if text in seen:
            return None
        branched = _if_else_non_candidate(source, text, pos)
        if branched is not None:
            return _resolve_non_candidate(source, branched, pos, seen | {text})
        bound = _initializer_before(source, text, pos)
        if bound is None:
            return None
        return _resolve_non_candidate(source, bound, pos, seen | {text})
    return text


def _is_github_sha(expr: str | None) -> bool:
    if not expr:
        return False
    compact = re.sub(r"\s+", "", expr)
    if compact == "process.env.GITHUB_SHA":
        return True
    return re.fullmatch(r"""process\.env\[['"]GITHUB_SHA['"]\]""", compact) is not None


def _split_args(arglist: str) -> list[str]:
    args: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(arglist):
        ch = arglist[i]
        if quote:
            buf.append(ch)
            if ch == "\\":
                if i + 1 < len(arglist):
                    buf.append(arglist[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            buf.append(ch)
        elif ch in "({[":
            depth += 1
            buf.append(ch)
        elif ch in ")}]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    return args


def _plan_version_callees(source: str) -> set[str]:
    """Local names bound to planVersion by a static import from plan.mjs."""
    names: set[str] = set()
    for match in re.finditer(
        r"import\s*\{([^}]*)\}\s*from\s*(['\"])([^'\"]+)\2",
        source,
    ):
        if not match.group(3).replace("\\", "/").endswith("plan.mjs"):
            continue
        for part in match.group(1).split(","):
            piece = part.strip()
            if not piece:
                continue
            bits = re.split(r"\s+as\s+", piece)
            if bits[0].strip() == "planVersion":
                names.add(bits[-1].strip())
    for match in re.finditer(
        r"import\s*\*\s*as\s+(\w+)\s+from\s*(['\"])([^'\"]+)\2",
        source,
    ):
        if match.group(3).replace("\\", "/").endswith("plan.mjs"):
            names.add(f"{match.group(1)}.planVersion")
    return names


def _call_sites(source: str, callee: str) -> list[tuple[int, str]]:
    pattern = re.compile(rf"(?<![\w$.]){re.escape(callee)}\s*\(")
    sites: list[tuple[int, str]] = []
    for match in pattern.finditer(source):
        if re.search(r"\bfunction\s*$", source[: match.start()]):
            continue
        open_paren = match.end() - 1
        close_paren = _matching_closer(source, open_paren)
        sites.append((open_paren, source[open_paren + 1 : close_paren]))
    return sites


def _passes_github_sha_for_non_candidate(source: str, pos: int, arglist: str) -> bool:
    args = _split_args(arglist)
    if len(args) < 2:
        return False
    resolved = _resolve_non_candidate(source, args[1], pos, set())
    return _is_github_sha(resolved)


def _assert_run_passes_github_sha_for_non_candidate(source: str) -> None:
    """Non-candidate runs must hand process.env.GITHUB_SHA to planVersion as expected.

    The handoff may be a direct argument or a binding (`expected = candidate ? ... :
    process.env.GITHUB_SHA`) passed into an imported or called planVersion. Formatting
    may wrap; a GITHUB_SHA used for something else does not count.
    """
    code = _strip_js_comments(source)
    imported = _plan_version_callees(code)
    imported_calls: list[tuple[int, str]] = []
    for callee in imported:
        imported_calls.extend(_call_sites(code, callee))
    bare_calls = _call_sites(code, "planVersion")
    assert imported or bare_calls, "run.mjs must import or call planVersion"
    calls = imported_calls or bare_calls
    assert any(
        _passes_github_sha_for_non_candidate(code, pos, arglist) for pos, arglist in calls
    ), "run.mjs must pass process.env.GITHUB_SHA to planVersion as expected for a non-candidate run"


_HEAD_MOVING_GIT = frozenset({"pull", "checkout", "switch", "reset", "merge", "rebase"})


def _js_static_text(token: str) -> str:
    text = token.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"`":
        return text[1:-1]
    return text


def _git_exec_argvs(source: str) -> list[tuple[int, list[str]]]:
    """(index, argv) for each exec('git', [...]) invocation, quotes removed."""
    found: list[tuple[int, list[str]]] = []
    pattern = re.compile(r"""(['"])git\1\s*,\s*\[""")
    for match in pattern.finditer(source):
        if _index_in_string(source, match.start()):
            continue
        open_bracket = match.end() - 1
        close_bracket = _matching_closer(source, open_bracket)
        elements = _split_args(source[open_bracket + 1 : close_bracket])
        found.append((match.start(), [_js_static_text(element) for element in elements]))
    return found


def _assert_tag_only_fetch(argv: list[str]) -> None:
    """A fetch may mirror one tag onto that same local tag. It must not update a branch or HEAD."""
    assert argv and argv[0] == "fetch", argv
    flags = [arg for arg in argv[1:] if arg.startswith("-")]
    positionals = [arg for arg in argv[1:] if not arg.startswith("-")]
    assert "--update-head-ok" not in flags, argv
    refspecs = [arg for arg in positionals if arg.startswith("refs/") or ":" in arg]
    others = [arg for arg in positionals if arg not in refspecs]
    assert len(refspecs) == 1 and len(others) == 1, (
        "git fetch must name a remote and one refs/tags refspec, not a branch"
    )
    spec = refspecs[0][1:] if refspecs[0].startswith("+") else refspecs[0]
    src, sep, dst = spec.partition(":")
    same_tag = re.sub(r"\s+", "", src) == re.sub(r"\s+", "", dst)
    assert (
        sep
        and src.startswith("refs/tags/")
        and dst.startswith("refs/tags/")
        and same_tag
        and "refs/heads" not in spec
        and "/HEAD" not in f"/{dst}"
        and not src.endswith("/")
    ), f"git fetch refspec must copy one tag onto itself, not {spec!r}"


def _functions(source: str) -> list[tuple[str, int, int, list[str], str]]:
    found = []
    for match in re.finditer(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", source):
        open_paren = match.end() - 1
        close_paren = _matching_closer(source, open_paren)
        params = []
        for part in _split_args(source[open_paren + 1 : close_paren]):
            params.append(re.split(r"\s*=", part.strip(), maxsplit=1)[0].strip())
        open_brace = source.find("{", close_paren)
        assert open_brace != -1, f"{match.group(1)} is missing a body"
        close_brace = _matching_closer(source, open_brace)
        found.append(
            (
                match.group(1),
                open_brace,
                close_brace,
                params,
                source[open_brace + 1 : close_brace],
            )
        )
    return found


def _enclosing_function(
    functions: list[tuple[str, int, int, list[str], str]], index: int
) -> tuple[str, int, int, list[str], str]:
    matches = [fn for fn in functions if fn[1] <= index <= fn[2]]
    assert matches, "git invocation is outside a function"
    return min(matches, key=lambda fn: fn[2] - fn[1])


def _assert_fetches_do_not_retarget_release(source: str) -> None:
    """A fetch may refresh one tag ref, then must throw unless that tag is plan.commit.

    planVersion still chooses the commit. Nothing in run.mjs may pull, check out, switch,
    reset, merge, or rebase onto another ref, and the fetch result is not what gets released.
    """
    code = _strip_js_comments(source)
    assert not re.search(
        r"\bgit\s+(?:pull|checkout|switch|reset|merge|rebase)\b",
        code,
    ), "run.mjs must not pull, check out, switch, reset, merge, or rebase"
    functions = _functions(code)
    fetches: list[int] = []
    for index, argv in _git_exec_argvs(code):
        if not argv:
            continue
        assert argv[0] not in _HEAD_MOVING_GIT, f"run.mjs must not git {argv[0]}"
        if argv[0] == "fetch":
            _assert_tag_only_fetch(argv)
            fetches.append(index)
    for shell in re.finditer(r"\bgit\s+fetch\b([^\n;]*)", code):
        _assert_tag_only_fetch(["fetch", *shell.group(1).split()])
        fetches.append(shell.start())
    assert fetches, "run.mjs must fetch one tag ref and throw unless that tag is plan.commit"
    for index in fetches:
        name, _open, _close, params, body = _enclosing_function(functions, index)
        assert len(params) >= 2 and re.fullmatch(r"[A-Za-z_$][\w$]*", params[1]), params
        expected = params[1]
        collapsed = re.sub(r"\s+", " ", body)
        compared = re.search(
            rf"(?:\b\w+\s*!==\s*{expected}\b|\b{expected}\s*!==\s*\w+\b)"
            rf".{{0,500}}?throw\s+new\s+Error\s*\(",
            collapsed,
        )
        assert compared, f"{name} must throw when the fetched tag is not {expected}"
        assert "refs/tags/" in body and re.search(r"\brev-parse\b", body), (
            f"{name} must resolve the fetched tag, not check it out"
        )
        calls = _call_sites(code, name)
        assert calls, f"{name} must run against the planned commit"
        for pos, arglist in calls:
            args = _split_args(arglist)
            assert len(args) >= 2, f"{name} must be called with the plan commit"
            passed = _strip_wrapping_parens(args[1])
            if re.sub(r"\s+", "", passed) == "plan.commit":
                continue
            bound = _initializer_before(code, passed, pos)
            assert bound is not None and re.sub(r"\s+", "", bound) == "plan.commit", (
                f"{name} must compare the tag to plan.commit, not {passed!r}"
            )


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
def test_pinned_shared_workflow_releases_exactly_github_sha() -> None:
    """With head_sha == github.sha required by the gate, these facts mean: tag the tested commit.

    Checkout stays on GITHUB_SHA. plan.mjs refuses when that HEAD is not the expected
    commit, and run.mjs passes GITHUB_SHA as expected unless the run is a candidate.
    A fetch in run.mjs may update only that same tag ref, then must throw unless the
    tag is plan.commit. The guidelines tag is the one release.yml pins. A Renovate bump
    that changes any of these facts must be reviewed together with this gate.
    """
    workflow = yaml.safe_load(_raw(".github/workflows/semantic-release.yml"))
    steps = workflow["jobs"]["release"]["steps"]
    (checkout,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert "ref" not in checkout.get("with", {}), (
        "checkout must default to GITHUB_REF at GITHUB_SHA"
    )
    assert checkout["with"].get("persist-credentials") is False
    assert workflow["jobs"]["release"]["if"] == "github.ref == 'refs/heads/main'"
    _assert_plan_refuses_head_mismatch(_raw(".github/actions/semantic-release/plan.mjs"))
    runner = _raw(".github/actions/semantic-release/run.mjs")
    _assert_run_passes_github_sha_for_non_candidate(runner)
    _assert_fetches_do_not_retarget_release(runner)


# Minimal shape of the pinned scripts. Offline tests patch `_raw` with these, so CI
# exercises the guards without LIVINGPROOF_NETWORK. The network test above uses the pin.
_PLAN_FIXTURE = """\
export async function planVersion(cwd, expected) {
  const git = (...args) => execFileSync('git', args, { cwd, encoding: 'utf8' });
  const commit = git('rev-parse', 'HEAD').trim();
  if (!shaPattern.test(expected ?? '') || commit !== expected) {
    throw new Error('Checked-out HEAD must match the triggering commit');
  }
  return { commit };
}
"""

_RUN_FIXTURE = """\
import { planVersion } from './plan.mjs';

export async function verifyTagMatchesCommit(gitTag, expectedCommit, { cwd } = {}) {
  execFileSync('git', ['fetch', 'origin', `refs/tags/${gitTag}:refs/tags/${gitTag}`, '--force'], {
    cwd,
    encoding: 'utf8',
  });
  const taggedCommit = execFileSync('git', ['rev-parse', '--verify', `refs/tags/${gitTag}^{commit}`], {
    cwd,
    encoding: 'utf8',
  }).trim();
  if (taggedCommit !== expectedCommit) {
    throw new Error(`Tag ${gitTag} resolves to ${taggedCommit}, expected ${expectedCommit}`);
  }
}

export async function run() {
  const candidate = process.env.RELEASE_CANDIDATE === 'true';
  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : process.env.GITHUB_SHA;
  const plan = await planVersion(cwd, expected, { candidate });
  const head = plan.commit;
  verifyTagMatchesCommit(plan.gitTag, head, { cwd });
}
"""

_RUN_SHA_IN_VARIABLE = _RUN_FIXTURE.replace(
    "  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : process.env.GITHUB_SHA;\n",
    "  const triggering = process.env.GITHUB_SHA;\n"
    "  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : triggering;\n",
)

_RUN_SHA_IN_ELSE = _RUN_FIXTURE.replace(
    "  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : process.env.GITHUB_SHA;\n",
    "  let expected;\n"
    "  if (!candidate) {\n"
    "    expected = process.env.GITHUB_SHA;\n"
    "  } else {\n"
    "    expected = process.env.RELEASE_EXPECTED_COMMIT;\n"
    "  }\n",
)


def _check_release_scripts(plan: str, runner: str) -> None:
    _assert_plan_refuses_head_mismatch(plan)
    _assert_run_passes_github_sha_for_non_candidate(runner)
    _assert_fetches_do_not_retarget_release(runner)


def _patch_release_scripts(monkeypatch: pytest.MonkeyPatch, plan: str, runner: str) -> None:
    """Serve fixture text from the same fetch the network test uses."""

    def raw(path: str) -> str:
        if path.endswith("/plan.mjs"):
            return plan
        if path.endswith("/run.mjs"):
            return runner
        raise AssertionError(f"offline fixture does not fetch {path}")

    monkeypatch.setattr(f"{__name__}._raw", raw)


def _fetched_release_scripts() -> tuple[str, str]:
    return (
        _raw(".github/actions/semantic-release/plan.mjs"),
        _raw(".github/actions/semantic-release/run.mjs"),
    )


def test_offline_fixture_accepts_the_current_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_release_scripts(monkeypatch, _PLAN_FIXTURE, _RUN_FIXTURE)
    _check_release_scripts(*_fetched_release_scripts())


def test_offline_fixture_accepts_github_sha_held_in_a_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_release_scripts(monkeypatch, _PLAN_FIXTURE, _RUN_SHA_IN_VARIABLE)
    _check_release_scripts(*_fetched_release_scripts())


def test_offline_fixture_accepts_github_sha_assigned_in_the_non_candidate_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_release_scripts(monkeypatch, _PLAN_FIXTURE, _RUN_SHA_IN_ELSE)
    _check_release_scripts(*_fetched_release_scripts())


@pytest.mark.parametrize(
    ("plan", "runner", "message"),
    [
        pytest.param(
            _PLAN_FIXTURE.replace(
                "  if (!shaPattern.test(expected ?? '') || commit !== expected) {\n"
                "    throw new Error('Checked-out HEAD must match the triggering commit');\n"
                "  }\n",
                "",
            ),
            _RUN_FIXTURE,
            "checked-out commit is not",
            id="missing-plan-guard",
        ),
        pytest.param(
            _PLAN_FIXTURE.replace(
                "const commit = git('rev-parse', 'HEAD').trim();",
                "const commit = expected;",
            ),
            _RUN_FIXTURE,
            "git rev-parse HEAD",
            id="commit-copied-from-expected",
        ),
        pytest.param(
            _PLAN_FIXTURE,
            _RUN_FIXTURE.replace("process.env.GITHUB_SHA", "process.env.GITHUB_REF"),
            "GITHUB_SHA",
            id="missing-github-sha",
        ),
        pytest.param(
            _PLAN_FIXTURE,
            _RUN_SHA_IN_VARIABLE.replace(
                "const triggering = process.env.GITHUB_SHA;",
                "const triggering = process.env.GITHUB_REF;",
            ),
            "GITHUB_SHA",
            id="variable-is-not-github-sha",
        ),
        pytest.param(
            _PLAN_FIXTURE,
            _RUN_FIXTURE.replace(
                "  execFileSync('git', ['fetch', 'origin', "
                "`refs/tags/${gitTag}:refs/tags/${gitTag}`, '--force'], {\n"
                "    cwd,\n"
                "    encoding: 'utf8',\n"
                "  });\n",
                "",
            ),
            "must fetch one tag",
            id="missing-tag-fetch",
        ),
        pytest.param(
            _PLAN_FIXTURE,
            _RUN_FIXTURE.replace(
                "refs/tags/${gitTag}:refs/tags/${gitTag}",
                "refs/heads/main:refs/heads/main",
            ),
            "refs/heads/main",
            id="branch-fetch",
        ),
    ],
)
def test_offline_fixture_refuses_a_broken_release_script(
    monkeypatch: pytest.MonkeyPatch, plan: str, runner: str, message: str
) -> None:
    assert plan != _PLAN_FIXTURE or runner != _RUN_FIXTURE
    _patch_release_scripts(monkeypatch, plan, runner)
    with pytest.raises(AssertionError, match=message):
        _check_release_scripts(*_fetched_release_scripts())
