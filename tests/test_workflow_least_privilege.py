# SPDX-License-Identifier: Apache-2.0
"""test.yml stays the only pytest CI, least-privilege, SHA-pinned, on amd64 and arm64."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
TEST = WORKFLOWS / "test.yml"


def _code(text: str) -> str:
    return "\n".join(line.split(" #", 1)[0].rstrip() for line in text.splitlines())


def test_test_workflow_is_read_only_and_uses_no_secrets():
    code = _code(TEST.read_text(encoding="utf-8"))
    top = re.search(r"^permissions:\n((?:[ \t]+.+\n?)+)", code + "\n", re.M)
    assert top is not None and top.group(1).strip() == "contents: read"
    # Only the top-level block; no job widens it.
    assert len(re.findall(r"^\s*permissions\s*:", code, re.M)) == 1
    assert "write" not in code and "write-all" not in code
    assert "secrets" not in code and "GITHUB_TOKEN" not in code
    assert "pull_request_target" not in code and "workflow_run" not in code
    assert "persist-credentials: false" in code


def test_test_workflow_triggers_on_pull_requests_and_main():
    code = _code(TEST.read_text(encoding="utf-8"))
    on = re.search(r"^on:\n((?:[ \t]+.+\n)+)", code + "\n", re.M)
    assert on is not None
    block = on.group(1)
    assert re.search(r"^  pull_request:\s*$", block, re.M)
    assert re.search(r"^  push:\n    branches: \[main\]\s*$", block, re.M)


def test_every_action_is_pinned_to_a_full_commit():
    code = _code(TEST.read_text(encoding="utf-8"))
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", code, re.M)
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref
    # Steps only: no reusable workflow call at job level.
    assert not re.search(r"^  [\w-]+:\n(?:    .*\n)*?    uses:", code + "\n", re.M)


def test_matrix_runs_on_both_native_architectures():
    code = _code(TEST.read_text(encoding="utf-8"))
    matrix = re.search(r"^\s*os:\s*\[([^\]]*)\]", code, re.M)
    assert matrix is not None
    assert [s.strip() for s in matrix.group(1).split(",")] == ["ubuntu-24.04", "ubuntu-24.04-arm"]
    assert "runs-on: ${{ matrix.os }}" in code
    assert "binfmt" not in code and "qemu" not in code.lower()


def test_test_yml_is_the_only_workflow_that_runs_pytest():
    runners = sorted(
        p.name for p in WORKFLOWS.glob("*.y*ml") if "pytest" in _code(p.read_text(encoding="utf-8"))
    )
    assert runners == ["test.yml"]
    assert not (WORKFLOWS / "verify.yml").exists()
