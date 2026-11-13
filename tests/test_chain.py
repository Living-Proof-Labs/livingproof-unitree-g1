# SPDX-License-Identifier: Apache-2.0
"""verification.lock.json, the thin verify workflow and the evidence-only boundary.

repin and its tests live in capability-verifier (capability_verifier.repin)."""

import json
import re
import tomllib
from pathlib import Path

import pytest

import lpl_g1
from lpl_g1 import ROOT, evidence
from livingproof_sim_mujoco.pins import sha256

LOCK = json.loads((ROOT / "verification.lock.json").read_text())
WORKFLOW = (ROOT / ".github/workflows/verify.yml").read_text()
STEPS = "\n".join(line for line in WORKFLOW.splitlines() if not line.lstrip().startswith("#"))


def test_lock_holds_plan_world_configurations_and_every_release_ref():
    assert LOCK["format"] == "livingproof-verification-lock/1.0.0"
    assert LOCK["plan"] == {"trials": {"count": 3}, "command_speed": 0.5}
    assert "seeds" not in LOCK["plan"] and "required_passes" not in LOCK["plan"]
    assert set(LOCK["releases"]) == {"specs", "runner", "verifier", "reporter"}
    assert LOCK["releases"]["specs"]["repository"] == "Living-Proof-Labs/robot-capability-specs"
    assert LOCK["world"]["id"] == "hallway-open" and LOCK["world"]["version"] == "0.1.0"
    robot = json.loads((ROOT / "robot.json").read_text())
    assert [c["name"] for c in LOCK["configurations"]] == [
        c["name"] for c in robot["configurations"]
    ]
    for c in LOCK["configurations"]:
        profile = json.loads((ROOT / c["path"]).read_text())
        assert sha256(ROOT / c["path"]) == c["sha256"], c["name"]
        assert profile["version"] == c["version"], c["name"]


def test_release_refs_come_from_the_lock():
    """No pinned ref is written here: the lock is the source, and every other place that
    names a release (verify.yml uses:, the pyproject runner tag, uv.lock) must agree with it."""
    releases = LOCK["releases"]
    for name, ref in releases.items():
        assert re.fullmatch(r"v\d+\.\d+\.\d+", ref["tag"]), name
        assert re.fullmatch(r"[0-9a-f]{40}", ref["commit"]), name
    runner, verifier = releases["runner"], releases["verifier"]
    assert f"walk-hallway.yml@{verifier['commit']}" in STEPS
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["uv"]["sources"]["livingproof-sim-mujoco"]["tag"] == runner["tag"]
    uv_lock = (ROOT / "uv.lock").read_text()
    assert f"livingproof-sim-mujoco.git?tag={runner['tag']}#{runner['commit']}" in uv_lock
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", releases["reporter"]["digest"])
    assert evidence.pending(LOCK) == []


def test_evidence_refuses_pending_refs(tmp_path, monkeypatch):
    lock = json.loads(json.dumps(LOCK))
    lock["releases"]["verifier"].update(tag="PENDING_RELEASE", commit="PENDING_RELEASE")
    monkeypatch.setattr(evidence.generic, "load_lock", lambda *a, **k: lock)
    with pytest.raises(SystemExit, match="PENDING_RELEASE refs \\(verifier\\)"):
        evidence.evidence("stock", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_verify_is_a_thin_call_to_the_verifier_workflow():
    assert "Living-Proof-Labs/capability-verifier/.github/workflows/walk-hallway.yml@" in STEPS
    assert "pull-requests: write" in STEPS
    assert "--require-pass" not in STEPS
    assert "runs-on:" not in STEPS  # every job lives in the reusable workflow
    assert not (ROOT / "tools/repin.py").exists() and not (ROOT / "tools/pr_comment.py").exists()
    assert not (ROOT / "src/lpl_g1/fetch.py").exists()  # livingproof_sim_mujoco.pins fetches


def test_robot_repository_does_no_grading():
    package = Path(lpl_g1.__file__).parent
    assert not (package / "clearance.py").exists() and not (package / "stats.py").exists()
    from lpl_g1 import scenario

    for name in ("evaluate", "grade", "compare", "interval", "manifest"):
        assert not hasattr(scenario, name), name


def test_ci_runs_through_the_shared_workflow_and_the_adapter():
    import importlib
    import tomllib

    ref = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["livingproof"]["adapter"]
    module, name = ref.split(":")
    adapter = getattr(importlib.import_module(module), name)
    assert ref == "lpl_g1.adapter:ROBOT" and adapter.root == ROOT
    assert sorted(p.name for p in (ROOT / ".github/workflows").iterdir()) == [
        "release.yml",
        "verify.yml",
    ]
    taskfile = (ROOT / "Taskfile.yml").read_text()
    for generic in ("setup", "fetch", "check", "test", "evidence"):
        assert f"\n  {generic}:\n" not in taskfile, generic
