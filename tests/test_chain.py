# SPDX-License-Identifier: Apache-2.0
"""verification.lock.json, the thin verify workflow and the evidence-only boundary.

repin and its tests live in capability-verifier (capability_verifier.repin)."""

import json
from pathlib import Path

import pytest

import lpl_g1
from lpl_g1 import ROOT, evidence
from lpl_g1.fetch import sha256

LOCK = json.loads((ROOT / "verification.lock.json").read_text())
WORKFLOW = (ROOT / ".github/workflows/verify.yml").read_text()
STEPS = "\n".join(line for line in WORKFLOW.splitlines() if not line.lstrip().startswith("#"))


def test_lock_holds_plan_world_configurations_and_every_release_ref():
    assert LOCK["format"] == "livingproof-verification-lock/1.0.0"
    assert LOCK["plan"] == {
        "trials": 20,
        "seeds": list(range(1, 21)),
        "command_speed": 0.5,
        "required_passes": 18,
    }
    assert set(LOCK["releases"]) == {"specs", "runner", "verifier", "reporter"}
    assert LOCK["releases"]["specs"] == {
        "repository": "Living-Proof-Labs/robot-capability-specs",
        "tag": "v1.6.1",
        "commit": "0bdf1db3f975b044692c0b8f1ec37737ad8b889f",
    }
    assert LOCK["world"]["id"] == "hallway-open" and LOCK["world"]["version"] == "0.1.0"
    robot = json.loads((ROOT / "robot.json").read_text())
    assert [c["name"] for c in LOCK["configurations"]] == [
        c["name"] for c in robot["configurations"]
    ]
    for c in LOCK["configurations"]:
        profile = json.loads((ROOT / c["path"]).read_text())
        assert sha256(ROOT / c["path"]) == c["sha256"], c["name"]
        assert profile["version"] == c["version"], c["name"]


def test_every_release_ref_is_a_real_tag():
    runner, reporter = LOCK["releases"]["runner"], LOCK["releases"]["reporter"]
    verifier = LOCK["releases"]["verifier"]
    assert (verifier["tag"], verifier["commit"]) == (
        "v1.5.0",
        "c3aab9ea977b07ee395d20720bd4bbbac7305283",
    )
    assert (runner["tag"], runner["commit"]) == (
        "v1.4.1",  # the GitHub release for v1.4.1 is a draft; the tag and commit exist
        "95ecbd3aae3cfcf81a684398fbf7651281e4eb3a",
    )
    assert reporter["tag"] == "v1.9.0" and reporter["digest"] == (
        "sha256:a7ef8603f44cc5296962e97e8153f9f40e145ef7a631e562a905adefa6fe386e"
    )
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
    assert sorted(p.name for p in (ROOT / ".github/workflows").iterdir()) == ["verify.yml"]
    taskfile = (ROOT / "Taskfile.yml").read_text()
    for generic in ("setup", "fetch", "check", "test", "evidence"):
        assert f"\n  {generic}:\n" not in taskfile, generic
