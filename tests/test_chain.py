# SPDX-License-Identifier: Apache-2.0
"""The robot package publishes an adapter and configurations.

It does not grade, and it does not select a world or a scenario. The simulator
dependency stays a released tag because the adapter tests import it.
"""

import importlib
import re
import tomllib
from pathlib import Path

import lpl_g1
from lpl_g1 import ROOT

TAG = re.compile(r"v\d+\.\d+\.\d+")


def test_adapter_is_the_published_contract():
    ref = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["livingproof"]["adapter"]
    module, name = ref.split(":")
    adapter = getattr(importlib.import_module(module), name)
    assert ref == "lpl_g1.adapter:ROBOT"
    assert adapter.id == "unitree-g1" and adapter.root == ROOT
    assert not hasattr(adapter, "world")


def test_simulator_dependency_is_a_released_tag():
    """Adapter tests import livingproof-sim-mujoco. The pin is a release tag."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    source = pyproject["tool"]["uv"]["sources"]["livingproof-sim-mujoco"]
    assert set(source) == {"git", "tag"}
    assert TAG.fullmatch(source["tag"])
    assert "rev" not in source and "branch" not in source
    uv_lock = (ROOT / "uv.lock").read_text()
    assert f"livingproof-sim-mujoco.git?tag={source['tag']}#" in uv_lock


def test_robot_repository_does_no_grading():
    package = Path(lpl_g1.__file__).parent
    assert not (package / "clearance.py").exists() and not (package / "stats.py").exists()
    assert not (package / "scenario.py").exists() and not (package / "evidence.py").exists()
    for path in package.glob("*.py"):
        text = path.read_text()
        assert "def evaluate" not in text and "def grade" not in text, path.name


def test_workflows_and_tasks_do_not_run_verification():
    assert sorted(p.name for p in (ROOT / ".github/workflows").iterdir()) == [
        "manual.yml",
        "release.yml",
    ]
    assert not (ROOT / "tools/repin.py").exists() and not (ROOT / "tools/pr_comment.py").exists()
    assert not (ROOT / "src/lpl_g1/fetch.py").exists()  # livingproof_sim_mujoco.pins fetches
    taskfile = (ROOT / "Taskfile.yml").read_text()
    for generic in ("setup", "fetch", "check", "test", "evidence", "repin", "stand"):
        assert f"\n  {generic}:\n" not in taskfile, generic
