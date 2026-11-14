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


def test_ci_runs_the_suite_on_pull_request():
    """The suite runs in CI. Deleting the workflow or its pull_request trigger fails here."""
    path = ROOT / ".github/workflows/test.yml"
    text = path.read_text()
    manual = (ROOT / ".github/workflows/manual.yml").read_text()
    assert "\n  pull_request:\n" in text
    assert "\n  push:\n    branches: [main]\n" in text
    assert "permissions:\n  contents: read\n" in text
    assert text.count("permissions:") == 1
    assert "ubuntu-24.04" in text and "ubuntu-24.04-arm" in text
    assert "MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa uv run pytest tests" in text
    assert "uv run ruff check ." in text
    assert "uv run ruff format --check ." in text
    uses = re.findall(r"uses:\s*(\S+)", text)
    manual_uses = set(re.findall(r"uses:\s*(\S+)", manual))
    assert uses
    assert set(uses) <= manual_uses
    assert "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in uses
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in uses
    assert "upload-artifact" not in text and "deploy-pages" not in text
    assert "secrets." not in text


def test_workflows_and_tasks_do_not_run_verification():
    assert sorted(p.name for p in (ROOT / ".github/workflows").iterdir()) == [
        "manual.yml",
        "release.yml",
        "test.yml",
    ]
    assert not (ROOT / "tools/repin.py").exists() and not (ROOT / "tools/pr_comment.py").exists()
    assert not (ROOT / "src/lpl_g1/fetch.py").exists()  # livingproof_sim_mujoco.pins fetches
    taskfile = (ROOT / "Taskfile.yml").read_text()
    for generic in ("setup", "fetch", "check", "test", "evidence", "repin", "stand"):
        assert f"\n  {generic}:\n" not in taskfile, generic
