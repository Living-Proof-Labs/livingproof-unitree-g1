# SPDX-License-Identifier: Apache-2.0
"""Leftover world coupling in the robot scene (livingproof-unitree-g1#37).

The simulator takes the world, its walls and its places from livingproof-worlds, so the
robot scene carries no ``walls``. A start place that is not on the compiled model is an
error, as in the Go1 and T1 adapters, and the package reports its release version.
"""

import dataclasses
import re
import subprocess
import tomllib
from importlib.metadata import version

import mujoco
import pytest

from lpl_g1 import ROOT
from lpl_g1.adapter import ROBOT
from lpl_g1.profiles import controller
from lpl_g1.scene import Scene, build


def test_scene_has_no_walls_field():
    assert "walls" not in {f.name for f in dataclasses.fields(Scene)}
    assert not hasattr(build(), "walls")


def test_initial_state_fails_loudly_when_the_place_is_not_on_the_model():
    scene = build()
    for name in ("stock", "playground"):
        c = controller(name, scene)
        with pytest.raises(KeyError, match="spawn-entrance"):
            ROBOT.initial_state(scene.model, mujoco.MjData(scene.model), c, "spawn-entrance")


def _release(text: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in re.match(r"v?(\d+)\.(\d+)\.(\d+)", text).groups())


def test_package_version_follows_the_release_tag():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" not in project["project"], "version must come from the release tag"
    assert "version" in project["project"].get("dynamic", [])
    assert project["tool"]["hatch"]["version"]["source"] == "vcs"
    installed = version("livingproof-unitree-g1")
    assert installed != "0.1.0"
    tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if tag.returncode != 0:
        pytest.skip("no release tag in this checkout")
    assert _release(installed) >= _release(tag.stdout.strip()), (installed, tag.stdout)
