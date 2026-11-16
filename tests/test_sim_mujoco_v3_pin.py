# SPDX-License-Identifier: Apache-2.0
"""The sim-mujoco v3 pin and its --lock contract, checked against the installed release.

Vera's review checks, automated: pyproject, uv.lock and the installed package name the
same released v3 tag, and the installed release's evidence and ``walk stand`` commands
refuse to run without ``--lock``.
"""

import importlib.metadata
import re
import subprocess
import sys
import tomllib

from lpl_g1 import ROOT

ADAPTER = "lpl_g1.adapter:ROBOT"
URL = "https://github.com/Living-Proof-Labs/livingproof-sim-mujoco.git"


def _source_tag():
    sources = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]
    entry = sources["livingproof-sim-mujoco"]
    assert entry["git"] == URL
    return entry["tag"]


def test_pyproject_lock_and_installed_release_agree_on_a_v3_tag():
    tag = _source_tag()
    assert re.fullmatch(r"v3\.\d+\.\d+", tag), tag
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (package,) = [p for p in lock["package"] if p["name"] == "livingproof-sim-mujoco"]
    assert package["version"] == tag[1:]
    assert package["source"]["git"].startswith(f"{URL}?tag={tag}#")
    assert importlib.metadata.version("livingproof-sim-mujoco") == tag[1:]


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", *args], capture_output=True, text=True, timeout=120, cwd=ROOT
    )


def test_installed_stand_command_requires_lock():
    result = _run("livingproof_sim_mujoco.walk", "stand", ADAPTER, "--configuration", "playground")
    assert result.returncode == 2, result.stderr
    assert "required" in result.stderr and "--lock" in result.stderr


def test_installed_evidence_command_requires_lock():
    result = _run("livingproof_sim_mujoco.evidence", ADAPTER, "playground", "out")
    assert result.returncode == 2, result.stderr
    assert "required" in result.stderr and "--lock" in result.stderr
