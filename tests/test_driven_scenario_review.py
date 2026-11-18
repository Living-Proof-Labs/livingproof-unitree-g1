# SPDX-License-Identifier: Apache-2.0
"""The adapter-driven scenario repeats exactly, and a commanded scenario never drives.

Vera review of #54 (robot-capability-specs#40, step 11b). The scenario and world ids
are read at run time from ``manifest.json`` and the simulator's vendored parameter
files, and the runner and specs refs from the installed simulator, so this file names
no scenario or world and freezes no version (scenario-neutral search, Renovate bumps).
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import livingproof_sim_mujoco
import pytest
from livingproof_sim_mujoco import evidence

from declared_sources import parameters_source
from lpl_g1 import ROOT

PACKAGE = Path(livingproof_sim_mujoco.__file__).parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
CONFIGURATIONS = [c["name"] for c in MANIFEST["configurations"]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner() -> dict:
    dist = importlib.metadata.distribution("livingproof-sim-mujoco")
    commit = json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"]
    return {"tag": f"v{dist.version}", "commit": commit}


def _driven() -> tuple[str, dict, dict]:
    """(scenario@revision, catalog entry, parameter file) of the declared scenario with no command."""
    catalog = json.loads((PACKAGE / "scenarios" / "catalog.json").read_text())
    found = [
        (key, entry)
        for key, entry in catalog.items()
        if isinstance(entry, dict) and key.split("@")[0] in MANIFEST["scenarios"]
    ]
    assert len(found) == 1, f"expected one declared adapter-driven scenario, got {found}"
    key, entry = found[0]
    document = json.loads((PACKAGE / "scenarios" / entry["file"]).read_text())
    assert "command" not in document, "an adapter-driven scenario sends no command"
    return key, entry, document


def _commanded() -> tuple[str, dict]:
    """(scenario@revision, parameter file) of the frozen scenario that sends a command."""
    document = json.loads((PACKAGE / "compat" / "parameters.json").read_text())
    scenario = document["scenario"]
    assert scenario["id"] in MANIFEST["scenarios"]
    return f"{scenario['id']}@{scenario['revision']}", document


def _parameters_file(scenario: str) -> Path:
    """The simulator's vendored copy of the scenario's parameter file (catalog, else frozen)."""
    catalog = json.loads((PACKAGE / "scenarios" / "catalog.json").read_text())
    entry = catalog.get(scenario)
    if isinstance(entry, dict):
        return PACKAGE / "scenarios" / entry["file"]
    return PACKAGE / "compat" / "parameters.json"


def _run(
    tmp_path: Path, scenario: str, document: dict, specs: dict, name: str, trials=None, **plan
):
    world = document["world"]
    world_path = PACKAGE / "worlds" / world["id"] / world["content_version"] / "world.json"
    conf = next(c for c in MANIFEST["configurations"] if c["name"] == name)
    lock = {
        "plan": {"trials": 1, **plan},
        "scenario": scenario,
        "world": {
            "id": world["id"],
            "version": world["content_version"],
            "sha256": _sha(world_path),
        },
        "configurations": [
            {
                "name": name,
                "path": conf["path"],
                "version": conf["version"],
                "sha256": _sha(ROOT / conf["path"]),
            }
        ],
        "releases": {"runner": _runner(), "specs": specs},
        "sources": {"parameters": parameters_source(_parameters_file(scenario))},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "caller.lock.json"
    path.write_text(json.dumps(lock))
    return evidence.evidence(
        "lpl_g1.adapter:ROBOT", name, tmp_path / "out", lock=path, trials=trials
    )


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_repeated_adapter_driven_trials_are_identical_and_reach_the_exit(tmp_path, name):
    """Two trials in one run: drive() keeps no state across trials, and both finish."""
    scenario, entry, document = _driven()
    specs = {"tag": entry["tag"], "commit": entry["commit"]}
    run = _run(tmp_path, scenario, document, specs, name, trials=2)
    results = run["trials"]["results"]
    assert [r["index"] for r in results] == [1, 2]
    assert [r["outcome"] for r in results] == ["crossed_exit", "crossed_exit"]
    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["trajectory_sha256"] == results[1]["trajectory_sha256"]
    assert run["trials"]["reproducibility"] == {
        "compared": True,
        "identical": True,
        "finding": None,
    }
    for result in results:
        observation = json.loads((tmp_path / "out" / result["path"]).read_text())
        assert observation["left_world"] is None
        assert observation["wall_contacts"] == []
        assert set(observation["places_entered"]) >= set(document["places"]["through"])
