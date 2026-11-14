# SPDX-License-Identifier: Apache-2.0
"""Variants of world or scenario selection that the name and key guard does not catch.

tests/test_scenario_neutral.py catches the removed names and the exact keys. These
checks catch selection that comes back under a neutral name: a key spelled
``world_id`` or ``scenarios``, a pinned world file in another group, a scenario
key on the adapter table, a Taskfile flag, or simulator scenario and world code
imported from a new module.
"""

import ast
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "lpl_g1"
# A key that names a world or a scenario: world, worlds, world_id, scenario_ref, ...
SELECTING_KEY = re.compile(r"^(world|scenario)s?([_-].*)?$", re.IGNORECASE)
REMOVED = (
    "verification.lock.json",
    ".github/workflows/verify.yml",
    "tools/repin.py",
    "src/lpl_g1/scenario.py",
    "src/lpl_g1/evidence.py",
)


def _keys(value, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)


def _json_documents():
    tracked = subprocess.run(
        ["git", "ls-files", "*.json"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.split()
    paths = {ROOT / p for p in tracked} | set(ROOT.glob("*.json"))
    paths |= set((ROOT / "configurations").glob("*.json"))
    return sorted(p for p in paths if p.is_file() and "tests/data" not in p.as_posix())


def test_removed_files_stay_removed():
    assert [p for p in REMOVED if (ROOT / p).exists()] == []


def test_no_json_document_has_a_world_or_scenario_key_under_any_spelling():
    offenders = []
    for path in _json_documents():
        found: list[str] = []
        _keys(json.loads(path.read_text(encoding="utf-8")), found)
        bad = sorted({k for k in found if SELECTING_KEY.match(k)})
        if bad:
            offenders.append(f"{path.relative_to(ROOT)}: {bad}")
    assert offenders == []


def test_pins_hold_no_world_or_scenario_file_in_any_group():
    groups = json.loads((ROOT / "pins.json").read_text(encoding="utf-8"))["groups"]
    offenders = []
    for name, group in groups.items():
        if group.get("repository", "").endswith("/livingproof-worlds"):
            offenders.append(f"{name}: {group['repository']}")
        for path in group.get("files", {}):
            if path.startswith("worlds/") or "/worlds/" in path or "/scenarios/" in path:
                offenders.append(f"{name}: {path}")
    assert offenders == []


def test_adapter_table_names_only_the_adapter():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(pyproject["tool"]["livingproof"]) == {"adapter"}


def test_taskfile_passes_no_world_or_scenario_and_runs_no_evidence_loop():
    taskfile = (ROOT / "Taskfile.yml").read_text(encoding="utf-8")
    for needle in (
        "--scenario",
        "--world",
        "livingproof_sim_mujoco.evidence",
        "livingproof_sim_mujoco.walk",
        "verification.lock",
        "capability_verifier",
    ):
        assert needle not in taskfile, needle


def test_package_imports_no_simulator_scenario_or_world_code():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = {a.name for a in node.names}
                if node.module in {
                    "livingproof_sim_mujoco.scenario",
                    "livingproof_sim_mujoco.evidence",
                    "livingproof_sim_mujoco.worlds",
                }:
                    offenders.append(f"{path.name}: from {node.module}")
                if names & {"add_world", "load_world", "scenario", "evidence", "worlds"}:
                    offenders.append(f"{path.name}: {sorted(names)} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[-1] in {"scenario", "evidence", "worlds"}:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.Attribute) and node.attr in {"add_world", "load_world"}:
                offenders.append(f"{path.name}: .{node.attr}")
    assert offenders == []


def test_renovate_config_tracks_no_world_or_lock():
    text = (ROOT / "renovate.json").read_text(encoding="utf-8")
    for needle in ("livingproof-worlds", "verification.lock", "verify.yml"):
        assert needle not in text, needle


def test_names_are_searched_in_every_text_file_not_only_listed_suffixes():
    """test_scenario_neutral.py reads only listed suffixes; a .sh or .json5 file is skipped."""
    from test_scenario_neutral import ALLOW, LEGACY_EVIDENCE, NAMES, SKIP_DIRS

    offenders = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in SKIP_DIRS for part in relative.parts):
            continue
        if relative in ALLOW or relative in LEGACY_EVIDENCE:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary: npz traces, meshes
        if NAMES.search(path.name) or NAMES.search(text):
            offenders.append(str(relative))
    assert offenders == []
