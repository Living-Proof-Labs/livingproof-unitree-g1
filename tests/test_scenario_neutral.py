# SPDX-License-Identifier: Apache-2.0
"""The robot package does not select a world or a scenario.

Names of a scenario or a world may appear only on an allowlisted path, or as the
scenario ids the robot manifest declares. This repository has no recorded
scenario evidence to read, so the legacy evidence allowlist is empty.
"""

import json
import re
from pathlib import Path

import lpl_g1
from lpl_g1 import ROOT
from lpl_g1.adapter import G1

# This file names the forbidden words. The supported-scenarios test names them too,
# because it has to say which id is declared and which is not.
ALLOW = {
    # Exact released inputs for adapter-only regression tests; not runtime selection.
    Path("tests/test_shared_traversals.py"),
    Path("tests/data/reuse/world.json"),
    Path("tests/data/reuse/release.json"),
    Path("tests/data/reuse/left.json"),
    Path("tests/data/reuse/right.json"),
    Path("tests/data/reuse/sources.json"),
    Path("tests/test_scenario_neutral.py"),
    Path("tests/test_supported_scenarios.py"),
    # Runs the declared scenarios end to end, so it names their worlds (#40, 11b).
    Path("tests/test_route_drive.py"),
}
LEGACY_EVIDENCE: set[Path] = set()
ROBOT_MANIFEST = Path("manifest.json")
DECLARED_SCENARIOS = ("walk-hallway", "walk-corner")
NAMES = re.compile(r"hallway|walk-hallway|hallway-open|\bcorner\b", re.IGNORECASE)
SELECTING_KEYS = {"world", "scenario", "worlds", "scenarios"}


def declaration_text(relative: Path, text: str) -> str:
    """Drop the declared scenario ids. Any other scenario or world name still matches."""
    if relative == ROBOT_MANIFEST:
        for scenario in DECLARED_SCENARIOS:
            text = text.replace(f'"{scenario}"', "")
    return text


SKIP_DIRS = {
    ".git",
    ".cache",
    ".venv",
    ".venv-docs",
    "artifacts",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "downloads",
}
TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yml",
    ".yaml",
    ".md",
    ".toml",
    ".txt",
    ".in",
    ".lock",
}


def _selected_keys(value, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SELECTING_KEYS:
                found.append(key)
            _selected_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _selected_keys(item, found)


def _text_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"Taskfile.yml", "LICENSE"}:
            continue
        yield path


def test_scenario_and_world_names_stay_on_legacy_evidence_paths():
    assert LEGACY_EVIDENCE == set()
    offenders = []
    for path in _text_files():
        relative = path.relative_to(ROOT)
        if relative in ALLOW or relative in LEGACY_EVIDENCE:
            continue
        text = declaration_text(relative, path.read_text(encoding="utf-8"))
        if NAMES.search(path.name) or NAMES.search(text):
            offenders.append(str(relative))
    assert offenders == []


def test_pins_runner_and_locks_do_not_select_a_world_or_scenario():
    documents = [
        ROOT / "pins.json",
        ROOT / "runner.json",
        ROOT / "robot.json",
        ROOT / "manifest.json",
        *sorted((ROOT / "configurations").glob("*.json")),
        *sorted(ROOT.glob("*.lock.json")),
    ]
    assert not (ROOT / "verification.lock.json").exists()
    for path in documents:
        found: list[str] = []
        _selected_keys(json.loads(path.read_text(encoding="utf-8")), found)
        if path.relative_to(ROOT) == ROBOT_MANIFEST:
            found = [key for key in found if key != "scenarios"]
        assert found == [], f"{path.relative_to(ROOT)} has {found}"


def test_workflows_have_no_world_or_scenario_inputs():
    workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))
    assert workflows
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        assert "walk-hallway.yml" not in text
        keys = []
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if ":" not in code:
                continue
            key = code.split(":", 1)[0].strip()
            if key in SELECTING_KEYS:
                keys.append(key)
        assert keys == [], f"{path.name} has {keys}"


def test_adapter_does_not_offer_a_world_or_scenario():
    package = Path(lpl_g1.__file__).parent
    assert not hasattr(G1, "world")
    assert not (package / "scenario.py").exists()
    assert "load_world" not in (package / "scene.py").read_text(encoding="utf-8")
    assert "add_world" not in (package / "scene.py").read_text(encoding="utf-8")
