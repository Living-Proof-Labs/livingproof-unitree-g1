# SPDX-License-Identifier: Apache-2.0
"""The robot package does not select a world or a scenario.

Names of a scenario or a world may appear only on an allowlisted legacy evidence
path. This repository has no recorded scenario evidence to read, so that
allowlist is empty. This file is the only place those names are written, because
the search has to name them.
"""

import json
import re
from pathlib import Path

import lpl_g1
from lpl_g1 import ROOT
from lpl_g1.adapter import G1

# This file names the forbidden words. Nothing else may.
ALLOW = {Path("tests/test_scenario_neutral.py")}
LEGACY_EVIDENCE: set[Path] = set()
NAMES = re.compile(r"hallway|walk-hallway|hallway-open|\bcorner\b", re.IGNORECASE)
SELECTING_KEYS = {"world", "scenario", "worlds", "scenarios"}
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
        if NAMES.search(path.name) or NAMES.search(path.read_text(encoding="utf-8")):
            offenders.append(str(relative))
    assert offenders == []


def test_pins_runner_and_locks_do_not_select_a_world_or_scenario():
    documents = [
        ROOT / "pins.json",
        ROOT / "runner.json",
        ROOT / "robot.json",
        *sorted((ROOT / "configurations").glob("*.json")),
        *sorted(ROOT.glob("*.lock.json")),
    ]
    assert not (ROOT / "verification.lock.json").exists()
    for path in documents:
        found: list[str] = []
        _selected_keys(json.loads(path.read_text(encoding="utf-8")), found)
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
