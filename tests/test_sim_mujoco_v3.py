# SPDX-License-Identifier: Apache-2.0
"""livingproof-sim-mujoco v3 compatibility for this robot package.

v3's evidence command and walk stand command take the caller's verification lock
with --lock, and v3 owns the world. Pin files record each artifact's byte count.
"""

import json
import re

from livingproof_sim_mujoco.pins import local, sha256

from lpl_g1 import ROOT

# evidence is invoked as a module. walk's stand subcommand is `walk stand`.
# Imports such as `from livingproof_sim_mujoco.walk import initial_state` are not calls.
CALL = re.compile(
    r"-m\s+livingproof_sim_mujoco\.(?:evidence|walk)\b"
    r"|livingproof_sim_mujoco\.evidence\b"
    r"|livingproof_sim_mujoco\.walk\s+stand\b"
)
SCAN_ROOTS = (
    ROOT / "src",
    ROOT / ".github" / "workflows",
    ROOT / "tools",
    ROOT / "Taskfile.yml",
)


def _text_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "tests" in path.parts:
                continue
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            yield path


def test_evidence_and_stand_calls_pass_lock():
    """A call to evidence or walk stand must pass --lock. This package has none."""
    offenders = []
    for path in _text_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            if CALL.search(line) and "--lock" not in line:
                offenders.append(f"{path.relative_to(ROOT)}: {line.strip()}")
    assert offenders == []


def test_package_does_not_add_the_world():
    package = ROOT / "src" / "lpl_g1"
    offenders = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "add_world" in text or "load_world" in text:
            offenders.append(path.name)
    assert offenders == []


def test_pin_byte_counts_match_the_downloaded_artifacts():
    groups = json.loads((ROOT / "pins.json").read_text())["groups"]
    problems = []
    for name, group in groups.items():
        for path, entry in group["files"].items():
            label = f"{name}:{path}"
            recorded = entry.get("bytes")
            if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 0:
                problems.append(f"{label}: bytes is {recorded!r}")
                continue
            artifact = local(name, path, ROOT)
            if sha256(artifact) != entry["sha256"]:
                problems.append(f"{label}: cached file is not the pinned artifact")
                continue
            actual = artifact.stat().st_size
            if recorded != actual:
                problems.append(f"{label}: pin {recorded} artifact {actual}")
    assert problems == []
