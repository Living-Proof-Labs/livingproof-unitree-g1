# SPDX-License-Identifier: Apache-2.0
"""Numpy keeps the per-Python versions main already resolves.

``requires-python`` is ``>=3.11``. Main's ``uv.lock`` forks numpy: 2.4.6 below
3.12 and 2.5.4 on 3.12 and newer. CI is Python 3.12.16, so a single
``numpy==2.4.6`` pin downgrades that path. These checks fail that pin.
"""

from __future__ import annotations

import sys
import tomllib
from typing import Any

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from lpl_g1 import ROOT

# Versions origin/main's uv.lock resolves for this requires-python range.
BELOW_312 = "2.4.6"
AT_LEAST_312 = "2.5.4"
ENVIRONMENTS = (
    ("3.11.9", "linux", BELOW_312),
    ("3.11.9", "darwin", BELOW_312),
    ("3.12.0", "linux", AT_LEAST_312),
    ("3.12.16", "linux", AT_LEAST_312),
    ("3.12.16", "darwin", AT_LEAST_312),
    ("3.13.1", "linux", AT_LEAST_312),
)
KEPT_PINS = (
    "pyyaml==6.0.3",
    "imageio[ffmpeg]==2.38.1",
    "jsonschema==4.26.0",
    "referencing==0.37.0",
)


def _environment(python_full: str, sys_platform: str) -> dict[str, str]:
    env = default_environment()
    env["python_full_version"] = python_full
    env["python_version"] = ".".join(python_full.split(".")[:2])
    env["sys_platform"] = sys_platform
    return env


def _exact_version(specifier: SpecifierSet) -> str | None:
    specs = list(specifier)
    if len(specs) == 1 and specs[0].operator == "==":
        return specs[0].version
    return None


def _matches(marker: str | None, env: dict[str, str]) -> bool:
    return marker is None or Marker(marker).evaluate(env)


def _fork_applies(markers: list[str] | None, env: dict[str, str]) -> bool:
    """True when this lock distribution is the one uv selects for env.

    Missing markers mean the distribution applies to every environment, which is
    how a collapsed single pin is written. Otherwise any listed fork may match.
    """
    if not markers:
        return True
    return Marker(" or ".join(f"({marker})" for marker in markers)).evaluate(env)


def numpy_pin_problems(dependencies: list[str], packages: list[dict[str, Any]]) -> list[str]:
    """Problems when the numpy pin or lock downgrades a Python main already resolved."""
    requirements = [Requirement(dep) for dep in dependencies if Requirement(dep).name == "numpy"]
    problems: list[str] = []
    distributions = [
        package
        for package in packages
        if package.get("name") == "numpy" and "version" in package and "wheels" in package
    ]
    roots = [package for package in packages if package.get("name") == "livingproof-unitree-g1"]
    if len(roots) != 1:
        problems.append(f"expected one livingproof-unitree-g1 lock entry, found {len(roots)}")
        root: dict[str, Any] = {}
    else:
        root = roots[0]
    declared = [
        item
        for item in root.get("metadata", {}).get("requires-dist", [])
        if item.get("name") == "numpy"
    ]
    resolved = [item for item in root.get("dependencies", []) if item.get("name") == "numpy"]

    for python_full, platform, expected in ENVIRONMENTS:
        env = _environment(python_full, platform)
        where = f"Python {python_full} on {platform}"
        pinned = [
            _exact_version(req.specifier) or str(req.specifier)
            for req in requirements
            if req.marker is None or req.marker.evaluate(env)
        ]
        if pinned != [expected]:
            problems.append(f"{where} pyproject numpy pin is {pinned}, expected ['{expected}']")
        locked = [
            package["version"]
            for package in distributions
            if _fork_applies(package.get("resolution-markers"), env)
        ]
        if locked != [expected]:
            problems.append(f"{where} uv.lock numpy is {locked}, expected ['{expected}']")
        required = [
            _exact_version(SpecifierSet(item.get("specifier") or "")) or item.get("specifier")
            for item in declared
            if _matches(item.get("marker"), env)
        ]
        if required != [expected]:
            problems.append(
                f"{where} lock requires-dist numpy is {required}, expected ['{expected}']"
            )
        dependency_versions = [
            item.get("version") for item in resolved if _matches(item.get("marker"), env)
        ]
        if dependency_versions != [expected]:
            problems.append(
                f"{where} lock dependency numpy is {dependency_versions}, expected ['{expected}']"
            )
    return problems


def _project_and_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return project, lock


def test_numpy_pin_keeps_mains_python_312_resolution():
    project, lock = _project_and_lock()
    assert project["project"]["requires-python"] == ">=3.11"
    dependencies = project["project"]["dependencies"]
    for pin in KEPT_PINS:
        assert pin in dependencies
    problems = numpy_pin_problems(dependencies, lock["package"])
    assert problems == []


def test_single_numpy_pin_downgrades_python_312():
    """The pin Vera rejected: one ``numpy==2.4.6`` and a lock with no 3.12 fork."""
    packages = [
        {"name": "numpy", "version": BELOW_312, "wheels": []},
        {
            "name": "livingproof-unitree-g1",
            "dependencies": [{"name": "numpy", "version": BELOW_312}],
            "metadata": {"requires-dist": [{"name": "numpy", "specifier": f"=={BELOW_312}"}]},
        },
    ]
    problems = numpy_pin_problems([f"numpy=={BELOW_312}"], packages)
    downgrade = [
        problem
        for problem in problems
        if "3.12.16" in problem and BELOW_312 in problem and AT_LEAST_312 in problem
    ]
    assert downgrade


def test_installed_numpy_matches_this_interpreter():
    import numpy

    expected = AT_LEAST_312 if sys.version_info >= (3, 12) else BELOW_312
    assert numpy.__version__ == expected


def test_workflows_keep_the_python_3_12_16_pin():
    for name in ("test.yml", "manual.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text()
        assert "python-version: '3.12.16'" in text
