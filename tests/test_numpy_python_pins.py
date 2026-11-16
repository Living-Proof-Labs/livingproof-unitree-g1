# SPDX-License-Identifier: Apache-2.0
"""Per-Python numpy pins stay resolvable without freezing today's versions.

``requires-python`` covers both sides of 3.12. numpy 2.5 needs Python >=3.12, so
the ``python_version < '3.12'`` pin stays below 2.5 and the ``>= 3.12`` pin stays
at or above 2.5.4. A later bump of the 3.12 line still passes when pyproject,
``uv.lock`` ``requires-dist``, the locked package and the install all agree.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import tomllib
from typing import Any

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from lpl_g1 import ROOT

# Bounds, not the versions currently locked. 2.5.4 is the floor already resolved
# on Python >=3.12; anything below 2.5 can still be installed on Python 3.11.
BELOW_3_12 = SpecifierSet("<2.5")
AT_LEAST_3_12 = SpecifierSet(">=2.5.4")
ENVIRONMENTS = (
    ("3.11.9", "linux"),
    ("3.11.9", "darwin"),
    ("3.12.0", "linux"),
    ("3.12.0", "darwin"),
    ("3.13.1", "linux"),
)
NUMPY_RENOVATE_RULE = {
    "description": (
        "numpy 2.5+ requires Python >=3.12; keep the python_version < '3.12' line on 2.4.x."
    ),
    "matchManagers": ["pep621"],
    "matchPackageNames": ["numpy"],
    "matchCurrentValue": r"/^==2\.4\./",
    "allowedVersions": "<2.5",
}
PYTHON_VERSION = re.compile(r"(?m)^[ \t]*python-version:[ \t]*['\"]?([^'\"#\s]+)['\"]?[ \t]*$")


def _environment(python_full: str, sys_platform: str) -> dict[str, str]:
    env = dict(default_environment())
    env["python_full_version"] = python_full
    env["python_version"] = ".".join(python_full.split(".")[:2])
    env["sys_platform"] = sys_platform
    return env


def _bound(python_full: str) -> SpecifierSet:
    if Version(python_full) < Version("3.12"):
        return BELOW_3_12
    return AT_LEAST_3_12


def _exact_version(specifier: SpecifierSet | str | None) -> str | None:
    if specifier is None or specifier == "":
        return None
    parsed = specifier if isinstance(specifier, SpecifierSet) else SpecifierSet(str(specifier))
    specs = list(parsed)
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


def _numpy_distributions(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        package
        for package in packages
        if package.get("name") == "numpy" and "version" in package and "wheels" in package
    ]


def _root(packages: list[dict[str, Any]]) -> dict[str, Any] | None:
    roots = [package for package in packages if package.get("name") == "livingproof-unitree-g1"]
    if len(roots) != 1:
        return None
    return roots[0]


def numpy_pin_problems(dependencies: list[str], packages: list[dict[str, Any]]) -> list[str]:
    """Problems when a Python sees anything but one in-range numpy pin, agreed everywhere."""
    requirements = [Requirement(dep) for dep in dependencies if Requirement(dep).name == "numpy"]
    distributions = _numpy_distributions(packages)
    root = _root(packages)
    if root is None:
        found = [package.get("name") for package in packages]
        return [f"expected one livingproof-unitree-g1 lock entry, found {found}"]
    declared = [
        item
        for item in root.get("metadata", {}).get("requires-dist", [])
        if item.get("name") == "numpy"
    ]
    resolved = [item for item in root.get("dependencies", []) if item.get("name") == "numpy"]
    problems: list[str] = []

    for python_full, sys_platform in ENVIRONMENTS:
        env = _environment(python_full, sys_platform)
        where = f"Python {python_full} on {sys_platform}"
        pinned = [
            _exact_version(req.specifier)
            for req in requirements
            if req.marker is None or req.marker.evaluate(env)
        ]
        if len(pinned) != 1 or pinned[0] is None:
            problems.append(f"{where}: expected exactly one exact numpy pin, found {pinned}")
            continue
        version = pinned[0]
        bound = _bound(python_full)
        if Version(version) not in bound:
            problems.append(f"{where}: numpy pin {version} does not satisfy {bound}")
        required = [
            _exact_version(item.get("specifier") or "")
            for item in declared
            if _matches(item.get("marker"), env)
        ]
        locked = [
            package["version"]
            for package in distributions
            if _fork_applies(package.get("resolution-markers"), env)
        ]
        dependency_versions = [
            item.get("version") for item in resolved if _matches(item.get("marker"), env)
        ]
        if required != [version] or locked != [version] or dependency_versions != [version]:
            problems.append(
                f"{where}: numpy sources disagree: pyproject {version}, "
                f"requires-dist {required}, locked package {locked}, "
                f"dependency {dependency_versions}"
            )
    return problems


def _project_and_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return project, lock


def _locked_numpy_version(
    packages: list[dict[str, Any]], python_full: str, sys_platform: str
) -> str:
    env = _environment(python_full, sys_platform)
    versions = [
        package["version"]
        for package in _numpy_distributions(packages)
        if _fork_applies(package.get("resolution-markers"), env)
    ]
    if len(versions) != 1:
        raise AssertionError(f"{python_full} on {sys_platform} locked numpy {versions}")
    return versions[0]


def _selected_pin(dependencies: list[str], python_full: str, sys_platform: str) -> list[str | None]:
    env = _environment(python_full, sys_platform)
    return [
        _exact_version(req.specifier)
        for req in (Requirement(dep) for dep in dependencies)
        if req.name == "numpy" and (req.marker is None or req.marker.evaluate(env))
    ]


def test_each_python_has_one_numpy_pin_and_every_source_agrees():
    project, lock = _project_and_lock()
    requires = SpecifierSet(project["project"]["requires-python"])
    assert requires.contains("3.11.9")
    assert requires.contains("3.12.0")
    dependencies = project["project"]["dependencies"]
    assert numpy_pin_problems(dependencies, lock["package"]) == []

    import numpy

    selected = _selected_pin(dependencies, platform.python_version(), sys.platform)
    assert selected == [numpy.__version__]


def test_single_numpy_pin_downgrades_python_312():
    """Collapsing to the lock's own < 3.12 numpy drops the >= 3.12 fork."""
    _, lock = _project_and_lock()
    packages = lock["package"]
    below = _locked_numpy_version(packages, "3.11.9", "linux")
    above = _locked_numpy_version(packages, "3.12.0", "linux")
    assert Version(below) in BELOW_3_12
    assert Version(above) in AT_LEAST_3_12
    assert Version(below) < Version(above)
    collapsed = [
        {"name": "numpy", "version": below, "wheels": []},
        {
            "name": "livingproof-unitree-g1",
            "dependencies": [{"name": "numpy", "version": below}],
            "metadata": {"requires-dist": [{"name": "numpy", "specifier": f"=={below}"}]},
        },
    ]
    problems = numpy_pin_problems([f"numpy=={below}"], collapsed)
    assert problems
    for problem in problems:
        python_full = problem.split()[1]
        assert Version(python_full) >= Version("3.12"), problem
        assert below in problem
        assert "does not satisfy" in problem


def test_renovate_keeps_the_pre_3_12_numpy_line_below_2_5():
    config = json.loads((ROOT / "renovate.json").read_text())
    assert NUMPY_RENOVATE_RULE in config.get("packageRules", [])


def test_workflow_python_versions_agree():
    found: list[tuple[str, str]] = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        found.extend((path.name, version) for version in PYTHON_VERSION.findall(path.read_text()))
    assert found, "expected at least one workflow python-version"
    versions = {version for _, version in found}
    assert len(versions) == 1, found
