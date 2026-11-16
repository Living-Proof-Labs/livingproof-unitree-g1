# SPDX-License-Identifier: Apache-2.0
"""Review of livingproof-unitree-g1#37: release source archives and a missing start place.

1. A GitHub release "Source code" tarball has no ``.git``. With setuptools-scm's
   ``.git_archival.txt`` (export-subst) it still builds the tag's version, not 0.0.0.
2. A start place the composed world does not have stops the trial as the simulator
   does: ``INCOMPATIBLE``, no trial recorded, naming the place, with no advice to
   compose a world the simulator has already composed. No trial starts at the origin.
"""

import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest

from lpl_g1 import ROOT
from lpl_g1.adapter import ROBOT

CONFIGURATIONS = ("stock", "playground")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _next_release() -> str:
    tags = _git(ROOT, "tag", "--list", "v[0-9]*").split()
    found = [
        tuple(int(p) for p in m.groups())
        for m in map(re.compile(r"^v(\d+)\.(\d+)\.(\d+)$").match, tags)
        if m
    ]
    major, minor, patch = max(found, default=(0, 0, 0))
    return f"{major}.{minor}.{patch + 1}"


def _wheel_version(source: Path, out: Path) -> str:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        text = archive.read(metadata).decode()
    return re.search(r"^Version: (.+)$", text, re.MULTILINE).group(1)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the wheel")
def test_release_source_archive_builds_the_tag_version(tmp_path):
    """The committed tree, tagged as the next release, then built like GitHub's tarball.

    The tag goes on a fresh empty commit, so it is the only tag there. Tagging HEAD
    directly would add a second tag when HEAD is already a release (every release
    commit on main), and the archive's ``describe-name`` would name the older tag.
    """
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--no-local", str(ROOT), str(clone))
    release = _next_release()
    _git(clone, "commit", "-q", "--allow-empty", "-m", "release candidate")
    _git(clone, "tag", f"v{release}")
    assert _git(clone, "tag", "--points-at", "HEAD").split() == [f"v{release}"]
    archive = tmp_path / "source.tar"
    _git(
        clone,
        "archive",
        "--format=tar",
        f"--prefix=src-{release}/",
        "-o",
        str(archive),
        f"v{release}",
    )
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "archive", filter="data")
    source = tmp_path / "archive" / f"src-{release}"
    assert not (source / ".git").exists()
    assert _wheel_version(source, tmp_path / "from-archive") == release
    archival = (source / ".git_archival.txt").read_text()
    assert f"describe-name: v{release}" in archival, archival
    assert _wheel_version(clone, tmp_path / "from-checkout") == release


def _composed_world() -> dict:
    """A world the pinned simulator vendors with a pose place named spawn-entrance."""
    from livingproof_sim_mujoco import compose

    for release in sorted((files("livingproof_sim_mujoco") / "worlds").glob("*/*/world.json")):
        world = json.loads(release.read_text())
        places = {p["name"]: p for p in world.get("places", [])}
        if places.get("spawn-entrance", {}).get("type") == "pose":
            fixture, version = release.parent.parent.name, release.parent.name
            try:
                return compose.vendored_world({"world": {"id": fixture, "version": version}})[0]
            except SystemExit:
                continue
    pytest.skip("the pinned simulator vendors no world with spawn-entrance")


def test_start_place_missing_from_the_composed_world_is_incompatible():
    from livingproof_sim_mujoco import compose

    world = _composed_world()
    for name in CONFIGURATIONS:
        _model, data, _ctrl, _walls = compose.place(ROBOT, name, world, "spawn-entrance")
        assert data.qpos[0] < -1.0, "a real place still starts there"
        with pytest.raises(SystemExit) as caught:
            compose.place(ROBOT, name, world, "spawn-nowhere")
        text = str(caught.value)
        assert text == str(caught.value.code)
        assert text.startswith("INCOMPATIBLE: start place 'spawn-nowhere'"), text
        assert "No trial was recorded" in text
        assert "place:spawn-nowhere" in text
        assert "compose the world first" not in text.lower()
        assert not text.startswith('"'), "str() of the error is the message, not its repr"
        # Callers that caught the earlier KeyError still do.
        assert isinstance(caught.value, KeyError)
