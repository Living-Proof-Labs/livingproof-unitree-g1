# SPDX-License-Identifier: Apache-2.0
"""A release build reports the release tag's version, never a 0.0.0 or dev fallback.

Proposed by Vera in the livingproof-unitree-g1#49 review (livingproof-unitree-g1#37). A scratch clone of
this checkout is tagged with the next release, then built from (a) the tagged checkout and
(b) ``git archive`` of the tag, which is what GitHub's release "Source code" tarball holds.
Both wheels must carry that version. No upstream version is frozen: the tag is derived
from the newest local release tag.
"""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from lpl_g1 import ROOT

DIST = "livingproof_unitree_g1"


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _wheel_version(source: Path, out: Path) -> str:
    subprocess.run(["uv", "build", "-q", "--wheel", "-o", str(out), str(source)], check=True)
    (wheel,) = out.glob(f"{DIST}-*.whl")
    with zipfile.ZipFile(wheel) as z:
        meta = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode()
    return re.search(r"^Version: (.+)$", meta, re.MULTILINE).group(1)


@pytest.fixture
def tagged(tmp_path):
    if not shutil.which("uv") or not (ROOT / ".git").exists():
        pytest.skip("needs uv and a git checkout")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--no-local", str(ROOT), str(clone)], check=True)
    tags = [
        t
        for t in _git(ROOT, "tag", "--list", "v[0-9]*").split()
        if re.fullmatch(r"v\d+\.\d+\.\d+", t)
    ]
    if not tags:
        pytest.skip("no release tag in this checkout")
    major, minor, patch = max(tuple(map(int, t[1:].split("."))) for t in tags)
    tag = f"v{major}.{minor}.{patch + 1}"
    # Build exactly this checkout's tree, uncommitted test files included.
    for path in _git(ROOT, "ls-files", "--others", "--modified", "--exclude-standard").splitlines():
        if (ROOT / path).is_file():
            (clone / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, clone / path)
    ident = ["-c", "user.name=review", "-c", "user.email=review@example.invalid"]
    _git(clone, "add", "-A")
    _git(clone, *ident, "commit", "-q", "--allow-empty", "-m", "release candidate")
    _git(clone, *ident, "tag", tag)
    return clone, tag[1:]


def test_a_tagged_checkout_builds_the_release_version(tagged, tmp_path):
    clone, release = tagged
    assert _wheel_version(clone, tmp_path / "checkout") == release


def test_the_release_source_archive_builds_the_release_version(tagged, tmp_path):
    clone, release = tagged
    archive = tmp_path / "archive"
    archive.mkdir()
    tar = subprocess.run(
        ["git", "archive", f"v{release}"], cwd=clone, check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(archive)], input=tar, check=True)
    built = _wheel_version(archive, tmp_path / "archive-wheel")
    assert built == release, f"GitHub's source tarball for v{release} would install as {built}"
