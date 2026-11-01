# SPDX-License-Identifier: Apache-2.0
"""Download every pin in this package's pins.json into .cache/ and verify its sha256
(livingproof-sim-mujoco's generic pins, bound to this repository)."""

from __future__ import annotations

from pathlib import Path
import sys
import urllib.request

from livingproof_sim_mujoco import pins
from livingproof_sim_mujoco.pins import ChecksumError, sha256, url_for  # noqa: F401

from lpl_g1 import ROOT


def load_pins(root: Path = ROOT) -> dict:
    return pins.load_pins(root)


def local(group_name: str, path: str, root: Path = ROOT) -> Path:
    return pins.local(group_name, path, root)


def fetch(root: Path = ROOT, opener=urllib.request.urlopen) -> list[Path]:
    return pins.fetch(root, opener)


def main() -> int:
    files = fetch()
    print(f"{len(files)} pinned files present and verified under .cache/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
