# SPDX-License-Identifier: Apache-2.0
"""Download every pinned file by commit into .cache/ and verify its sha256."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from lpl_g1 import ROOT


class ChecksumError(RuntimeError):
    pass


def load_pins(root: Path = ROOT) -> dict:
    return json.loads((root / "pins.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def url_for(group: dict, path: str, entry: dict) -> str:
    return entry.get("url") or (
        f"https://raw.githubusercontent.com/{group['repository']}/{group['commit']}/{path}"
    )


def local(group_name: str, path: str, root: Path = ROOT) -> Path:
    return root / load_pins(root)["groups"][group_name]["dest"] / path


def fetch(root: Path = ROOT, opener=urllib.request.urlopen) -> list[Path]:
    """Download missing files; verify every file, cached or not. Raises on a mismatch."""
    done = []
    for name, group in load_pins(root)["groups"].items():
        for path, entry in group["files"].items():
            dest = root / group["dest"] / path
            if not dest.exists() or sha256(dest) != entry["sha256"]:
                data = opener(url_for(group, path, entry)).read()
                got = hashlib.sha256(data).hexdigest()
                if got != entry["sha256"]:
                    raise ChecksumError(
                        f"{name}:{path}: expected {entry['sha256']}, downloaded {got}"
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            elif sha256(dest) != entry["sha256"]:
                raise ChecksumError(f"{name}:{path}: cached file does not match its pin")
            done.append(dest)
    return done


def main() -> int:
    files = fetch()
    print(f"{len(files)} pinned files present and verified under .cache/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
