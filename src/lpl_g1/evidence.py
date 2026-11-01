# SPDX-License-Identifier: Apache-2.0
"""walk-hallway sim evidence for one G1 configuration: livingproof-sim-mujoco's generic
loop driven by ``lpl_g1.adapter``. Records observations; never grades.

    python -m lpl_g1.evidence <configuration> <out> [--allow-pending]
"""

from __future__ import annotations

from pathlib import Path
import sys

from livingproof_sim_mujoco import evidence as generic
from livingproof_sim_mujoco.evidence import FORMAT, LOCK, PENDING, pending  # noqa: F401

from lpl_g1 import ROOT

ADAPTER = "lpl_g1.adapter:ROBOT"


def load_lock(root: Path = ROOT) -> dict:
    return generic.load_lock(root)


def evidence(name: str, out: Path, *, jobs: int | None = None, allow_pending=False) -> dict:
    return generic.evidence(ADAPTER, name, out, jobs=jobs, allow_pending=allow_pending)


def main(argv=None) -> int:
    return generic.main([ADAPTER, *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
