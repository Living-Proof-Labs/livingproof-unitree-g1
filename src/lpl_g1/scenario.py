# SPDX-License-Identifier: Apache-2.0
"""walk-hallway@0.1.0 parameters and preconditions for the G1 (livingproof-sim-mujoco's
generic scenario on this package's profiles and world). Evidence only."""

from __future__ import annotations

import json
from pathlib import Path

from livingproof_sim_mujoco import scenario as generic
from livingproof_sim_mujoco.scenario import (  # noqa: F401
    COMMAND_SPEED,
    EXIT,
    SCENARIO,
    SPAWN,
    WALL_CLEARANCE,
    git_head,
    place,
)

from lpl_g1 import ROOT
from lpl_g1.profiles import configuration, robot
from lpl_g1.scene import load_world


def parameters(name: str, root: Path = ROOT) -> dict:
    return generic.parameters(robot(root), configuration(name, root)[0], load_world(root))


def preconditions(name: str, root: Path = ROOT) -> list[dict]:
    return generic.preconditions(
        robot(root), load_world(root), name, lambda n: configuration(n, root)[0]
    )


def dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
