# SPDX-License-Identifier: Apache-2.0
"""The Playground G1 scene.

The Playground MJCF is loaded unchanged: its solver options and its feet-only collision
pairs are not touched. This package does not add a world. ``walls`` is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from lpl_g1 import ROOT
from livingproof_sim_mujoco import scene as generic
from livingproof_sim_mujoco.pins import load_pins, local

SCENE = "mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"


@dataclass
class Scene:
    model: mujoco.MjModel
    position_kp: dict[str, float]  # published <position> gains, by actuator name
    ctrlrange: dict[str, tuple[float, float]]
    walls: list[str]


def build(actuation: str = "pd_torque", root: Path = ROOT) -> Scene:
    """Compile the robot scene. ``pd_torque`` turns every <position> actuator into a <motor>."""
    load_pins(root)
    spec = mujoco.MjSpec.from_file(str(local("playground", SCENE, root)))
    position_kp, ctrlrange = generic.actuator_table(spec.compile())
    if actuation == "pd_torque":
        generic.to_pd_torque(spec)
    elif actuation != "position":
        raise ValueError(f"unknown actuation {actuation!r}")
    return Scene(spec.compile(), position_kp, ctrlrange, [])
