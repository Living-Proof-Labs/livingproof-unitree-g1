# SPDX-License-Identifier: Apache-2.0
"""The Playground G1 scene plus the pinned hallway world.

The Playground MJCF is loaded unchanged: its solver options and its feet-only collision
pairs are not touched. The world's walls are added as visual-only boxes (contype 0,
conaffinity 0), so they create no contacts; wall clearance is checked geometrically
(see ``clearance``). Named places become sites, read back from the compiled model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from livingproof_sim_mujoco import scene as generic
from livingproof_sim_mujoco.scene import PLACE, WORLD, place_pose  # noqa: F401
import mujoco

from lpl_g1 import ROOT
from livingproof_sim_mujoco.pins import load_pins, local  # noqa: F401

SCENE = "mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"


@dataclass
class Scene:
    model: mujoco.MjModel
    world: dict
    position_kp: dict[str, float]  # published <position> gains, by actuator name
    ctrlrange: dict[str, tuple[float, float]]
    walls: list[str]


def load_world(root: Path = ROOT) -> dict:
    return generic.load_world(root)


def build(actuation: str = "pd_torque", root: Path = ROOT) -> Scene:
    """Compile the scene. ``pd_torque`` turns every <position> actuator into a <motor>."""
    load_pins(root)
    spec = mujoco.MjSpec.from_file(str(local("playground", SCENE, root)))
    world = load_world(root)
    walls = generic.add_world(spec, world)
    position_kp, ctrlrange = generic.actuator_table(spec.compile())
    if actuation == "pd_torque":
        generic.to_pd_torque(spec)
    elif actuation != "position":
        raise ValueError(f"unknown actuation {actuation!r}")
    return Scene(spec.compile(), world, position_kp, ctrlrange, walls)
