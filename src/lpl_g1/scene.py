# SPDX-License-Identifier: Apache-2.0
"""The Playground G1 scene plus the pinned hallway world.

The Playground MJCF is loaded unchanged: its solver options and its feet-only collision
pairs are not touched. The world's walls are added as visual-only boxes (contype 0,
conaffinity 0), so they create no contacts; wall clearance is checked geometrically
(see ``clearance``). Named places become sites, read back from the compiled model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from lpl_g1 import ROOT
from lpl_g1.fetch import load_pins, local

SCENE = "mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"
WORLD = "worlds/hallway-open/0.1.0/world.json"
PLACE = "place:"


@dataclass
class Scene:
    model: mujoco.MjModel
    world: dict
    position_kp: dict[str, float]  # published <position> gains, by actuator name
    ctrlrange: dict[str, tuple[float, float]]
    walls: list[str]


def load_world(root: Path = ROOT) -> dict:
    return json.loads(local("worlds", WORLD, root).read_text())


def build(actuation: str = "pd_torque", root: Path = ROOT) -> Scene:
    """Compile the scene. ``pd_torque`` turns every <position> actuator into a <motor>."""
    load_pins(root)
    spec = mujoco.MjSpec.from_file(str(local("playground", SCENE, root)))
    world = load_world(root)
    walls = []
    for geom in world["geometry"]:
        if geom["role"] != "wall":
            continue
        g = spec.worldbody.add_geom()
        g.name, g.type = geom["name"], mujoco.mjtGeom.mjGEOM_BOX
        g.size = [s / 2 for s in geom["size"]]
        g.pos = geom["centre"]
        g.contype = g.conaffinity = 0
        g.rgba = [0.55, 0.6, 0.7, 0.45]
        walls.append(geom["name"])
    for place in world["places"]:
        site = spec.worldbody.add_site()
        site.name = PLACE + place["name"]
        site.pos = place.get("position") or place.get("reference") or [0.0, 0.0, 0.0]
        yaw = math.radians(place.get("yaw_deg", 0.0))
        site.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        site.size = [0.03, 0.03, 0.03]
        site.rgba = [0.1, 0.6, 0.2, 0.5]
    position_kp, ctrlrange = {}, {}
    compiled = spec.compile()
    for i in range(compiled.nu):
        name = compiled.actuator(i).name
        position_kp[name] = float(compiled.actuator_gainprm[i][0])
        ctrlrange[name] = tuple(float(v) for v in compiled.actuator_ctrlrange[i])
    if actuation == "pd_torque":
        for act in spec.actuators:
            act.set_to_motor()
            act.inheritrange = 0
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        # torque limits stay those of each joint's actuatorfrcrange
    elif actuation != "position":
        raise ValueError(f"unknown actuation {actuation!r}")
    return Scene(spec.compile(), world, position_kp, ctrlrange, walls)


def place_pose(model: mujoco.MjModel, name: str) -> tuple[np.ndarray, float]:
    """Position and yaw of a named world place, read from the compiled model."""
    site = model.site(PLACE + name)
    w, _, _, z = model.site_quat[site.id]
    return model.site_pos[site.id].copy(), 2 * math.atan2(z, w)
