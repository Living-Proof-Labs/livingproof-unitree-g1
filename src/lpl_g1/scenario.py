# SPDX-License-Identifier: Apache-2.0
"""walk-hallway@0.1.0: parameters and preconditions resolved from the robot, configuration
and world. Evidence only: grading is the capability verifier's (see verification.lock.json)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from lpl_g1 import ROOT
from lpl_g1.profiles import Unsupported, configuration, robot
from lpl_g1.scene import load_world

SCENARIO = "walk-hallway@0.1.0"
COMMAND_SPEED = 0.5  # m/s, shared
WALL_CLEARANCE = {"humanoid": 0.05, "quadruped": 0.10}
SPAWN, EXIT = "spawn-entrance", "exit"


def place(world: dict, name: str) -> dict:
    return next(p for p in world["places"] if p["name"] == name)


def parameters(name: str, root: Path = ROOT) -> dict:
    """Acceptance parameters resolved from the scenario, robot, configuration and world."""
    rob, (conf, _) = robot(root), configuration(name, root)
    world = load_world(root)
    route = place(world, EXIT)["reference"][0] - place(world, SPAWN)["position"][0]
    clear = world["dimensions"]["clear_width"]
    width = rob["dimensions"]["width"]["value"]
    return {
        "command_speed": COMMAND_SPEED,
        "route_length": route,
        "time_limit": 2 * route / COMMAND_SPEED + 5,
        "fall_height": conf["locomotion"]["fall_base_height"]["value"],
        "wall_clearance": WALL_CLEARANCE[rob["class"]],
        "lateral_limit": round(clear / 2 - width / 2, 6),
        "exit_x": place(world, EXIT)["reference"][0],
        "spawn": SPAWN,
    }


def preconditions(name: str, root: Path = ROOT) -> list[dict]:
    """Checked before any trial and recorded in the evidence manifest."""
    rob = robot(root)
    world = load_world(root)
    width = rob["dimensions"]["width"]["value"]
    clear = world["dimensions"]["clear_width"]
    out = []
    need = width + 2 * WALL_CLEARANCE[rob["class"]]
    out.append(
        ("fits_width", need <= clear, f"{width} + 2 x clearance = {need:.2f} m vs {clear} m")
    )
    try:
        conf, _ = configuration(name, root)
        out.append(("configuration_offered", True, name))
        iface = conf["controller"]["interface"]
        out.append(("velocity_interface", "velocity_command" in iface, f"interface {iface}"))
    except Unsupported as e:
        out.append((e.precondition, False, e.detail))
    out.append(
        ("simulator_declared", "mujoco" in rob["simulators"], f"simulators {rob['simulators']}")
    )
    return [{"name": n, "met": bool(ok), "detail": d} for n, ok, d in out]


def git_head(path: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--", ".", ":!artifacts"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return head, dirty


def dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
