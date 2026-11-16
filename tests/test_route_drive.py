# SPDX-License-Identifier: Apache-2.0
"""walk-corner on hallway-corner 0.1.0 with this adapter's drive() (robot-capability-specs#40, 11b).

livingproof-sim-mujoco v3.2.0 sends no command for walk-corner: a robot whose
manifest declares it is driven by its adapter's ``drive``. These tests run the real
scenario for every configuration and check the observation against the walk-corner
0.1.0 limits, and check that walk-hallway trials are byte-identical to the published
run 43402628666 (drive is not used there). This file names the world and the
scenarios, so it is on the scenario-neutral allowlist.
"""

import hashlib
import json
import platform
from pathlib import Path

import livingproof_sim_mujoco
import numpy as np
import pytest
from livingproof_sim_mujoco import evidence

from lpl_g1 import ROOT, route
from lpl_g1.adapter import ROBOT

PACKAGE = Path(livingproof_sim_mujoco.__file__).parent
RUNNER = {"tag": "v3.2.0", "commit": "4ff176028a0f20ce30f022d45796d098bd8f1025"}
SPECS = {"tag": "v2.3.0", "commit": "57317e617014a6df97bb7fba9d92a890af8d42e5"}
# Trial observation sha256 in livingproof-verification run 43402628666 (sim v2.0.3).
PUBLISHED_HALLWAY = {
    "stock": "16223272b2ddd3e13b4053e2b756a61b251c1c2ff2d858a6936efbefa4120a46",
    "playground": "48b6078f3deb763cd1595d727bff52baac95b335de96e57db1402ee6ece9840b",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _world(world_id: str) -> Path:
    return PACKAGE / f"worlds/{world_id}/0.1.0/world.json"


def _run(tmp_path: Path, scenario: str, world_id: str, name: str, **plan) -> tuple[dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    conf = next(c for c in manifest["configurations"] if c["name"] == name)
    lock = {
        "plan": {"trials": 1, **plan},
        "scenario": f"{scenario}@0.1.0",
        "world": {"id": world_id, "version": "0.1.0", "sha256": _sha(_world(world_id))},
        "configurations": [
            {
                "name": name,
                "path": conf["path"],
                "version": conf["version"],
                "sha256": _sha(ROOT / conf["path"]),
            }
        ],
        "releases": {"runner": RUNNER, "specs": SPECS},
    }
    path = tmp_path / "caller.lock.json"
    path.write_text(json.dumps(lock))
    out = tmp_path / "out"
    run = evidence.evidence("lpl_g1.adapter:ROBOT", name, out, lock=path)
    observation = json.loads((out / "trials/trial-01.json").read_text())
    return run, observation


def test_manifest_declares_walk_corner_and_the_adapter_drives():
    assert json.loads((ROOT / "manifest.json").read_text())["scenarios"] == [
        "walk-hallway",
        "walk-corner",
    ]
    assert callable(ROBOT.drive)


def _straight_then_left():
    return route._route([[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])


def test_on_a_straight_route_the_command_is_forward_only():
    xy, arc = _straight_then_left()
    vx, vy, wz = route.command(xy, arc, np.array([-2.5, 0.0]), 0.0)
    assert vx == pytest.approx(route.SPEED) and vy == pytest.approx(0.0) and wz == 0.0


def test_at_a_left_turn_the_command_turns_left_and_slows():
    xy, arc = _straight_then_left()
    vx, _, wz = route.command(xy, arc, np.array([-0.1, 0.0]), 0.0)
    assert wz > 0.0 and vx < route.SPEED


def test_off_the_route_the_command_steps_back_toward_it():
    xy, arc = _straight_then_left()
    _, vy, _ = route.command(xy, arc, np.array([-2.0, 0.2]), 0.0)
    assert -route.CROSS_MAX <= vy < 0.0  # the route is to the robot's right


def test_commands_stay_inside_the_published_command_range():
    stock = json.loads((ROOT / "configurations/stock.json").read_text())
    vx_max, vy_max, wz_max = stock["settings"]["deploy"]["max_cmd"]
    assert route.SPEED <= vx_max and route.CROSS_MAX <= vy_max and route.YAW_RATE_MAX <= wz_max


@pytest.mark.parametrize("name", ["stock", "playground"])
def test_walk_corner_is_completed_within_every_limit(tmp_path, name):
    run, obs = _run(tmp_path, "walk-corner", "hallway-corner", name)
    world = json.loads(_world("hallway-corner").read_text())
    width = json.loads((ROOT / "robot.json").read_text())["dimensions"]["width"]["value"]
    conf = json.loads((ROOT / f"configurations/{name}.json").read_text())
    route_length = 2.5 + 3.0  # spawn-entrance (-2.5, 0) to the exit reference (0, 3)
    time_limit = 2 * route_length / 0.5 + 5
    lateral_limit = world["dimensions"]["clear_width"] / 2 - width / 2
    assert run["trials"]["results"][0]["outcome"] == "crossed_exit"
    assert obs["end"] == "crossed_exit" and obs["left_world"] is None
    assert obs["exit_time"] <= time_limit
    assert "corner" in obs["places_entered"]
    assert obs["max_route_distance"]["value"] <= lateral_limit - 0.1  # keep a margin
    assert obs["wall_contacts"] == []
    assert obs["min_base_height"]["value"] >= conf["locomotion"]["fall_base_height"]["value"]


@pytest.mark.parametrize("name", sorted(PUBLISHED_HALLWAY))
def test_walk_hallway_never_calls_drive_and_matches_an_adapter_without_it(
    tmp_path, monkeypatch, name
):
    """walk-hallway sends its command, so drive is not called: the bytes are main's."""
    adapter = type(ROBOT)

    def forbidden(*args, **kwargs):
        raise AssertionError("drive was called for a scenario with a command")

    monkeypatch.setattr(adapter, "drive", forbidden)
    with_drive, _ = _run(tmp_path / "with", "walk-hallway", "hallway-open", name, command_speed=0.5)
    monkeypatch.delattr(adapter, "drive")
    without, _ = _run(tmp_path / "without", "walk-hallway", "hallway-open", name, command_speed=0.5)
    assert with_drive["trials"]["results"][0]["sha256"] == without["trials"]["results"][0]["sha256"]


@pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"),
    reason="the published run is x86_64; MuJoCo and torch rollouts differ by CPU architecture",
)
@pytest.mark.parametrize("name", sorted(PUBLISHED_HALLWAY))
def test_walk_hallway_trials_are_byte_identical_to_the_published_run(tmp_path, name):
    run, _ = _run(tmp_path, "walk-hallway", "hallway-open", name, command_speed=0.5)
    assert run["trials"]["results"][0]["sha256"] == PUBLISHED_HALLWAY[name]
