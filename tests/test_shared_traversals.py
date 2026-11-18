# SPDX-License-Identifier: Apache-2.0
"""Adapter mechanics for the released shared corner, not a robot capability trial.

The pinned simulator accepts world format 0.3. These tests validate the released
fixture and exercise composition and the real adapter with a recording policy.
Body states at the bend/finish are assigned test inputs, not a simulated trajectory.
Support declarations must wait for both directions through the evidence runner.
"""

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from livingproof_scenario_parameters import crosses_exit, route_length
from livingproof_sim_mujoco import compose, hallway, scenario

from lpl_g1 import ROOT
from lpl_g1.adapter import ROBOT

DATA = Path(__file__).parent / "data/reuse"
CONFIGURATION = "stock"


def document(name):
    return json.loads((DATA / name).read_bytes())


def test_released_fixture_bytes_and_provenance():
    sources = document("sources.json")
    for name, source in sources.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == source["sha256"]
        expected = (
            ("v1.4.0", "971b39ee845a18c50da627a7410d22e3b98171f9")
            if name in ("world.json", "release.json")
            else ("v3.1.0", "9ee830201f05943feb2b9787eea2ea747431d935")
        )
        assert (source["tag"], source["commit"]) == expected
    assert sources["world.json"]["sha256"] == document("release.json")["files"]["world.json"]
    world = document("world.json")
    assert world["format"] == "livingproof-world/0.3"
    for direction in ("left", "right"):
        assert document(direction + ".json")["world"] == {
            "id": world["id"],
            "content_version": world["content_version"],
        }


@pytest.mark.parametrize("direction", ["left", "right"])
def test_new_id_is_not_yet_declared(direction):
    identity = document(direction + ".json")["scenario"]["id"]
    result = scenario.scenario_supported_precondition(ROOT, identity)
    assert result["name"] == "scenario_supported"
    assert result["met"] is False
    assert identity in result["detail"]


def test_unknown_id_refuses_and_historical_support_is_preserved():
    assert scenario.scenario_supported_precondition(ROOT, "not-declared")["met"] is False
    assert scenario.scenario_supported_precondition(ROOT, "walk-corner")["met"] is True


def test_pinned_runner_accepts_shared_world_without_changing_it():
    world = document("world.json")
    original = json.dumps(world, sort_keys=True)
    hallway.supported(world)
    assert json.dumps(world, sort_keys=True) == original


def test_pinned_runner_still_refuses_unknown_world_format():
    world = document("world.json")
    world["format"] = "livingproof-world/999"
    with pytest.raises(hallway.Unsupported, match="livingproof-world/999"):
        hallway.supported(world)


def test_pinned_runner_refuses_unknown_traversal_place():
    world = document("world.json")
    world["traversals"][0]["start"] = "missing-start"
    with pytest.raises(hallway.Unsupported, match="missing-start"):
        hallway.supported(world)


class RecordingPolicy:
    def __init__(self):
        self.commands = []

    def step(self, data, command):
        self.commands.append(np.array(command))


def set_state(model, data, xy, yaw):
    data.qpos[:2] = xy
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_actual_adapter_spawn_drive_and_finish(direction):
    world = document("world.json")
    parameters = document(direction + ".json")
    places = scenario.scenario_places(world, parameters)
    traversal = next(
        t
        for t in world["traversals"]
        if t["name"] == ("forward" if direction == "left" else "reverse")
    )
    assert [places[k]["name"] for k in ("start", "route", "exit")] == [
        traversal[k] for k in ("start", "route", "finish")
    ]
    assert places["through"][0]["name"] == "corner"
    yaw = 0.0 if direction == "left" else -math.pi / 2
    start = [-2.5, 0.0] if direction == "left" else [0.0, 2.5]
    # Real pinned robot model/controller initial pose; no policy inference or
    # integration trial. The released world is validated by the pinned runner.
    hallway.supported(world)
    model, data, ctrl, walls = compose.place(ROBOT, CONFIGURATION, world, places["start"]["name"])
    assert set(walls) == {g["name"] for g in world["geometry"] if g["role"] == "wall"}
    assert data.qpos[:2] == pytest.approx(start)
    body = model.body(ROBOT.base_body(CONFIGURATION)).id
    w, x, y, z = data.xquat[body]
    actual_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    assert actual_yaw == pytest.approx(yaw)
    policy = RecordingPolicy()
    step = ROBOT.drive(CONFIGURATION, model, ctrl, policy, places)
    step(data)
    vx, vy, wz = policy.commands[-1]
    assert vx > 0 and vy == pytest.approx(0, abs=1e-6)
    assert wz == pytest.approx(0, abs=1e-6)
    # Command in world coordinates: east for left, south for right. This is a
    # requested velocity, not measured locomotion of a policy-controlled robot.
    world_velocity = [
        math.cos(yaw) * vx - math.sin(yaw) * vy,
        math.sin(yaw) * vx + math.cos(yaw) * vy,
    ]
    assert world_velocity == pytest.approx([vx, 0] if direction == "left" else [0, -vx])
    bend = [-0.1, 0] if direction == "left" else [0, 0.1]
    set_state(model, data, bend, yaw)
    step(data)
    turn_vx, _, turn = policy.commands[-1]
    assert turn > 0 if direction == "left" else turn < 0
    assert 0 <= turn_vx < vx
    finish_yaw = math.pi / 2 if direction == "left" else -math.pi
    before, after = (
        ([0, 2.9, 0], [0, 3.1, 0]) if direction == "left" else ([-2.9, 0, 0], [-3.1, 0, 0])
    )
    set_state(model, data, before[:2], finish_yaw)
    step(data)
    end_vx, end_vy, end_wz = policy.commands[-1]
    assert end_vx > 0
    assert end_vy == pytest.approx(0, abs=1e-6)
    assert end_wz == pytest.approx(0, abs=1e-6)
    points = places["route"]["points"]
    assert crosses_exit(before, after, points, places["exit"])
    assert not crosses_exit(after, before, points, places["exit"])
    assert route_length(world, parameters["places"]) == pytest.approx(5.5)
    assert len(policy.commands) == 3
