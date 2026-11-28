# SPDX-License-Identifier: Apache-2.0
"""Adapter mechanics for the released shared corner, not a robot capability trial.

The pinned simulator accepts world format 0.3 (livingproof-sim-mujoco#89). These tests
deliberately exercise its lower-level composition primitive and the real adapter with a
recording policy, without claiming that the evidence runner accepts this fixture. Body
states at the bend/finish are assigned test inputs, not a simulated trajectory.
The original-policy tests below exercise both directions through the evidence runner;
recording-policy mechanics alone do not establish robot performance.
"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from livingproof_scenario_parameters import crosses_exit, route_length
from livingproof_sim_mujoco import compose, evidence, hallway, scenario

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
def test_new_id_is_declared(direction):
    identity = document(direction + ".json")["scenario"]["id"]
    result = scenario.scenario_supported_precondition(ROOT, identity)
    assert result["name"] == "scenario_supported"
    assert result["met"] is True


def test_unknown_id_refuses_and_historical_support_is_preserved():
    assert scenario.scenario_supported_precondition(ROOT, "not-declared")["met"] is False
    assert scenario.scenario_supported_precondition(ROOT, "walk-corner")["met"] is True


def test_pinned_runner_accepts_new_world_format():
    # The released simulator (livingproof-sim-mujoco#89) supports world format 0.3.
    # This does not claim the evidence runner has run this fixture; see the module docstring.
    assert hallway.supported(document("world.json")) is None


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
    # integration trial. compose.place is below the runner's format gate.
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


def run_original_policy_trial(direction, output):
    """One original policy-controlled simulation, with explicit immutable fixture inputs."""
    parameters = document(direction + ".json")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    configuration = next(c for c in manifest["configurations"] if c["name"] == CONFIGURATION)
    distribution = importlib.metadata.distribution("livingproof-sim-mujoco")
    runner_commit = json.loads(distribution.read_text("direct_url.json"))["vcs_info"]["commit_id"]
    sources = document("sources.json")
    lock = {
        "plan": {"trials": 1},
        "scenario": parameters["scenario"]["id"] + "@" + parameters["scenario"]["revision"],
        "world": {
            "id": parameters["world"]["id"],
            "version": parameters["world"]["content_version"],
            "sha256": sources["world.json"]["sha256"],
        },
        "configurations": [{k: configuration[k] for k in ("name", "path", "version", "sha256")}],
        "releases": {
            "runner": {"tag": "v" + distribution.version, "commit": runner_commit},
            "specs": {k: sources[direction + ".json"][k] for k in ("tag", "commit")},
        },
        "sources": {
            kind: {
                "path": (DATA / filename).relative_to(ROOT).as_posix(),
                "sha256": sources[filename]["sha256"],
            }
            for kind, filename in (("world", "world.json"), ("parameters", direction + ".json"))
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    lock_path = output / "caller.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return evidence.evidence(
        "lpl_g1.adapter:ROBOT", CONFIGURATION, output / "evidence", lock=lock_path
    )


@pytest.mark.parametrize("direction", ["left", "right"])
def test_original_policy_runs_each_declared_direction(tmp_path, direction):
    run = run_original_policy_trial(direction, tmp_path / direction)
    assert run["trials"]["count"] == 1
    assert len(run["trials"]["results"]) == 1
    assert run["trials"]["results"][0]["outcome"] == "crossed_exit"

    obs_path = tmp_path / direction / "evidence/trials/trial-01.json"
    assert obs_path.is_file()
    obs = json.loads(obs_path.read_text())

    world = document("world.json")
    robot_profile = json.loads((ROOT / "robot.json").read_text())
    conf_profile = json.loads((ROOT / f"configurations/{CONFIGURATION}.json").read_text())
    fall_height = conf_profile["locomotion"]["fall_base_height"]["value"]
    lateral_limit = (
        world["dimensions"]["clear_width"] / 2 - robot_profile["dimensions"]["width"]["value"] / 2
    )
    length = route_length(world, document(direction + ".json")["places"])
    time_limit = 2 * length / 0.5 + 5

    assert obs["end"] == "crossed_exit"
    assert obs["left_world"] is None
    assert obs["exit_time"] is not None and obs["exit_time"] <= time_limit
    assert obs["places_entered"]["corner"] is not None
    assert obs["places_entered"]["corner"] < obs["exit_time"]
    assert obs["wall_contacts"] == []
    assert obs["max_route_distance"]["value"] <= lateral_limit
    assert obs["min_base_height"]["value"] >= fall_height


@pytest.mark.parametrize("direction", ["left", "right"])
def test_configurable_route_control_mcap_outcome_and_pose_api(tmp_path, direction):
    from livingproof_sim_mujoco import observations
    from livingproof_sim_mujoco.media import Recorder
    from mcap.reader import make_reader

    world = document("world.json")
    parameters = document(direction + ".json")
    places = scenario.scenario_places(world, parameters)
    conf_profile = json.loads((ROOT / f"configurations/{CONFIGURATION}.json").read_text())
    fall_height = conf_profile["locomotion"]["fall_base_height"]["value"]
    robot_profile = json.loads((ROOT / "robot.json").read_text())
    lateral_limit = (
        world["dimensions"]["clear_width"] / 2 - robot_profile["dimensions"]["width"]["value"] / 2
    )
    length = route_length(world, parameters["places"])
    time_limit = 2 * length / 0.5 + 5

    model, data, ctrl, walls = compose.place(ROBOT, CONFIGURATION, world, places["start"]["name"])
    start_pose = ROBOT.get_pose(model, data, CONFIGURATION)
    expected_yaw = 0.0 if direction == "left" else -math.pi / 2
    assert start_pose.yaw == pytest.approx(expected_yaw, abs=1e-4)

    policy = ROBOT.policy(model, ctrl)
    mcap_path = tmp_path / f"{direction}.mcap"
    recorder = Recorder(model, ROBOT.base_body(CONFIGURATION), mcap_path=mcap_path, video_path=None)
    step = ROBOT.drive(
        CONFIGURATION,
        model,
        ctrl,
        policy,
        places,
        route_control={"mode": "feedback", "turn_direction": direction},
    )
    obs = observations.observe_route(
        model,
        data,
        step,
        base_body=ROBOT.base_body(CONFIGURATION),
        walls=walls,
        seconds=time_limit,
        route=places["route"],
        exit_place=places["exit"],
        through=places["through"],
        sample_hz=ROBOT.runner_settings()["sample_hz"],
        on_step=recorder,
    )
    recorder.close()

    assert obs["end"] == "crossed_exit"
    assert obs["wall_contacts"] == []
    assert obs["max_route_distance"]["value"] <= lateral_limit
    assert obs["min_base_height"]["value"] >= fall_height

    poses = []
    with mcap_path.open("rb") as handle:
        for _, channel, message in make_reader(handle).iter_messages(topics=["/base_pose"]):
            if channel.topic == "/base_pose":
                poses.append(json.loads(message.data))
    assert len(poses) >= 20
    xyz = []
    for msg in poses:
        pos = msg["pose"]["position"]
        assert pos["z"] >= fall_height
        xyz.append([pos["x"], pos["y"], pos["z"]])
    assert any(
        crosses_exit(a, b, places["route"]["points"], places["exit"])
        for a, b in zip(xyz[:-1], xyz[1:])
    )
