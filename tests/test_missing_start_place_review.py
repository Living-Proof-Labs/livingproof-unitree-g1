# SPDX-License-Identifier: Apache-2.0
"""A scenario start that is not a place in the composed world fails with a clear message.

Proposed by Vera in the livingproof-unitree-g1#49 review (livingproof-unitree-g1#37). This goes through the
simulator's own path, ``compose.place``, which composes a vendored world before calling
``initial_state``. The world is found by its places, not by name. A pose place in it
starts the robot exactly there. A place the world does not have stops the run as an
``INCOMPATIBLE`` ``SystemExit`` that names the place and does not tell the caller to
compose a world that was already composed.
"""

import json
from importlib.resources import files

import pytest
from livingproof_sim_mujoco import compose

from lpl_g1.adapter import ROBOT

NAMES = ("stock", "playground")
MISSING = "start-place-not-in-this-world"


def _world_and_pose():
    for release in sorted((files("livingproof_sim_mujoco") / "worlds").glob("*/*/world.json")):
        poses = [
            p
            for p in json.loads(release.read_text()).get("places", [])
            if p.get("type") == "pose" and abs(float(p["position"][0])) > 0.5
        ]
        if not poses:
            continue
        ident = {"id": release.parent.parent.name, "version": release.parent.name}
        try:
            return compose.vendored_world({"world": ident})[0], poses[0]
        except SystemExit:
            continue
    pytest.skip("the pinned simulator vendors no world with a pose place off the origin")


@pytest.fixture(scope="module")
def world_and_pose():
    return _world_and_pose()


@pytest.mark.parametrize("name", NAMES)
def test_a_pose_place_in_the_world_is_where_the_robot_starts(world_and_pose, name):
    world, pose = world_and_pose
    model, data, _ctrl, _walls = compose.place(ROBOT, name, world, pose["name"])
    base = data.xpos[model.body(ROBOT.base_body(name)).id]
    assert base[0] == pytest.approx(pose["position"][0], abs=1e-6)
    assert base[1] == pytest.approx(pose["position"][1], abs=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_a_start_place_missing_from_the_world_stops_as_incompatible(world_and_pose, name):
    world, _pose = world_and_pose
    assert MISSING not in {p["name"] for p in world["places"]}
    with pytest.raises(SystemExit) as caught:
        compose.place(ROBOT, name, world, MISSING)
    message = str(caught.value)
    assert message.startswith("INCOMPATIBLE"), message
    assert MISSING in message, message
    assert "no trial was recorded" in message.lower(), message
    assert "compose the world first" not in message.lower(), message
