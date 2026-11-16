# SPDX-License-Identifier: Apache-2.0
"""A scenario start that is not a place in the composed world fails with a clear message.

Proposed by Vera in the livingproof-unitree-g1#49 review (livingproof-unitree-g1#37). This goes through the
simulator's own path, ``compose.place``, which composes the vendored world before calling
``initial_state``. The error must name the missing place and must not tell the caller to
compose a world that was already composed. A start place that is in the world still starts
there, away from the origin.
"""

import json
from pathlib import Path

import livingproof_sim_mujoco
import pytest
from livingproof_sim_mujoco import compose

from lpl_g1.adapter import ROBOT

WORLD = Path(livingproof_sim_mujoco.__file__).parent / "worlds/hallway-open/0.1.0/world.json"
NAMES = ("stock", "playground")


@pytest.fixture(scope="module")
def world():
    if not WORLD.exists():
        pytest.skip("this simulator release does not vendor hallway-open 0.1.0")
    return json.loads(WORLD.read_text())


@pytest.mark.parametrize("name", NAMES)
def test_a_start_place_in_the_world_is_used(world, name):
    model, data, _ctrl, _walls = compose.place(ROBOT, name, world, "spawn-entrance")
    place = next(p for p in world["places"] if p["name"] == "spawn-entrance")
    base = data.xpos[model.body(ROBOT.base_body(name)).id]
    assert base[0] == pytest.approx(place["position"][0], abs=1e-6)
    assert base[1] == pytest.approx(place["position"][1], abs=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_a_start_place_missing_from_the_world_fails_clearly(world, name):
    assert "spawn-nowhere" not in {p["name"] for p in world["places"]}
    with pytest.raises((KeyError, SystemExit, ValueError)) as caught:
        compose.place(ROBOT, name, world, "spawn-nowhere")
    message = caught.value.args[0] if caught.value.args else str(caught.value)
    assert "spawn-nowhere" in message, message
    assert "compose the world first" not in message.lower(), message
