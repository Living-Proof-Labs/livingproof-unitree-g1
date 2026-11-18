# SPDX-License-Identifier: Apache-2.0
"""Keep collision-physics limits separate from geometric measurement coverage."""

import mujoco
import numpy as np
import pytest

from livingproof_sim_mujoco.observations import Reach, RouteReach
from lpl_g1.collision_audit import census
from lpl_g1.scene import build


def test_census_covers_every_pinned_model_body_without_changing_physics():
    model = build().model
    before = (
        model.geom_contype.copy(),
        model.geom_conaffinity.copy(),
        model.pair_geom1.copy(),
        model.pair_geom2.copy(),
    )
    report = census(model)
    assert len(report["bodies"]) == model.nbody - 1 == 30
    assert len(report["selection"]["visual_fallback_bodies"]) == 22
    assert report["selection"]["geoms"] == 55
    assert all(body["measurement_geoms"] > 0 for body in report["bodies"])
    assert {body["measurement"] for body in report["bodies"]} == {
        "collision_class",
        "visual_fallback",
    }
    assert "torso_link" in report["selection"]["visual_fallback_bodies"]
    for actual, expected in zip(
        (model.geom_contype, model.geom_conaffinity, model.pair_geom1, model.pair_geom2), before
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("keyframe", ["home", "knees_bent"])
def test_straight_and_corner_measure_the_same_fallback_bodies(keyframe):
    model = build().model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    mujoco.mj_forward(model, data)
    with pytest.warns(UserWarning, match="visual geoms"):
        straight = Reach(model)
    with pytest.warns(UserWarning, match="visual geoms"):
        corner = RouteReach(model)
    assert straight.robot == corner.robot
    assert straight.record == corner.record
    for body in straight.record["visual_fallback_bodies"]:
        geoms = [g for g in straight.robot if model.body(int(model.geom_bodyid[g])).name == body]
        assert geoms, body
        for g in geoms:
            for axis in range(3):
                extent = straight.extent(data, g, axis)
                assert np.isfinite(extent).all()
                assert extent[0] <= extent[1]
                assert corner.extent(data, g, axis) == extent


@pytest.mark.parametrize("reach_class", [Reach, RouteReach])
def test_visual_only_torso_intrusion_is_not_invisible(reach_class):
    # Minimal regression: paired feet establish collision group 3; an unpaired visual
    # torso in group 2 crosses the wall. This is a measurement test, not a simulation.
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom name="floor" type="plane" size="0 0 .1"/>
        <geom name="wall" type="box" size="2 .05 2" pos="0 .55 1"
              contype="0" conaffinity="0"/>
        <body name="robot" pos="0 0 1"><freejoint/>
          <geom name="foot" type="sphere" size=".05" group="3"
                contype="0" conaffinity="0"/>
          <body name="torso" pos="0 .5 0">
            <geom name="torso_visual" type="box" size=".1 .1 .1" group="2"
                  contype="0" conaffinity="0"/>
          </body>
        </body>
      </worldbody><contact><pair geom1="foot" geom2="floor"/></contact></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.warns(UserWarning, match="torso"):
        reach = reach_class(model)
    assert reach.record["visual_fallback_bodies"] == ["torso"]
    span = {"id": model.geom("wall").id, 0: (-2, 2), 2: (-1, 3)}
    if reach_class is Reach:
        penetration = reach.toward(data, {"wall": {"span": span, "side": 1}})["wall"] - 0.5
    else:
        penetration = reach.past(
            data, {"wall": {"span": {**span, "axis": 1}, "side": -1, "inner": 0.5}}
        )["wall"]
    assert penetration == pytest.approx(0.1)
