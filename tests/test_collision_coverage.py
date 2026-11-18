# SPDX-License-Identifier: Apache-2.0
"""Keep collision-physics limits separate from geometric measurement coverage."""

import logging
import re

import mujoco
import numpy as np
import pytest

from livingproof_sim_mujoco.observations import Reach, RouteReach
from lpl_g1 import ROOT
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
def test_straight_and_corner_measure_the_same_fallback_bodies(keyframe, caplog):
    model = build().model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    mujoco.mj_forward(model, data)
    with caplog.at_level(logging.INFO):
        straight = Reach(model)
    assert any(
        record.name == "livingproof_sim_mujoco.observations"
        and record.levelno == logging.INFO
        and "visual geoms" in record.getMessage()
        for record in caplog.records
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        corner = RouteReach(model)
    assert any(
        record.name == "livingproof_sim_mujoco.observations"
        and record.levelno == logging.INFO
        and "visual geoms" in record.getMessage()
        for record in caplog.records
    )
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
def test_visual_only_torso_intrusion_is_not_invisible(reach_class, caplog):
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
    with caplog.at_level(logging.INFO):
        reach = reach_class(model)
    assert any(
        record.name == "livingproof_sim_mujoco.observations"
        and record.levelno == logging.INFO
        and "visual geoms" in record.getMessage()
        for record in caplog.records
    )
    assert reach.record["visual_fallback_bodies"] == ["torso"]
    span = {"id": model.geom("wall").id, 0: (-2, 2), 2: (-1, 3)}
    if reach_class is Reach:
        penetration = reach.toward(data, {"wall": {"span": span, "side": 1}})["wall"] - 0.5
    else:
        penetration = reach.past(
            data, {"wall": {"span": {**span, "axis": 1}, "side": -1, "inner": 0.5}}
        )["wall"]
    assert penetration == pytest.approx(0.1)


def test_documented_body_census_matches_the_model():
    model = build().model
    report = {body["body"]: body for body in census(model)["bodies"]}
    manual = (ROOT / "docs/collision-coverage.md").read_text()
    rows = [line.split("|")[1:4] for line in manual.splitlines() if line.startswith("| `")]
    documented = []
    for names, representation, _ in rows:
        mode = "visual_fallback" if "visual" in representation.lower() else "collision_class"
        for name in re.findall(r"`([^`]+)`", names):
            documented.append(name)
            assert report[name]["measurement"] == mode, name
            if "Three" in representation:
                assert report[name]["measurement_geoms"] == 3
            if "Six" in representation:
                assert report[name]["measurement_geoms"] == 6
    assert len(documented) == len(set(documented)) == model.nbody - 1
    assert set(documented) == set(report)


def test_publisher_contact_pairs_are_preserved():
    model = build().model
    expected = {
        frozenset(("floor", "left_foot")),
        frozenset(("floor", "right_foot")),
        frozenset(("left_thigh", "left_hand_collision")),
        frozenset(("left_foot", "right_foot")),
        frozenset(("right_thigh", "right_hand_collision")),
    }
    actual = {
        frozenset((model.geom(int(a)).name, model.geom(int(b)).name))
        for a, b in zip(model.pair_geom1, model.pair_geom2)
    }
    assert model.npair == len(expected)
    assert actual == expected


def test_robot_collision_masks_and_pair_thresholds_are_zero():
    model = build().model
    robot = model.geom_bodyid != 0
    assert robot.sum() == 71
    for values in (model.geom_contype, model.geom_conaffinity, model.geom_margin, model.geom_gap):
        np.testing.assert_array_equal(values[robot], 0)
    np.testing.assert_array_equal(model.pair_margin, 0)
    np.testing.assert_array_equal(model.pair_gap, 0)
