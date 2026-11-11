# SPDX-License-Identifier: Apache-2.0
import io
import json

import mujoco
import numpy as np
import pytest
import yaml

from lpl_g1 import ROOT
from lpl_g1 import scenario as sc
from livingproof_sim_mujoco import pins
from livingproof_sim_mujoco.pins import ChecksumError, sha256
from lpl_g1.profiles import Unsupported, configuration, controller, robot, validate
from lpl_g1.scene import SCENE, build, place_pose
from livingproof_sim_mujoco.walk import initial_state

from lpl_g1.adapter import ROBOT


def local(group: str, path: str):
    return pins.local(group, path, ROOT)


def fetch(root=ROOT, opener=None):
    return pins.fetch(root, opener) if opener else pins.fetch(root)


CONFIGS = ["stock", "playground"]
SCENARIO_README = "specs/scenarios/walk-hallway/README.md"


def test_profiles_match_the_pinned_v2_0_2_schemas():
    pins = json.loads((ROOT / "pins.json").read_text())["groups"]
    assert pins["specs"]["tag"] == "v2.0.2"
    assert pins["specs"]["commit"].startswith("16524cb")
    rob = robot()  # validates
    assert rob["id"] == "unitree-g1" and rob["default_configuration"] in CONFIGS
    assert [c["name"] for c in rob["configurations"]] == CONFIGS
    for name in CONFIGS:
        conf, digest = configuration(name)  # validates
        assert digest == sha256(ROOT / f"configurations/{name}.json")
        assert conf["version"] == {"stock": "1.1.0", "playground": "1.0.0"}[name]
    with pytest.raises(ValueError, match="robot does not match"):
        validate("robot", {**rob, "extra": 1})


def test_artifacts_and_model_source_are_pinned_files():
    rob = robot()
    ms = rob["model_source"]
    assert sha256(local("playground", ms["path"])) == ms["sha256"]
    want = {"stock": ("torchscript", "Unitree Robotics"), "playground": ("onnx", "Google DeepMind")}
    for name in CONFIGS:
        c = configuration(name)[0]["controller"]
        assert (c["format"], c["publisher"]) == want[name]
        group = "rl_gym" if name == "stock" else "playground"
        assert sha256(local(group, c["artifact"]["path"])) == c["artifact"]["sha256"]


def test_stock_gains_equal_the_published_deploy_config():
    deploy = yaml.safe_load(local("rl_gym", "deploy/deploy_real/configs/g1.yaml").read_text())
    s = configuration("stock")[0]["settings"]
    assert {k: deploy[k] for k in s["deploy"]} == s["deploy"]
    c = controller("stock", build())
    assert list(c.spec.kp) == deploy["kps"] and list(c.spec.kd) == deploy["kds"]
    assert list(c.spec.default_pose) == deploy["default_angles"]
    assert [h[3] for h in c.spec.hold] == deploy["arm_waist_kps"]
    assert c.spec.decimation == 10


def test_playground_gains_and_limits_equal_the_published_mjcf():
    scene = build("position")
    m = scene.model
    s = configuration("playground")[0]["settings"]
    for i in range(m.nu):
        a, j = m.actuator(i).name, m.joint(m.actuator_trnid[i][0]).name
        assert s["gains"]["kp"][j] == scene.position_kp[a]
        assert s["limits"]["target_range_rad"][j] == list(scene.ctrlrange[a])
        assert s["limits"]["torque_limits_nm"]["value"][j] == list(m.jnt_actfrcrange[m.joint(j).id])
        assert s["gains"]["joint_damping"]["value"][j] == m.dof_damping[m.joint(j).dofadr[0]]


def test_scenario_parameters_follow_the_pinned_acceptance_table():
    text = local("specs", SCENARIO_README).read_text()
    for row in (
        "| `command_speed` | 0.5 m/s",
        "Route length 5.5 m",
        "| `trials` | 1 |",
        "exact nominal pose at `spawn-entrance`",
        "0.05 m humanoid",
    ):
        assert row in text, row
    assert "required_passes" not in text and "seeds" not in text
    p = sc.parameters("stock")
    assert (p["time_limit"], p["exit_x"], p["wall_clearance"]) == (27.0, 3.0, 0.05)
    assert p["lateral_limit"] == pytest.approx(0.75 - robot()["dimensions"]["width"]["value"] / 2)
    plan = json.loads((ROOT / "verification.lock.json").read_text())["plan"]
    # The specification default is 1. Published G1 reports use 3 identical trials.
    assert plan == {"trials": {"count": 3}, "command_speed": 0.5}
    assert "seeds" not in plan and "required_passes" not in plan


def test_preconditions_give_unsupported_before_trials():
    assert all(c["met"] for c in sc.preconditions("playground"))
    pre = {c["name"]: c["met"] for c in sc.preconditions("vendor-x")}
    assert pre["configuration_offered"] is False
    with pytest.raises(Unsupported):
        configuration("vendor-x")


def test_fetch_rejects_a_download_with_the_wrong_checksum(tmp_path):
    pins = {
        "groups": {
            "g": {
                "repository": "o/r",
                "commit": "c",
                "dest": "d",
                "files": {"f.txt": {"sha256": "0" * 64}},
            }
        }
    }
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    with pytest.raises(ChecksumError, match="f.txt"):
        fetch(tmp_path, opener=lambda url: io.BytesIO(b"tampered"))
    assert not (tmp_path / "d" / "f.txt").exists()


def test_scene_keeps_playground_solver_and_collisions():
    raw = mujoco.MjModel.from_xml_path(str(local("playground", SCENE)))
    scene = build()
    m = scene.model
    for attr in ("timestep", "iterations", "ls_iterations", "integrator", "disableflags"):
        assert getattr(m.opt, attr) == getattr(raw.opt, attr), attr

    def pairs(model):  # by name: added world geoms shift ids
        return [
            (model.geom(a).name, model.geom(b).name, *model.pair_friction[i], *model.pair_solref[i])
            for i, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2))
        ]

    assert pairs(m) == pairs(raw)
    robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in robot_geoms)
    assert scene.walls == ["wall-left", "wall-right"]


def test_spawn_comes_from_the_world_place():
    scene = build()
    world = {p["name"]: p for p in scene.world["places"]}
    pos, yaw = place_pose(scene.model, "spawn-entrance")
    np.testing.assert_allclose(pos, world["spawn-entrance"]["position"])
    assert yaw == pytest.approx(np.radians(world["spawn-entrance"]["yaw_deg"]))


def test_nominal_pose_is_exact_and_start_offset_is_one_declared_change():
    scene = build()
    m = scene.model
    for name in CONFIGS:
        c = controller(name, scene)
        d = mujoco.MjData(m)
        ROBOT.initial_state(m, d, c, "spawn-entrance")
        nominal = d.qpos.copy()
        np.testing.assert_allclose(d.qvel, 0.0)
        again = mujoco.MjData(m)
        initial_state(m, again, c, "spawn-entrance")
        np.testing.assert_array_equal(again.qpos, nominal)
        shifted = mujoco.MjData(m)
        ROBOT.initial_state(
            m, shifted, c, "spawn-entrance", start_offset={"base_xy_m": [0.0, 0.05]}
        )
        np.testing.assert_allclose(shifted.qpos[0], nominal[0])
        np.testing.assert_allclose(shifted.qpos[1], nominal[1] + 0.05)
        np.testing.assert_allclose(shifted.qpos[2:], nominal[2:])
        np.testing.assert_allclose(shifted.qvel, 0.0)


def test_in_step_torque_pd_reproduces_the_published_position_actuators():
    """Playground policy, 2 s of walking: pd_torque vs MuJoCo <position> actuators."""
    from livingproof_sim_mujoco.policy import OnnxPolicy

    tracks = []
    for actuation in ("position", "pd_torque"):
        scene = build(actuation)
        m, d = scene.model, mujoco.MjData(scene.model)
        c = controller("playground", scene, actuation=actuation)
        initial_state(m, d, c, "spawn-entrance")
        OnnxPolicy(m, c.policy_path, c.spec).run(d, 2.0, [0.5, 0.0, 0.0])
        tracks.append(d.qpos.copy())
    np.testing.assert_allclose(tracks[0], tracks[1], atol=1e-6)


def test_playground_matches_play_g1_joystick_tick_by_tick():
    from lpl_g1.reference import run

    for cmd in ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0)):
        r = run("playground", cmd, 100)
        assert r["first_policy_step"] == {"reference": 9, "runner": 9}
        assert r["bit_exact"], r["first_divergence"]


def test_stock_matches_deploy_mujoco_tick_by_tick():
    """First action at step 10 as deploy_mujoco.py; identical actions against its loop
    once its float32 gain and default-angle casts are float64."""
    from lpl_g1.reference import run

    r = run("stock", (0.5, 0.0, 0.0), 100)
    assert r["first_policy_step"] == {"reference": 10, "runner": 10}
    assert r["passed"], r["float64_reference"]
    assert r["max_abs_diff"]["action"] < 1e-4  # verbatim float32 loop


def test_position_initial_state_holds_the_default_pose():
    scene = build("position")
    m, d = scene.model, mujoco.MjData(scene.model)
    c = controller("playground", scene, actuation="position")
    initial_state(m, d, c, "spawn-entrance")
    np.testing.assert_array_equal(d.ctrl, c.spec.default_pose)
