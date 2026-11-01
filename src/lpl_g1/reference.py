# SPDX-License-Identifier: Apache-2.0
"""Our runner against each publisher's own control loop, tick by tick.

playground: play_g1_joystick.py ``OnnxController`` (mujoco_playground 12ad7f3). It is
  verbatim except the gamepad is a constant command. The control callback runs inside
  ``mj_step`` on the Playground MJCF's <position> actuators, starting from keyframe
  knees_bent.
stock: deploy_mujoco.py's loop with g1.yaml (unitree_rl_gym 276801e): torque PD, then
  ``mj_step``, ``counter += 1``, then every ``control_decimation`` steps the observation,
  the TorchScript policy and new targets. Two adaptations are needed for the 29-DoF
  Playground model: the 12 leg joints are addressed by name (upstream ``qpos[7:]`` is its
  12-DoF model), and waist and arms are held with deploy_real g1.yaml's arm_waist PD,
  written into the same physics step as the legs' PD. Gains, scales, decimation, phase
  and ordering come from the pinned g1.yaml and deploy_mujoco.py.

Both sides start from the same state on the same compiled model and get the same command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from lpl_g1 import ROOT
from livingproof_sim_mujoco.pins import local
from lpl_g1.profiles import configuration, controller
from lpl_g1.scene import build
from livingproof_sim_mujoco.walk import initial_state, runner_settings

DEPLOY_MUJOCO_YAML = "deploy/deploy_mujoco/configs/g1.yaml"


# ---------------------------------------------------------------- playground
class OnnxController:
    """play_g1_joystick.py OnnxController; the Gamepad is replaced by ``command``."""

    def __init__(self, policy_path, default_angles, ctrl_dt, n_substeps, command, action_scale=0.5):
        import onnxruntime as rt

        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._counter = 0
        self._n_substeps = n_substeps
        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt
        self._command = np.asarray(command, dtype=np.float64)
        self.log = []

    def get_obs(self, model, data):
        linvel = data.sensor("local_linvel_pelvis").data
        gyro = data.sensor("gyro_pelvis").data
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        obs = np.hstack(
            [
                linvel,
                gyro,
                gravity,
                self._command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            ]
        )
        return obs.astype(np.float32)

    def get_control(self, model, data):
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            pred = self._policy.run(self._output_names, {"obs": obs.reshape(1, -1)})[0][0]
            self._last_action = pred.copy()
            data.ctrl[:] = pred * self._action_scale + self._default_angles
            self.log.append((self._counter - 1, obs, pred.copy()))
            phase_tp1 = self._phase + self._phase_dt
            self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


PLAYGROUND_TERMS = [
    ("linvel", 3),
    ("gyro", 3),
    ("gravity", 3),
    ("command", 3),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("last_action", 29),
    ("phase", 4),
]
STOCK_TERMS = [
    ("base_angvel", 3),
    ("gravity", 3),
    ("command", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("last_action", 12),
    ("phase", 2),
]


def playground_reference(command, ticks, root=ROOT):
    scene = build("position", root)
    model = scene.model
    data = mujoco.MjData(model)
    key = model.key("knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    c = controller("playground", scene, root, actuation="position")
    pol = OnnxController(str(c.policy_path), np.array(model.key_qpos[key][7:]), 0.02, 10, command)
    mujoco.set_mjcb_control(pol.get_control)
    try:
        while len(pol.log) < ticks:
            mujoco.mj_step(model, data)
    finally:
        mujoco.set_mjcb_control(None)
    return pol.log[:ticks], data.qpos.copy()


def playground_ours(command, ticks, root=ROOT, actuation="position"):
    scene = build(actuation, root)
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("knees_bent").id)
    c = controller("playground", scene, root, actuation=actuation)
    return _run_ours(model, data, c, command, ticks)


# ---------------------------------------------------------------- stock
def get_gravity_orientation(quaternion):  # deploy_mujoco.py, verbatim
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):  # deploy_mujoco.py, verbatim
    return (target_q - q) * kp + (target_dq - dq) * kd


def _stock_start(root):
    scene = build("pd_torque", root)
    model = scene.model
    data = mujoco.MjData(model)
    c = controller("stock", scene, root)
    zero = {k: 0.0 for k in runner_settings(root)["initial_noise"]}
    initial_state(model, data, c, 1, zero, "spawn-entrance")
    return scene, model, data, c


def stock_reference(command, ticks, root=ROOT, dtype=np.float32):
    """deploy_mujoco.py's loop. ``dtype=np.float64`` replaces only its float32 casts of
    kps, kds, default_angles and cmd_scale, isolating arithmetic precision from the loop."""
    import torch
    import yaml

    config = yaml.safe_load(local("rl_gym", DEPLOY_MUJOCO_YAML, root).read_text())
    _, m, d, c = _stock_start(root)
    s = c.configuration["settings"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=dtype)
    kds = np.array(config["kds"], dtype=dtype)
    default_angles = np.array(config["default_angles"], dtype=dtype)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=dtype)
    num_actions, num_obs = config["num_actions"], config["num_obs"]
    cmd = np.array(command, dtype=np.float32)
    assert np.isclose(m.opt.timestep, simulation_dt)
    # adaptation 1: the 12 leg joints by name (upstream qpos[7:] / qvel[6:] / ctrl[:])
    legs = [m.joint(j) for j in s["leg_joints"]]
    qa, va = [j.qposadr[0] for j in legs], [j.dofadr[0] for j in legs]
    by_joint = {m.actuator_trnid[i][0]: i for i in range(m.nu)}
    la = [by_joint[j.id] for j in legs]
    # adaptation 2: waist and arms held with deploy_real g1.yaml arm_waist gains
    dep = s["deploy"]
    arms = [m.joint(j) for j in s["arm_waist_joints"]]
    aq, av = [j.qposadr[0] for j in arms], [j.dofadr[0] for j in arms]
    aa = [by_joint[j.id] for j in arms]
    a_t, a_kp, a_kd = (
        np.array(dep[k], dtype=np.float64)
        for k in ("arm_waist_target", "arm_waist_kps", "arm_waist_kds")
    )

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    policy = torch.jit.load(str(c.policy_path), map_location="cpu")
    log = []
    while len(log) < ticks:
        tau = pd_control(target_dof_pos, d.qpos[qa], kps, np.zeros_like(kds), d.qvel[va], kds)
        d.ctrl[la] = tau
        d.ctrl[aa] = a_kp * (a_t - d.qpos[aq]) - a_kd * d.qvel[av]
        mujoco.mj_step(m, d)
        counter += 1
        if counter % control_decimation == 0:
            qj = d.qpos[qa]
            dqj = d.qvel[va]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]
            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale
            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)
            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles
            # the step index of the first physics step that uses these targets
            log.append((counter, obs.copy(), np.asarray(action).copy()))
    return log, d.qpos.copy()


def stock_ours(command, ticks, root=ROOT):
    _, model, data, c = _stock_start(root)
    return _run_ours(model, data, c, command, ticks)


# ---------------------------------------------------------------- comparison
def _run_ours(model, data, c, command, ticks):
    """Our runner; each tick is logged with the physics step it acts in."""
    from livingproof_sim_mujoco.policy import OnnxPolicy

    policy = OnnxPolicy(model, c.policy_path, c.spec)
    log, backend = [], policy.backend

    def recording(obs):
        out = backend(obs)
        log.append((policy.counter, np.asarray(obs).copy(), np.asarray(out).copy()))
        return out

    policy.backend = recording
    while len(log) < ticks:
        policy.step(data, command)
    return log[:ticks], data.qpos.copy()


def compare(ref, ours, terms, ref_q, our_q) -> dict:
    first, worst = None, {"obs": 0.0, "action": 0.0}
    per_term = {n: 0.0 for n, _ in terms}
    for k, ((ri, ro, ra), (gi, go, ga)) in enumerate(zip(ref, ours)):
        d_obs, d_act = np.abs(ro - go), np.abs(np.ravel(ra) - np.ravel(ga))
        worst["obs"] = max(worst["obs"], float(d_obs.max()))
        worst["action"] = max(worst["action"], float(d_act.max()))
        o, by = 0, {}
        for name, n in terms:
            by[name] = float(d_obs[o : o + n].max())
            per_term[name] = max(per_term[name], by[name])
            o += n
        if first is None and (ri != gi or d_obs.max() > 0 or d_act.max() > 0):
            first = {
                "tick": k,
                "reference_step": int(ri),
                "runner_step": int(gi),
                "max_obs_diff_by_term": by,
                "max_action_diff": float(d_act.max()),
            }
    return {
        "ticks": len(ref),
        "first_policy_step": {"reference": int(ref[0][0]), "runner": int(ours[0][0])},
        "bit_exact": first is None,
        "first_divergence": first,
        "max_abs_diff": worst,
        "max_obs_diff_by_term": per_term,
        "final_base_xy": {
            "reference": [round(float(v), 4) for v in ref_q[:2]],
            "runner": [round(float(v), 4) for v in our_q[:2]],
        },
    }


def run(configuration_name: str, command, ticks: int, root=ROOT) -> dict:
    if configuration_name == "playground":
        ref, rq = playground_reference(command, ticks, root)
        ours, oq = playground_ours(command, ticks, root)
        out = compare(ref, ours, PLAYGROUND_TERMS, rq, oq)
        out["reference"] = "play_g1_joystick.py OnnxController @ mujoco_playground 12ad7f3"
        out["passed"] = out["bit_exact"]
        out["pass_rule"] = "bit-exact observations and actions on the <position> actuators"
        pd, pq = playground_ours(command, ticks, root, actuation="pd_torque")
        diag = compare(ref, pd, PLAYGROUND_TERMS, rq, pq)
        out["walk_actuation_pd_torque"] = {
            k: diag[k] for k in ("bit_exact", "max_abs_diff", "first_policy_step", "final_base_xy")
        }
    else:
        ref, rq = stock_reference(command, ticks, root)
        ours, oq = stock_ours(command, ticks, root)
        out = compare(ref, ours, STOCK_TERMS, rq, oq)
        out["reference"] = "deploy_mujoco.py loop + configs/g1.yaml @ unitree_rl_gym 276801e"
        f64, fq = stock_reference(command, ticks, root, dtype=np.float64)
        d64 = compare(f64, ours, STOCK_TERMS, fq, oq)
        out["float64_reference"] = {
            "note": "upstream's float32 kps/kds/default_angles/cmd_scale as float64; else verbatim",
            **{
                k: d64[k]
                for k in (
                    "first_policy_step",
                    "bit_exact",
                    "max_abs_diff",
                    "max_obs_diff_by_term",
                    "final_base_xy",
                )
            },
        }
        out["passed"] = (
            d64["first_policy_step"]["reference"] == d64["first_policy_step"]["runner"]
            and d64["max_abs_diff"]["action"] == 0.0
            and d64["max_abs_diff"]["obs"] <= 1e-12
        )
        out["pass_rule"] = (
            "same first policy step, and against the float64 reference identical actions and "
            "observations within 1e-12 (phase is accumulated by fmod in the runner and taken "
            "modulo in deploy_mujoco.py); the verbatim float32 difference is reported"
        )
    out["configuration"] = configuration_name
    out["configuration_sha256"] = configuration(configuration_name, root)[1]
    out["command"] = list(command)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configuration", action="append", default=None)
    ap.add_argument("--ticks", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "reference-compare.json")
    a = ap.parse_args(argv)
    res = {}
    for name in a.configuration or ["playground", "stock"]:
        for label, cmd in (("walk_0.5", (0.5, 0.0, 0.0)), ("stand_0", (0.0, 0.0, 0.0))):
            res[f"{name}/{label}"] = run(name, cmd, a.ticks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    ok = True
    for k, v in res.items():
        f = v["first_divergence"]
        print(
            f"{k}: {'PASS' if v['passed'] else 'FAIL'}; "
            f"{'bit-exact' if v['bit_exact'] else 'not bit-exact'} over {v['ticks']} ticks"
            f"; first step ref {v['first_policy_step']['reference']} runner "
            f"{v['first_policy_step']['runner']}; max diff obs {v['max_abs_diff']['obs']:.3g} "
            f"action {v['max_abs_diff']['action']:.3g}"
            + (f"; first divergence tick {f['tick']}" if f else "")
        )
        if "float64_reference" in v:
            print(f"    float64 reference: max diff {v['float64_reference']['max_abs_diff']}")
        ok &= v["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
