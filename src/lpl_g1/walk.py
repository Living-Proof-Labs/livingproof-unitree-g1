# SPDX-License-Identifier: Apache-2.0
"""Shared simulation setup (runner settings, seeded initial state) and the zero-command
standing check. Walk-hallway trials are recorded by ``lpl_g1.evidence`` and graded by
the capability verifier, never here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from lpl_g1 import ROOT
from lpl_g1 import scenario as sc
from lpl_g1.fetch import fetch
from lpl_g1.profiles import controller, joint_dofs
from lpl_g1.scene import build, place_pose


def runner_settings(root: Path = ROOT) -> dict:
    doc = json.loads((root / "runner.json").read_text())
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in doc.items()}


def initial_state(model, data, ctrl, seed: int, noise: dict, spawn: str) -> None:
    """Spawn pose from the world's named place plus the seed's initial-state noise."""
    rng = np.random.default_rng(seed)
    qadr, vadr = joint_dofs(model)
    # fixed draw order, so every configuration sees identical noise for a seed
    dq = rng.uniform(-1, 1, len(qadr)) * noise["joint_pos_rad"]
    dv = rng.uniform(-1, 1, len(vadr)) * noise["joint_vel_rad_s"]
    dxy = rng.uniform(-1, 1, 2) * noise["base_xy_m"]
    dyaw = rng.uniform(-1, 1) * noise["base_yaw_rad"]
    dbase = rng.uniform(-1, 1, 6) * noise["base_vel_m_s"]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = ctrl.initial_qpos
    position, yaw = place_pose(model, spawn)
    data.qpos[0:2] = position[0:2] + dxy
    yaw += dyaw
    data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qpos[qadr] += dq
    data.qvel[vadr] = dv
    data.qvel[0:6] = dbase
    if ctrl.spec.actuation == "position":  # hold the default pose until the first tick,
        for name, q in zip(ctrl.spec.actuators, ctrl.spec.default_pose):  # as keyframe ctrl
            data.ctrl[model.actuator(name).id] = q
    mujoco.mj_forward(model, data)


def simulate(
    name: str,
    seed: int,
    command,
    seconds: float,
    p: dict,
    *,
    stop_at_exit=True,
    clip: Path | None = None,
    root: Path = ROOT,
):
    """Run one simulation and return (samples [t, x, y, z], end, clip name)."""
    from livingproof_sim_mujoco.policy import OnnxPolicy

    rs = runner_settings(root)
    scene = build("pd_torque", root)
    model = scene.model
    ctrl = controller(name, scene, root)
    policy = OnnxPolicy(model, ctrl.policy_path, ctrl.spec)
    data = mujoco.MjData(model)
    initial_state(model, data, ctrl, seed, rs["initial_noise"], p["spawn"])
    base = model.body("pelvis").id
    every_sample = int(round(1 / (rs["sample_hz"] * model.opt.timestep)))
    renderer = None
    if clip is not None:
        renderer = mujoco.Renderer(model, 288, 512)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = base
        camera.distance, camera.elevation, camera.azimuth = 3.6, -38.0, 200.0
        frames, every_frame = [], int(round(1 / (25 * model.opt.timestep)))
    samples, end = [], "time_limit"
    for i in range(int(round(seconds / model.opt.timestep)) + 1):
        if renderer is not None and i % every_frame == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())
        if i % every_sample == 0:
            if not np.all(np.isfinite(data.qpos)):
                end = "non_finite_state"
                break
            x, y, z = (float(v) for v in data.xpos[base])
            samples.append([round(float(data.time), 3), x, y, z])
            if stop_at_exit and x >= p["exit_x"]:
                end = "crossed_exit"
                break
            if z < p["fall_height"]:
                end = "fell"  # outcome decided; nothing later can change it
                break
        policy.step(data, command)
    clip_name = None
    if renderer is not None:
        import imageio.v2 as imageio

        clip.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(clip, frames, fps=25, codec="libx264", quality=7, macro_block_size=8)
        renderer.close()
        clip_name = clip.name
    return samples, end, clip_name


def stand(name: str, out: Path | None = None, root: Path = ROOT) -> dict:
    """Zero-command standing check, seed 1; not part of the scenario grade."""
    fetch(root)
    p = sc.parameters(name, root)
    seconds = runner_settings(root)["stand_seconds"]
    samples, end, clip = simulate(
        name,
        1,
        [0.0, 0.0, 0.0],
        seconds,
        p,
        stop_at_exit=False,
        clip=out / "stand-zero-command.mp4" if out else None,
        root=root,
    )
    zs = [s[3] for s in samples]
    last = samples[-30:]  # final 0.6 s
    result = {
        "check": "zero-command-stand",
        "configuration": name,
        "configuration_sha256": sc.configuration(name, root)[1],
        "seconds": seconds,
        "seed": 1,
        "standing": end == "time_limit" and min(zs) >= p["fall_height"],
        "end": end,
        "min_base_height": round(min(zs), 4),
        "final_base_height_mean": round(sum(s[3] for s in last) / len(last), 4),
        "drift_m": round(math.dist(samples[0][1:3], samples[-1][1:3]), 4),
        "max_abs_lateral": round(max(abs(s[2]) for s in samples), 4),
        "clip": clip,
    }
    if out:
        sc.dump(out / "stand.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["stand"])
    ap.add_argument("--configuration", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or ROOT / "artifacts" / a.configuration
    print(json.dumps(stand(a.configuration, out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
