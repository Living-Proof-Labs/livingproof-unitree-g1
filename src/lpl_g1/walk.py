# SPDX-License-Identifier: Apache-2.0
"""walk-hallway@0.1.0 for one configuration profile, and a zero-command standing check.

The run manifest is resolved and written before any trial. Each trial records its
trajectory at the controller rate; metrics, trial outcomes and the grade are evaluated
from that recording afterwards (scenario.evaluate / scenario.grade).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import traceback

import mujoco
import numpy as np

from lpl_g1 import ROOT
from lpl_g1 import scenario as sc
from lpl_g1.clearance import Clearance
from lpl_g1.fetch import fetch, sha256
from lpl_g1.profiles import controller, joint_dofs
from lpl_g1.scene import build, place_pose


def runner_settings(root: Path = ROOT) -> dict:
    doc = json.loads((root / "runner.json").read_text())
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in doc.items()}


@dataclass
class Trial:
    seed: int
    outcome: str
    first_failing_metric: str | None
    first_failure_time: float | None
    metrics: dict
    end: str
    duration_s: float
    final_x: float
    min_wall_gap: float
    clip: str | None = None
    error: str | None = None
    samples: list = field(default_factory=list, repr=False)


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
    """Run one trial and return (samples [t, x, y, z, wall_gap], end, data, clip name)."""
    from livingproof_sim_mujoco.policy import OnnxPolicy

    rs = runner_settings(root)
    scene = build("pd_torque", root)
    model = scene.model
    ctrl = controller(name, scene, root)
    policy = OnnxPolicy(model, ctrl.policy_path, ctrl.spec)
    data = mujoco.MjData(model)
    initial_state(model, data, ctrl, seed, rs["initial_noise"], p["spawn"])
    clearance = Clearance(model, scene.walls)
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
            gap = min(clearance.gaps(data).values())
            samples.append([round(float(data.time), 3), x, y, z, gap])
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


def trial(name: str, seed: int, p: dict, clip: Path | None = None, root: Path = ROOT) -> Trial:
    command = [p["command_speed"], 0.0, 0.0]
    try:
        samples, end, clip_name = simulate(
            name, seed, command, p["time_limit"], p, clip=clip, root=root
        )
        if end == "non_finite_state":
            raise FloatingPointError("simulation state became non-finite")
    except Exception as e:  # a trial that cannot complete is ERROR, never FAIL
        return Trial(
            seed,
            "ERROR",
            None,
            None,
            {},
            "error",
            0.0,
            math.nan,
            math.nan,
            error="".join(traceback.format_exception_only(e)).strip(),
        )
    ev = sc.evaluate(samples, p)
    return Trial(
        seed=seed,
        outcome=ev["outcome"],
        first_failing_metric=ev["first_failing_metric"],
        first_failure_time=ev["first_failure_time"],
        metrics=ev["metrics"],
        end=end,
        duration_s=samples[-1][0],
        final_x=round(samples[-1][1], 4),
        min_wall_gap=round(min(s[4] for s in samples), 4),
        clip=clip_name,
        samples=[[round(v, 4) for v in s] for s in samples],
    )


def _trial(args):
    name, seed, p, out = args
    t = trial(name, seed, p, out / f"seed-{seed:02d}.mp4" if out else None)
    if out:
        sc.dump(
            out / "trajectories" / f"seed-{seed:02d}.json",
            {"seed": seed, "columns": ["t", "x", "y", "z", "wall_gap"], "samples": t.samples},
        )
    d = asdict(t)
    d.pop("samples")
    return d


def reason(t: dict, p: dict) -> str:
    if t["outcome"] == "ERROR":
        return t["error"]
    m, at = t["first_failing_metric"], t["first_failure_time"]
    text = {
        "reached_exit": f"did not cross exit within {p['time_limit']:.0f}s (final x {t['final_x']:.2f})",
        "time_to_exit": f"crossed exit at {t['metrics']['time_to_exit']}s > {p['time_limit']}s",
        "wall_contacts": "touched a side wall",
        "min_base_height": f"fell: base below {p['fall_height']} m",
        "max_abs_lateral": f"base beyond {p['lateral_limit']} m of the centreline",
    }[m]
    return f"{m}: {text} at t={at:.2f}s"


def walk(name: str, out: Path, *, record=True, jobs=None, allow_dirty=False, root: Path = ROOT):
    fetch(root)
    created = dt.datetime.now(dt.timezone.utc)
    run_id = f"walk-hallway-unitree-g1-{name}-{created:%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    pre = sc.preconditions(name, root)
    base = {
        "run_id": run_id,
        "scenario": sc.SCENARIO,
        "robot": "unitree-g1",
        "configuration": name,
        "created": created.isoformat(timespec="seconds"),
        "preconditions": pre,
    }
    if not all(c["met"] for c in pre):
        failed = next(c["name"] for c in pre if not c["met"])
        result = {
            **base,
            "configuration_sha256": None,
            "grade": "UNSUPPORTED",
            "unsupported_precondition": failed,
            "trials": [],
        }
        sc.dump(out / "result.json", result)
        return result
    manifest, trees = sc.manifest(name, run_id, root)
    if any(trees.values()) and not allow_dirty:
        raise SystemExit(f"uncommitted changes {trees}; commit first or pass --allow-dirty")
    sc.dump(out / "run-manifest.json", manifest)
    p = manifest["parameters"]
    work = [(name, s, p, out if record else None) for s in manifest["trials"]["seeds"]]
    with ProcessPoolExecutor(jobs or os.cpu_count()) as pool:
        trials = list(pool.map(_trial, work))
    outcomes = [t["outcome"] for t in trials]
    k, n = outcomes.count("PASS"), len(trials)
    result = {
        **base,
        "configuration_sha256": manifest["configuration"]["sha256"],
        "configuration_version": manifest["configuration"]["version"],
        "manifest_sha256": sha256(out / "run-manifest.json"),
        "source_trees": trees,
        "parameters": p,
        "passed": k,
        "trials": n,
        "errors": outcomes.count("ERROR"),
        "required_passes": manifest["trials"]["required_passes"],
        "pass_rate": k / n,
        "wilson95": sc.interval(k, n),
        "grade": sc.grade(outcomes, manifest["trials"]["required_passes"]),
        "failures": [
            {"seed": t["seed"], "outcome": t["outcome"], "reason": reason(t, p)}
            for t in trials
            if t["outcome"] != "PASS"
        ],
        "trial_results": [
            {
                "configuration": name,
                "configuration_sha256": manifest["configuration"]["sha256"],
                **t,
            }
            for t in trials
        ],
    }
    sc.dump(out / "result.json", result)
    return result


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
    ap.add_argument("command", choices=["walk", "stand", "compare"])
    ap.add_argument("--configuration", action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    art = ROOT / "artifacts"
    if a.command == "compare":
        res = [json.loads((art / c / "result.json").read_text()) for c in a.configuration]
        cmp = sc.compare(res)
        sc.dump(art / "comparison.json", cmp)
        for r in cmp["runs"]:
            print(
                f"{r['configuration']}: {r['passed']}/{r['trials']} {r['grade']} "
                f"Wilson95 {r['wilson95']}"
            )
        print(f"better: {cmp['better']} ({cmp['rule']})")
        return 0
    name = a.configuration[0]
    out = a.out or art / name
    if a.command == "stand":
        print(json.dumps(stand(name, out), indent=2))
        return 0
    r = walk(name, out, record=not a.no_record, jobs=a.jobs, allow_dirty=a.allow_dirty)
    if r["grade"] == "UNSUPPORTED":
        print(f"{name}: UNSUPPORTED ({r['unsupported_precondition']})")
        return 0
    print(
        f"{name} ({r['configuration_sha256'][:12]}): {r['passed']}/{r['trials']} passed, "
        f"Wilson 95% {r['wilson95']}, grade {r['grade']}"
    )
    for f in r["failures"]:
        print(f"  seed {f['seed']:2d}: {f['reason']}")
    return 0  # the grade is the result, not the exit code


if __name__ == "__main__":
    sys.exit(main())
