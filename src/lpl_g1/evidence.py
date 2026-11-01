# SPDX-License-Identifier: Apache-2.0
"""walk-hallway sim evidence for the capability verifier. Records observations; never grades.

Reads the committed verification.lock.json (trial plan and release refs) and, for one
configuration, runs every planned seed through the runner's ``hallway.observe``. It writes
one observation per trial and a ``livingproof-hallway-evidence/1.0.0`` manifest that echoes
the lock. The verifier assesses the trials and the capability report grades them.

    python -m lpl_g1.evidence <configuration> <out> [--allow-pending]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
from importlib import metadata
import json
import os
from pathlib import Path
import sys

import mujoco

from lpl_g1 import ROOT
from lpl_g1 import scenario as sc
from lpl_g1.fetch import fetch, sha256
from lpl_g1.profiles import controller
from lpl_g1.scene import build, load_world
from lpl_g1.walk import initial_state, runner_settings

ROBOT = "unitree-g1"
BASE_BODY = "pelvis"
ACTUATION = "pd_torque"
FORMAT = "livingproof-hallway-evidence/1.0.0"
PENDING = "PENDING_RELEASE"
LOCK = "verification.lock.json"


def load_lock(root: Path = ROOT) -> dict:
    return json.loads((root / LOCK).read_text())


def pending(lock: dict) -> list[str]:
    """Release entries that still say PENDING_RELEASE."""
    return sorted(
        name
        for name, ref in lock["releases"].items()
        if PENDING in (ref.get("tag"), ref.get("commit"))
    )


def locked_configuration(lock: dict, name: str, root: Path = ROOT) -> dict:
    entry = next((c for c in lock["configurations"] if c["name"] == name), None)
    if entry is None:
        raise SystemExit(f"{name} is not in {LOCK}")
    if sha256(root / entry["path"]) != entry["sha256"]:
        raise SystemExit(f"{entry['path']} differs from {LOCK}; update the lock in the same commit")
    return entry


def _trial(args):
    from livingproof_sim_mujoco import hallway
    from livingproof_sim_mujoco.policy import OnnxPolicy

    name, seed, plan, p, out = args
    rs = runner_settings(ROOT)
    scene = build(ACTUATION, ROOT)
    model = scene.model
    ctrl = controller(name, scene, ROOT)
    policy = OnnxPolicy(model, ctrl.policy_path, ctrl.spec)
    data = mujoco.MjData(model)
    initial_state(model, data, ctrl, seed, rs["initial_noise"], p["spawn"])
    command = [plan["command_speed"], 0.0, 0.0]
    obs = hallway.observe(
        model,
        data,
        lambda d: policy.step(d, command),
        base_body=BASE_BODY,
        walls=scene.walls,
        exit_x=p["exit_x"],
        seconds=p["time_limit"],
        sample_hz=rs["sample_hz"],
    )
    path = out / "trials" / f"seed-{seed:02d}.json"
    path.write_text(json.dumps(obs, separators=(",", ":")) + "\n")
    return {"seed": seed, "path": f"trials/{path.name}", "sha256": sha256(path)}


def evidence(name: str, out: Path, *, jobs: int | None = None, allow_pending=False) -> dict:
    lock = load_lock()
    waiting = pending(lock)
    if waiting and not allow_pending:
        raise SystemExit(f"{LOCK} has PENDING_RELEASE refs ({', '.join(waiting)}); run task repin")
    entry = locked_configuration(lock, name)
    fetch(ROOT)
    plan = lock["plan"]
    world = load_world(ROOT)
    p = sc.parameters(name, ROOT)
    # Recording length only; the verifier applies the scenario's time limit.
    p["time_limit"] = 2 * p["route_length"] / plan["command_speed"] + 5
    (out / "trials").mkdir(parents=True, exist_ok=False)
    created = dt.datetime.now(dt.timezone.utc)
    work = [(name, s, plan, p, out) for s in plan["seeds"]]
    with ProcessPoolExecutor(jobs or os.cpu_count()) as pool:
        trials = list(pool.map(_trial, work))
    conf_path = ROOT / entry["path"]
    manifest = {
        "format": FORMAT,
        "run_id": f"walk-hallway-{ROBOT}-{name}-{created:%Y%m%dT%H%M%SZ}",
        "created": created.isoformat(timespec="seconds"),
        "lock": {"path": LOCK, "sha256": sha256(ROOT / LOCK), "content": lock},
        "pending_release": waiting,
        "plan": plan,
        "preconditions": sc.preconditions(name, ROOT),
        "robot": {
            "id": ROBOT,
            "profile_sha256": sha256(ROOT / "robot.json"),
            "profile": json.loads((ROOT / "robot.json").read_text()),
        },
        "configuration": {
            "name": name,
            "version": entry["version"],
            "sha256": entry["sha256"],
            "profile": json.loads(conf_path.read_text()),
        },
        "world": {
            "id": lock["world"]["id"],
            "version": lock["world"]["version"],
            "clear_width": world["dimensions"]["clear_width"],
            "route_length": p["route_length"],
            "exit_x": p["exit_x"],
        },
        "environment": {
            "simulator": {"id": "mujoco", "version": mujoco.__version__},
            "runner": {
                "version": metadata.version("livingproof-sim-mujoco"),
                "tag": lock["releases"]["runner"]["tag"],
                "commit": lock["releases"]["runner"]["commit"],
            },
            "base_body": BASE_BODY,
            "walls": build(ACTUATION, ROOT).walls,
            "source_commit": sc.git_head(ROOT)[0],
        },
        "trials": sorted(trials, key=lambda t: t["seed"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configuration")
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--allow-pending", action="store_true", help="local runs before releases exist")
    a = ap.parse_args(argv)
    m = evidence(a.configuration, a.out, jobs=a.jobs, allow_pending=a.allow_pending)
    print(f"{len(m['trials'])} observations -> {a.out / 'manifest.json'}")
    return 0  # evidence only: no grade, so nothing to signal in the exit code


if __name__ == "__main__":
    sys.exit(main())
