# SPDX-License-Identifier: Apache-2.0
"""Shared simulation setup (runner settings, seeded initial state) and the zero-command
standing check. Walk-hallway trials are recorded by ``lpl_g1.evidence`` and graded by
the capability verifier, never here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from livingproof_sim_mujoco import walk as generic
from livingproof_sim_mujoco.walk import initial_state  # noqa: F401
import mujoco

from lpl_g1 import ROOT
from lpl_g1 import scenario as sc
from lpl_g1.fetch import fetch
from lpl_g1.profiles import controller
from lpl_g1.scene import build

BASE_BODY = "pelvis"
CAMERA = (3.6, -38.0, 200.0)  # tracking distance, elevation, azimuth


def runner_settings(root: Path = ROOT) -> dict:
    return generic.runner_settings(root)


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
    return generic.simulate(
        model,
        data,
        lambda d: policy.step(d, command),
        base_body=BASE_BODY,
        seconds=seconds,
        p=p,
        sample_hz=rs["sample_hz"],
        stop_at_exit=stop_at_exit,
        clip=clip,
        camera_view=CAMERA,
    )


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
    result = generic.stand_summary(
        name, sc.configuration(name, root)[1], seconds, samples, end, clip, p["fall_height"]
    )
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
