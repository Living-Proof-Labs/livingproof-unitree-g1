# SPDX-License-Identifier: Apache-2.0
"""Read-only census of the pinned model and the runner's measurement selection.

Run after fetching the pinned assets: ``python -m lpl_g1.collision_audit``.
This does not add geometry, change contact pairs, or run/grade a scenario.
"""

from __future__ import annotations

import json
import warnings
from importlib.metadata import version

import mujoco

from livingproof_sim_mujoco.observations import collision_geoms
from livingproof_sim_mujoco.pins import fetch, load_pins
from lpl_g1 import ROOT
from lpl_g1.scene import build


def census(model: mujoco.MjModel) -> dict:
    """Distinguish physical contact eligibility, measurement classes and fallback."""
    paired = {int(g) for g in (*model.pair_geom1, *model.pair_geom2)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected, record = collision_geoms(model)
    selected = set(selected)
    fallback = set(record["visual_fallback_bodies"])
    bodies = []
    for b in range(1, model.nbody):
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]
        marked = [
            g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g] or g in paired
        ]
        body = model.body(b).name
        bodies.append(
            {
                "body": body,
                "geoms": len(geoms),
                "direct_contact_geoms": len(marked),
                "measurement_geoms": len(selected.intersection(geoms)),
                "measurement": (
                    "visual_fallback"
                    if body in fallback
                    else "collision_class"
                    if selected.intersection(geoms)
                    else "no_geometry"
                ),
            }
        )
    return {
        "bodies": bodies,
        "selection": record,
        "explicit_contact_pairs": [
            [model.geom(int(a)).name, model.geom(int(b)).name]
            for a, b in zip(model.pair_geom1, model.pair_geom2)
        ],
        "warnings": [str(w.message) for w in caught],
    }


def audit() -> dict:
    fetch(ROOT)  # Verify the bytes actually compiled, including cached assets.
    pins = load_pins(ROOT)["groups"]
    return {
        "configurations": ["stock", "playground"],
        "shared_model": True,
        "runner_version": version("livingproof-sim-mujoco"),
        "model_commit": pins["playground"]["commit"],
        "model_sha256": pins["playground"]["files"][
            "mujoco_playground/_src/locomotion/g1/xmls/g1_mjx_feetonly.xml"
        ]["sha256"],
        **census(build().model),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
