# SPDX-License-Identifier: Apache-2.0
"""Robot dimensions and mass measured from the pinned model (robot.json source: measured).

Width and length are the extents of every mesh vertex in world coordinates at the two
start keyframes (knees_bent, home), taking the larger; mass is the sum of body masses.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from lpl_g1.scene import build


def measure() -> dict:
    m = build().model
    d = mujoco.MjData(m)
    ext = []
    for key in ("knees_bent", "home"):
        mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
        d.qpos[:2] = 0
        mujoco.mj_forward(m, d)
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0 or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
        p = np.vstack(pts)
        ext.append({"keyframe": key, "extent_xyz_m": (p.max(0) - p.min(0)).round(3).tolist()})
    return {
        "width_m": max(e["extent_xyz_m"][1] for e in ext),
        "length_m": max(e["extent_xyz_m"][0] for e in ext),
        "mass_kg": round(float(sum(m.body_mass)), 2),
        "keyframes": ext,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
