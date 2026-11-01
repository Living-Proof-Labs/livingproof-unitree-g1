# SPDX-License-Identifier: Apache-2.0
"""Wall clearance from geometry, because the policy model collides only at the feet.

Each robot geom's local bounding box (``geom_aabb``) is rotated into the world and
enclosed in an axis-aligned box; the gap to each wall box is the largest separation on
any axis. The result is conservative: an enclosing box can report contact slightly
before the real surface would touch, never after.
"""

from __future__ import annotations

import mujoco
import numpy as np


class Clearance:
    def __init__(self, model: mujoco.MjModel, walls: list[str]):
        self.robot = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] != 0])
        self.walls = {}
        for name in walls:
            g = model.geom(name).id
            half = model.geom_size[g]
            centre = model.geom_pos[g]
            self.walls[name] = (centre - half, centre + half)
        self.model = model

    def robot_boxes(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        aabb = self.model.geom_aabb[self.robot]  # local centre (3), half extents (3)
        rot = data.geom_xmat[self.robot].reshape(-1, 3, 3)
        centre = data.geom_xpos[self.robot] + np.einsum("nij,nj->ni", rot, aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rot), aabb[:, 3:])
        return centre - half, centre + half

    def gaps(self, data: mujoco.MjData) -> dict[str, float]:
        """Smallest gap in metres to each wall; <= 0 means the boxes touch or overlap."""
        lo, hi = self.robot_boxes(data)
        out = {}
        for name, (wlo, whi) in self.walls.items():
            sep = np.maximum(wlo - hi, lo - whi)  # per geom, per axis
            out[name] = float(sep.max(axis=1).min())
        return out
