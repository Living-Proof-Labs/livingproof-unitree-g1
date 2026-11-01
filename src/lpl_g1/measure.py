# SPDX-License-Identifier: Apache-2.0
"""Robot dimensions and mass measured from the pinned model (robot.json source: measured).

Width and length are the extents of every mesh vertex in world coordinates at the two
start keyframes (knees_bent, home), taking the larger; mass is the sum of body masses.
"""

from __future__ import annotations

import json

from livingproof_sim_mujoco.measure import keyframe_extents

from lpl_g1.scene import build


def measure() -> dict:
    return keyframe_extents(build().model, ("knees_bent", "home"))


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
