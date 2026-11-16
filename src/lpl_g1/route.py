# SPDX-License-Identifier: Apache-2.0
"""How this adapter drives a scenario that sends no command (robot-capability-specs#40).

livingproof-sim-mujoco v3.2.0 asks the adapter to drive such a scenario itself. ``drive`` returns a ``step(data)`` that steers the
published velocity controller along the world's route polyline: pure pursuit on a
point ``LOOKAHEAD`` metres ahead along the route, a yaw-rate command proportional to
the heading error, and a forward speed that slows while the heading error is large.
The controller, its gains and the robot are unchanged; only the command is computed.
"""

from __future__ import annotations

import math

import numpy as np

SPEED = 0.5  # m/s, the speed the verification plans command on a straight route
LOOKAHEAD = 0.7  # m along the route
YAW_GAIN = 2.5  # rad/s per rad of heading error
YAW_RATE_MAX = 1.0  # rad/s
SLOW_ERROR = 0.4  # rad; forward speed scales with cos(error) and stops past this
CROSS_GAIN = 1.0  # m/s sideways per metre off the route, in the robot's frame
CROSS_MAX = 0.3  # m/s


def _route(points: list) -> tuple[np.ndarray, np.ndarray]:
    """Route vertices in the plane, extended one segment past the exit, and arc lengths."""
    xy = np.array([p[:2] for p in points], dtype=float)
    last = xy[-1] + (xy[-1] - xy[-2])
    xy = np.vstack([xy, last])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return xy, arc


def _nearest(xy: np.ndarray, arc: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Arc length and point of the route nearest ``p``."""
    best, best_s, best_q = math.inf, 0.0, xy[0]
    for i in range(len(xy) - 1):
        a, b = xy[i], xy[i + 1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(q - p))
        if d < best:
            best, best_s, best_q = d, arc[i] + t * float(np.linalg.norm(ab)), q
    return best_s, best_q


def _point_at(xy: np.ndarray, arc: np.ndarray, s: float) -> np.ndarray:
    s = min(max(s, 0.0), arc[-1])
    i = int(np.searchsorted(arc, s, side="right") - 1)
    i = min(i, len(xy) - 2)
    t = (s - arc[i]) / max(arc[i + 1] - arc[i], 1e-12)
    return xy[i] + t * (xy[i + 1] - xy[i])


def command(xy, arc, position: np.ndarray, yaw: float) -> list[float]:
    """``[vx, vy, yaw_rate]`` toward the look-ahead point from ``position`` at ``yaw``."""
    along, nearest = _nearest(xy, arc, position)
    target = _point_at(xy, arc, along + LOOKAHEAD)
    dx, dy = target - position
    heading = math.atan2(dy, dx)
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    yaw_rate = float(np.clip(YAW_GAIN * error, -YAW_RATE_MAX, YAW_RATE_MAX))
    forward = SPEED * max(0.0, math.cos(min(abs(error) / SLOW_ERROR, 1.0) * math.pi / 2))
    # Sideways toward the route, in the robot's frame (+y is the robot's left).
    ox, oy = nearest - position
    left = -math.sin(yaw) * ox + math.cos(yaw) * oy
    sideways = float(np.clip(CROSS_GAIN * left, -CROSS_MAX, CROSS_MAX))
    return [forward, sideways, yaw_rate]


def follow_route(model, policy, base_body: str, route: dict):
    """``step(data)``: one policy step toward the route, at the policy's own rate."""
    xy, arc = _route(route["points"])
    body = model.body(base_body).id

    def step(data) -> None:
        position = np.array(data.xpos[body][:2], dtype=float)
        w, x, y, z = data.xquat[body]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        policy.step(data, command(xy, arc, position, yaw))

    return step


def drive(robot, name: str, model, ctrl, policy, places: dict):
    """The adapter's ``drive``: follow ``places['route']`` with the published controller.

    ``places`` is the simulator's (start, exit, route and through, the world's own place
    objects). Only the route polyline is used; this package names no world or place.
    """
    del ctrl
    return follow_route(model, policy, robot.base_body(name), places["route"])
