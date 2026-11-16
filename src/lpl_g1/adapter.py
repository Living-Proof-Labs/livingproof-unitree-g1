# SPDX-License-Identifier: Apache-2.0
"""The Unitree G1 adapter for livingproof-sim-mujoco.

Only robot-specific code and data live here: the scene build, the published controllers,
the nominal initial pose and the profiles. This package does not select a world or a scenario.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from lpl_g1 import ROOT


class G1:
    id = "unitree-g1"
    root: Path = ROOT
    width_decimals = 2
    camera = (3.6, -38.0, 200.0)  # tracking distance, elevation, azimuth
    actuation = "pd_torque"

    def base_body(self, name: str) -> str:
        return "pelvis"

    def scene(self, name: str):
        from lpl_g1.scene import build

        return build(self.actuation, self.root)

    def controller(self, name: str, scene):
        from lpl_g1.profiles import controller

        return controller(name, scene, self.root)

    def policy(self, model, ctrl):
        from livingproof_sim_mujoco.policy import OnnxPolicy

        return OnnxPolicy(model, ctrl.policy_path, ctrl.spec)

    def initial_state(self, model, data, ctrl, spawn: str, *, start_offset=None) -> None:
        """Exact nominal pose at the place ``spawn``. ``start_offset`` is one declared change.

        Signature matches livingproof-sim-mujoco v3.0.4 ``Robot.initial_state``:
        ``(model, data, ctrl, spawn, start_offset=None)``. There is no seed and no noise.

        ``spawn`` is a place the simulator has already compiled into ``model``. This package
        does not add one. A missing place raises ``MissingPlace``, the simulator's
        ``INCOMPATIBLE`` stop (a ``SystemExit``, also a ``KeyError``), as the Go1 and T1
        adapters do: the trial would otherwise start at the origin. Use ``nominal_state``
        for the controller's own nominal pose with no world.
        """
        from livingproof_sim_mujoco.scene import PLACE
        from livingproof_sim_mujoco.walk import initial_state

        try:
            model.site(PLACE + spawn)
        except KeyError:
            raise MissingPlace(missing_place(spawn)) from None
        initial_state(model, data, ctrl, spawn, start_offset=start_offset)

    def nominal_state(self, model, data, ctrl, *, start_offset=None) -> None:
        """The controller's nominal ``initial_qpos`` at the origin, with no world.

        For the upstream reference loops and the package tests, which run the published
        scene unchanged. Certification uses ``initial_state`` with a world place.
        """
        _nominal_state(model, data, ctrl, start_offset)

    def runner_settings(self) -> dict:
        from livingproof_sim_mujoco.walk import runner_settings

        return runner_settings(self.root)

    def robot(self) -> dict:
        from lpl_g1.profiles import robot

        return robot(self.root)

    def configuration(self, name: str) -> dict:
        from lpl_g1.profiles import configuration

        return configuration(name, self.root)[0]

    def environment(self, name: str) -> dict:
        return {}

    def drive(self, name: str, model, ctrl, policy, places: dict):
        """Steer the published controller along the scenario's route (``lpl_g1.route``).

        livingproof-sim-mujoco v3.2.0 calls this for a scenario with no command that
        ``manifest.json`` declares. ``step(data)`` sends one velocity command per step.
        """
        from lpl_g1.route import drive

        return drive(self, name, model, ctrl, policy, places)


def _start_offset(start_offset) -> tuple[np.ndarray, float]:
    """One declared xy/yaw change. Absent means the exact nominal pose."""
    if start_offset is None:
        return np.zeros(2), 0.0
    if not isinstance(start_offset, dict):
        raise TypeError(
            "start_offset must be a dict with base_xy_m and/or base_yaw_rad, "
            f"got {type(start_offset).__name__}. "
            "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
        )
    unknown = set(start_offset) - {"base_xy_m", "base_yaw_rad"}
    if unknown:
        raise ValueError(
            "start_offset only accepts base_xy_m and base_yaw_rad "
            f"(one named comparison change), got {sorted(unknown)}. "
            "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
        )
    dxy = np.zeros(2)
    if "base_xy_m" in start_offset:
        raw = start_offset["base_xy_m"]
        if len(raw) != 2:
            raise ValueError(
                "start_offset base_xy_m must be [x, y] in metres. "
                "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
            )
        dxy = np.array([float(raw[0]), float(raw[1])], dtype=float)
    return dxy, float(start_offset.get("base_yaw_rad", 0.0))


class MissingPlace(SystemExit, KeyError):
    """A start place the composed world does not have.

    The simulator's own stop for an incompatible run: a ``SystemExit`` whose text starts
    with ``INCOMPATIBLE`` and says no trial was recorded, so a run reports it the same way
    as the simulator's other start-place checks rather than as a traceback. It is also a
    ``KeyError`` for callers that catch a missing place. ``str()`` is the message.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def missing_place(spawn: str) -> str:
    """The error for a start place that is not on the compiled model."""
    from livingproof_sim_mujoco.scene import PLACE

    return (
        f"INCOMPATIBLE: start place {spawn!r} is not a place in the composed world "
        f"(no site {PLACE + spawn!r} on the compiled model). No trial was recorded. "
        "The scenario start must name a pose in that world, such as 'spawn-entrance'. "
        "For the controller's nominal pose with no world, call nominal_state(model, data, ctrl)."
    )


def _nominal_state(model, data, ctrl, start_offset) -> None:
    """Controller nominal pose, plus one declared ``start_offset`` when given."""
    dxy, dyaw = _start_offset(start_offset)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = ctrl.initial_qpos
    data.qpos[0:2] = data.qpos[0:2] + dxy
    if dyaw:
        w, _, _, z = (float(v) for v in data.qpos[3:7])
        yaw = math.atan2(z, w) * 2 + dyaw
        data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qvel[:] = 0.0
    if ctrl.spec.actuation == "position":  # hold the default pose until the first tick,
        for name, q in zip(ctrl.spec.actuators, ctrl.spec.default_pose):  # as keyframe ctrl
            data.ctrl[model.actuator(name).id] = q
    mujoco.mj_forward(model, data)


ROBOT = G1()
