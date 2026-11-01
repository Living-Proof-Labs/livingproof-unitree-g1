# SPDX-License-Identifier: Apache-2.0
"""The Unitree G1 adapter for livingproof-sim-mujoco's generic walk-hallway evidence loop.
Only robot-specific code and data live here: the scene build, the published controllers,
the seeded initial pose and the profiles."""

from __future__ import annotations

from pathlib import Path

from lpl_g1 import ROOT


class G1:
    id = "unitree-g1"
    root: Path = ROOT
    width_decimals = 2
    camera = (3.6, -38.0, 200.0)  # stand and media clips: tracking distance, elevation, azimuth
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

    def initial_state(self, model, data, ctrl, seed, noise, spawn) -> None:
        from lpl_g1.walk import initial_state

        initial_state(model, data, ctrl, seed, noise, spawn)

    def runner_settings(self) -> dict:
        from lpl_g1.walk import runner_settings

        return runner_settings(self.root)

    def robot(self) -> dict:
        from lpl_g1.profiles import robot

        return robot(self.root)

    def configuration(self, name: str) -> dict:
        from lpl_g1.profiles import configuration

        return configuration(name, self.root)[0]

    def world(self) -> dict:
        from lpl_g1.scene import load_world

        return load_world(self.root)

    def environment(self, name: str) -> dict:
        return {}


ROBOT = G1()
