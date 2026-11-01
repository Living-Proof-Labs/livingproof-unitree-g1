# SPDX-License-Identifier: Apache-2.0
"""Shared simulation setup: runner settings and the seeded initial state. Walk-hallway
trials are recorded by ``livingproof_sim_mujoco.evidence`` and graded by the capability
verifier; the zero-command stand check is ``python -m livingproof_sim_mujoco.walk stand
lpl_g1.adapter:ROBOT``."""

from __future__ import annotations

from pathlib import Path

from livingproof_sim_mujoco import walk as generic
from livingproof_sim_mujoco.walk import initial_state  # noqa: F401

from lpl_g1 import ROOT


def runner_settings(root: Path = ROOT) -> dict:
    return generic.runner_settings(root)
