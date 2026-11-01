# SPDX-License-Identifier: Apache-2.0
import pytest

from livingproof_sim_mujoco import pins

from lpl_g1 import ROOT


@pytest.fixture(scope="session", autouse=True)
def pinned_files():
    """Tests run against the real pinned upstream files, verified by sha256."""
    pins.fetch(ROOT)
