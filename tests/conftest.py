# SPDX-License-Identifier: Apache-2.0
import pytest

from lpl_g1.fetch import fetch


@pytest.fixture(scope="session", autouse=True)
def pinned_files():
    """Tests run against the real pinned upstream files, verified by sha256."""
    fetch()
