# SPDX-License-Identifier: Apache-2.0
"""The committed final pose is still compared on x86_64, and only that field is gated.

The x86_64 gate in test_playground_exact.py keys on platform.machine(). These tests
force the machine name, tamper with the committed final pose by one ulp, and check
that x86_64 catches it while other machines skip only that one field.
"""

import importlib
import platform

import numpy as np
import pytest

import test_playground_exact as exact


def _tampered(tmp_path, field):
    saved = dict(np.load(exact.TRACE))
    flat = saved[field].reshape(-1).copy()
    flat[0] = np.nextafter(flat[0], np.inf)
    saved[field] = flat.reshape(saved[field].shape).astype(saved[field].dtype)
    path = tmp_path / "trace.npz"
    np.savez_compressed(path, **saved)
    return path


def _force(monkeypatch, machine):
    """Pretend to run on ``machine``. Reload so a gate computed at import sees it too."""
    monkeypatch.setattr(platform, "machine", lambda: machine)
    importlib.reload(exact)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.delenv("LPL_REGEN_REFERENCE", raising=False)
    yield
    monkeypatch.undo()
    importlib.reload(exact)


def test_x86_64_compares_the_committed_final_pose(tmp_path, monkeypatch):
    _force(monkeypatch, "x86_64")
    monkeypatch.setattr(exact, "TRACE", _tampered(tmp_path, "final_qpos"))
    with pytest.raises(AssertionError, match="final_qpos"):
        exact.test_runner_equals_upstream_rollout_and_committed_trace_exactly()


def test_other_machines_still_compare_actions_with_the_committed_trace(tmp_path, monkeypatch):
    _force(monkeypatch, "aarch64")
    monkeypatch.setattr(exact, "TRACE", _tampered(tmp_path, "actions"))
    with pytest.raises(AssertionError, match="actions"):
        exact.test_runner_equals_upstream_rollout_and_committed_trace_exactly()


def test_the_gate_keys_on_platform_machine_only():
    with open(exact.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "platform.machine()" in source
    for broad in ("sys.platform", "os.name", "platform.system", "GITHUB_ACTIONS"):
        assert broad not in source.split('"""', 2)[-1], broad
