# SPDX-License-Identifier: Apache-2.0
"""Caller source fixtures obey the same digest/no-fallback contract as runs."""

import json

import pytest
from livingproof_sim_mujoco import sources
from source_bundle import bound_sources


def test_explicit_inputs_are_checked_and_temporary(tmp_path):
    world_path = tmp_path / "input.json"
    world_path.write_text("{}")
    parameters = {"test": "caller input"}
    lock = {}
    with bound_sources(tmp_path, lock, world_path, parameters):
        document, origin = sources.parameters_document_for(tmp_path, lock)
        assert document == parameters
        assert origin == "lock"
        path = tmp_path / lock["sources"]["parameters"]["path"]
        path.write_text(json.dumps({"test": "tampered"}))
        with pytest.raises(SystemExit, match="sha256 mismatch"):
            sources.parameters_document_for(tmp_path, lock)
    assert not path.exists()


def test_missing_parameter_binding_is_not_filled_from_defaults(tmp_path):
    with pytest.raises(SystemExit, match="sources.parameters is missing"):
        sources.parameters_document_for(tmp_path, {})
