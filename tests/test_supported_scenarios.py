# SPDX-License-Identifier: Apache-2.0
"""The robot manifest declares the scenarios this adapter supports (spec v2.3.0).

The declaration is a list of scenario ids on livingproof-robot-manifest/0.1.
It is not a world or a scenario selected for a run. walk-corner is on the list
because the adapter drives it (``drive``, robot-capability-specs#40 step 11b).
"""

import json

from livingproof_sim_mujoco.pins import local, sha256

from lpl_g1 import ROOT
from lpl_g1.profiles import validate

SPEC_TAG = "v2.3.0"
SPEC_COMMIT = "57317e617014a6df97bb7fba9d92a890af8d42e5"
ROBOT_MANIFEST_SCHEMA = "specs/robots/robot-manifest.schema.json"


def test_robot_manifest_declares_walk_hallway_and_walk_corner():
    """walk-hallway and walk-corner, valid against the v2.3.0 manifest schema."""
    path = ROOT / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    assert manifest.get("scenarios") == ["walk-hallway", "walk-corner"]

    specs = json.loads((ROOT / "pins.json").read_text())["groups"]["specs"]
    assert specs["tag"] == SPEC_TAG
    assert specs["commit"] == SPEC_COMMIT
    schema_pin = specs["files"][ROBOT_MANIFEST_SCHEMA]
    schema = local("specs", ROBOT_MANIFEST_SCHEMA, ROOT)
    assert schema_pin["bytes"] == schema.stat().st_size
    assert schema_pin["sha256"] == sha256(schema)
    validate("robot_manifest", manifest)

    assert manifest["format"] == "livingproof-robot-manifest/0.1"
    assert manifest["robot"] == "unitree-g1"
    assert manifest["source"]["repository"] == "Living-Proof-Labs/livingproof-unitree-g1"
    assert manifest["adapter"] == {
        "entry_point": "lpl_g1.adapter:ROBOT",
        "package": "livingproof-unitree-g1",
    }
    profile = ROOT / manifest["profile"]["path"]
    assert manifest["profile"]["sha256"] == sha256(profile)
    profile_doc = json.loads(profile.read_text())
    offered = {item["name"]: item["path"] for item in profile_doc["configurations"]}
    cited = {item["name"]: item for item in manifest["configurations"]}
    assert set(cited) == set(offered)
    for name, item in cited.items():
        config = ROOT / item["path"]
        document = json.loads(config.read_text())
        assert item["path"] == offered[name]
        assert item["version"] == document["version"]
        assert item["sha256"] == sha256(config)
        assert item.get("default", False) == (name == profile_doc["default_configuration"])
    pins = json.loads((ROOT / "pins.json").read_text())["groups"]
    assets = [
        (
            asset["group"],
            asset["repository"],
            asset["commit"],
            asset["path"],
            asset["sha256"],
            asset["bytes"],
        )
        for asset in manifest["assets"]
    ]
    expected = [
        (group, body["repository"], body["commit"], path, entry["sha256"], entry["bytes"])
        for group, body in pins.items()
        for path, entry in body["files"].items()
    ]
    assert assets == expected
