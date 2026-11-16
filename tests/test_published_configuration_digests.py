# SPDX-License-Identifier: Apache-2.0
"""A configuration profile version is one file. A comment edit needs a new version."""

import json

from livingproof_sim_mujoco.pins import sha256

from lpl_g1 import ROOT

# Published configuration versions at main 07ab4f6. A byte change needs a new version
# and a new digest here.
PUBLISHED_CONFIGURATION = {
    ("configurations/stock.json", "1.1.1"): (
        "8baf584eb025fa4408aad20a81521b73bc068a3a7e613dfeeb8f03d24eb02b5e"
    ),
    ("configurations/playground.json", "1.0.1"): (
        "1fef76d2e5b6eb0ab7627d1f93333e8ca2ec01249a5b1d1f68876186f0657e89"
    ),
}


def test_a_configuration_version_keeps_its_published_digest():
    """stock 1.1.1 and playground 1.0.1 keep the published 07ab4f6 digests."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    problems = []
    for item in manifest["configurations"]:
        path = item["path"]
        current = json.loads((ROOT / path).read_text())
        digest = sha256(ROOT / path)
        if current["version"] != item["version"]:
            problems.append(
                f"{path} version is {current['version']}, manifest says {item['version']}"
            )
        if digest != item["sha256"]:
            problems.append(f"{path} sha256 differs from the manifest")
        published = PUBLISHED_CONFIGURATION.get((path, current["version"]))
        if published is not None and digest != published:
            problems.append(
                f"{path} version {current['version']} changed bytes without a new version"
            )
    assert problems == []
