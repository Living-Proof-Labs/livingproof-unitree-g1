# SPDX-License-Identifier: Apache-2.0
"""The ``sources.parameters`` entry a caller lock declares for the evidence runner.

livingproof-sim-mujoco v3.4.1 refuses a lock with no ``sources.parameters``: pins.json
and the vendored parameter file are no longer a fallback (livingproof-verification#119).
A real run has the verifier place the declared file inside this repository. Here the
byte copy the simulator vendors is placed under ``.cache`` (git-ignored, like the pinned
files) and named by path and sha256, exactly as a verification lock records it.
"""

import hashlib
from pathlib import Path
import shutil

from lpl_g1 import ROOT

DECLARED = ROOT / ".cache" / "declared-parameters"


def parameters_source(source: Path) -> dict:
    """Copy the parameter file at ``source`` into the repository and return its lock entry."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dest = DECLARED / digest / "parameters.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return {"path": dest.relative_to(ROOT).as_posix(), "sha256": digest}
