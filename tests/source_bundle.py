# SPDX-License-Identifier: Apache-2.0
"""Explicit caller inputs for evidence tests; never a runtime fallback."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def bound_sources(root, lock, world_path, parameters):
    cache = Path(root) / ".cache"
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="test-bundle-", dir=cache) as directory:
        folder = Path(directory)
        inputs = {"world": world_path.read_bytes(), "parameters": json.dumps(parameters).encode()}
        lock["sources"] = {}
        for kind, data in inputs.items():
            path = folder / f"{kind}.json"
            path.write_bytes(data)
            lock["sources"][kind] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        yield
