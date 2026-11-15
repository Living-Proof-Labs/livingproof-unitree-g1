# SPDX-License-Identifier: Apache-2.0
"""Deterministic cache-key contract, not a reproduction of a user's browser."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlsplit

spec = importlib.util.spec_from_file_location("docs_build", Path(__file__).parents[1] / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class CacheContract(unittest.TestCase):
    def test_warm_asset_cache_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = b"body { color: black; }"
            new = b"body { color: white; }"
            old_url = build.stage_brand(root, old, hashlib.sha256(old).hexdigest())
            warm_cache = {old_url: old}
            new_url = build.stage_brand(root, new, hashlib.sha256(new).hexdigest())
            self.assertNotEqual(old_url, new_url)
            self.assertNotIn(new_url, warm_cache)
            # A cold request from old HTML still resolves to the stable file.
            self.assertEqual((root / urlsplit(old_url).path).read_bytes(), new)
            self.assertEqual((root / urlsplit(new_url).path).read_bytes(), new)
            self.assertEqual(build.stage_brand(root, new, hashlib.sha256(new).hexdigest()), new_url)
            page = root / "nested/page.html"
            parsed = build.Page(f'<link rel="stylesheet" href="../{new_url}">')
            build.check_brand(root, page, parsed, new_url, hashlib.sha256(new).hexdigest())

    def test_absolute_404_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            css = b"body {}"
            digest = hashlib.sha256(css).hexdigest()
            url = build.stage_brand(root, css, digest)
            build.check_brand(
                root,
                root / "404.html",
                build.Page(f'<link rel="stylesheet" href="/livingproof-unitree-g1/{url}">'),
                url,
                digest,
            )

    def test_stale_reference_and_corruption_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            css = b"body {}"
            digest = hashlib.sha256(css).hexdigest()
            url = build.stage_brand(root, css, digest)
            with self.assertRaises(AssertionError):
                build.check_brand(
                    root,
                    root / "index.html",
                    build.Page('<link rel="stylesheet" href="assets/brand.css">'),
                    url,
                    digest,
                )
            with self.assertRaises(AssertionError):
                build.check_brand(
                    root,
                    root / "index.html",
                    build.Page('<link rel="stylesheet" href="assets/brand.css?v=outdated">'),
                    url,
                    digest,
                )
            (root / urlsplit(url).path).write_bytes(b"tampered")
            with self.assertRaises(AssertionError):
                build.check_brand(
                    root,
                    root / "index.html",
                    build.Page(f'<link rel="stylesheet" href="{url}">'),
                    url,
                    digest,
                )
            with self.assertRaises(ValueError):
                build.stage_brand(root, css, "0" * 64)


if __name__ == "__main__":
    unittest.main()
