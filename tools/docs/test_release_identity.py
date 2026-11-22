# SPDX-License-Identifier: Apache-2.0
"""No-publish release and rendered artifact regression tests."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "release_identity", Path(__file__).with_name("release_identity.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SHA = "a" * 40
RELEASE = dict(
    tag_name="v1.2.3",
    draft=False,
    prerelease=False,
    immutable=True,
    published_at="2026-11-22T00:00:00Z",
)
ENV = dict(
    RELEASE_TAG="v1.2.3",
    RELEASE_VERSION="1.2.3",
    GITHUB_SHA=SHA,
    GITHUB_RUN_ID="42",
    GITHUB_RUN_ATTEMPT="2",
    GITHUB_REF="refs/heads/main",
    GITHUB_EVENT_NAME="workflow_run",
)


class ReleaseIdentityTests(unittest.TestCase):
    def make(self, env=None, release=None, resolved=SHA, dirty=""):
        def git(root, *args):
            if args == ("status", "--porcelain"):
                return dirty
            return resolved if "--verify" in args else SHA

        with (
            patch.dict(os.environ, ENV if env is None else env, clear=True),
            patch.object(module, "git", side_effect=git),
            patch.object(
                module.subprocess,
                "check_output",
                return_value=json.dumps(RELEASE if release is None else release),
            ) as api,
        ):
            result = module.identity(Path("."), "owner/repo")
            if result["kind"] == "release":
                self.assertEqual(
                    api.call_args.args[0], ["gh", "api", "repos/owner/repo/releases/tags/v1.2.3"]
                )
            else:
                api.assert_not_called()
            return result

    def test_published_release_and_read_only_retry(self):
        normal = self.make()
        retry = self.make(dict(ENV, GITHUB_EVENT_NAME="workflow_dispatch", RELEASE_VERSION=""))
        self.assertEqual(normal, retry)
        self.assertEqual(normal["label"], "v1.2.3")
        self.assertEqual(normal["artifact_identity"], "v1.2.3-42.2")

    def test_missing_draft_prerelease_or_unpublished_release(self):
        for change in (
            {},
            dict(RELEASE, immutable=False),
            dict(RELEASE, immutable=None),
            dict(RELEASE, draft=True),
            dict(RELEASE, prerelease=True),
            dict(RELEASE, published_at=None),
            dict(RELEASE, tag_name="v1.2.4"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make(release=change)

    def test_renderer_uses_validated_metadata_without_token_or_api(self):
        for mutable in [False, True]:
            with self.subTest(mutable=mutable), tempfile.TemporaryDirectory() as tmp:
                metadata = Path(tmp) / "release.json"
                metadata.write_text(json.dumps(dict(RELEASE, immutable=not mutable)))
                with (
                    patch.dict(os.environ, dict(ENV, RELEASE_METADATA=str(metadata)), clear=True),
                    patch.object(module, "git", side_effect=[SHA, "", SHA]),
                    patch.object(module.subprocess, "check_output") as api,
                ):
                    if mutable:
                        with self.assertRaises(ValueError):
                            module.identity(Path("."), "owner/repo", cached=True)
                    else:
                        self.assertEqual(
                            module.identity(Path("."), "owner/repo", cached=True)["release_tag"],
                            "v1.2.3",
                        )
                    api.assert_not_called()

    def test_api_failure_fails_closed(self):
        with (
            patch.dict(os.environ, ENV, clear=True),
            patch.object(module, "git", side_effect=[SHA, "", SHA]),
            patch.object(
                module.subprocess,
                "check_output",
                side_effect=RuntimeError("404 or permission/network failure"),
            ),
            self.assertRaises(RuntimeError),
        ):
            module.identity(Path("."), "owner/repo")

    def test_version_tag_source_and_dirty_mismatches(self):
        for change in (
            dict(RELEASE_VERSION="1.2.4"),
            dict(RELEASE_VERSION=""),
            dict(RELEASE_TAG="v1.2.3-rc.1"),
            dict(RELEASE_TAG="v01.2.3"),
            dict(GITHUB_SHA="b" * 40),
            dict(GITHUB_REF="refs/heads/other"),
            dict(GITHUB_EVENT_NAME="pull_request"),
            dict(SOURCE_COMMIT="b" * 40),
            dict(SOURCE_DIRTY="true"),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make(dict(ENV, **change))
        with self.assertRaises(ValueError):
            self.make(resolved="b" * 40)
        with self.assertRaises(ValueError):
            self.make(dirty=" M README.md")

    def test_preview_remains_development_and_attempt_specific(self):
        first = self.make(dict(GITHUB_RUN_ID="42", GITHUB_RUN_ATTEMPT="1"))
        second = self.make(dict(GITHUB_RUN_ID="42", GITHUB_RUN_ATTEMPT="2"))
        self.assertEqual(first["kind"], "development")
        self.assertIsNone(first["release_tag"])
        self.assertNotEqual(first["artifact_identity"], second["artifact_identity"])
        self.assertEqual(first["source_commit"], SHA)
        self.assertTrue(self.make({}, dirty=" M README.md")["source_dirty"])

    def test_partial_or_invalid_attempt_and_version_alone_rejected(self):
        for env in (
            {"GITHUB_RUN_ID": "42"},
            {"GITHUB_RUN_ATTEMPT": "1"},
            {"GITHUB_RUN_ID": "42", "GITHUB_RUN_ATTEMPT": "0"},
            {"RELEASE_VERSION": "1.2.3"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.make(env)

    def test_seal_covers_added_removed_and_changed_served_bytes(self):
        for mutation in ("added", "removed", "changed", "identity"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(os.environ, {}, clear=True),
            ):
                html = Path(tmp) / "html"
                html.mkdir()
                record = self.make({})
                (html / "build.json").write_text(json.dumps(record))
                (html / "index.html").write_text("tested")
                module.seal(html, record)
                module.verify(html)
                if mutation == "added":
                    (html / "extra.html").write_text("new")
                elif mutation == "removed":
                    (html / "index.html").unlink()
                elif mutation == "identity":
                    (html / "build.json").write_text("{}")
                else:
                    (html / "index.html").write_text("tampered")
                with self.assertRaises(ValueError):
                    module.verify(html)

    def test_prior_attempt_seal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            html = Path(tmp) / "html"
            html.mkdir()
            record = self.make(dict(GITHUB_RUN_ID="42", GITHUB_RUN_ATTEMPT="1"))
            (html / "build.json").write_text(json.dumps(record))
            module.seal(html, record)
            with (
                patch.dict(os.environ, GITHUB_RUN_ID="42", GITHUB_RUN_ATTEMPT="2"),
                self.assertRaises(ValueError),
            ):
                module.verify(html)


if __name__ == "__main__":
    unittest.main()
