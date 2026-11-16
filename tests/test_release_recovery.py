# SPDX-License-Identifier: Apache-2.0
"""Offline safety tests for the one-off v2.1.4 reconciler."""

import base64
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

PATH = Path(__file__).parents[1] / "tools/release/recover_v214.py"
spec = importlib.util.spec_from_file_location("recover_v214", PATH)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class FakeAPI:
    def __init__(self, note=False, release=False):
        self.note = note
        self.release = self.release_data() if release else None
        self.target = recovery.SHA
        self.note_payload = recovery.NOTE
        self.writes = []
        self.fail_path = None
        self.race = False

    def release_data(self):
        return {
            "tag_name": recovery.TAG,
            "name": recovery.TAG,
            "draft": False,
            "prerelease": False,
            "body": recovery.read_notes(),
        }

    def request(self, method, path, data=None, optional=False):
        if path == self.fail_path:
            raise RuntimeError("simulated unexpected API failure")
        if method == "GET":
            if path.startswith("git/ref/tags/"):
                sha = self.target if path.endswith(recovery.TAG) else recovery.PREVIOUS_SHA
                return {"object": {"type": "commit", "sha": sha}}
            if path.startswith("git/ref/notes/"):
                return {"object": {"type": "commit", "sha": "note"}} if self.note else None
            if path == "git/commits/note":
                return {"tree": {"sha": "tree"}}
            if path == "git/trees/tree?recursive=1":
                return {
                    "truncated": False,
                    "tree": [
                        {"path": recovery.SHA, "mode": "100644", "type": "blob", "sha": "blob"}
                    ],
                }
            if path == "git/blobs/blob":
                return {
                    "encoding": "base64",
                    "content": base64.b64encode(self.note_payload.encode()).decode(),
                }
            if path.startswith("releases/tags/"):
                return copy.deepcopy(self.release)
            raise AssertionError(path)
        assert method == "POST", "No update/delete methods allowed"
        self.writes.append((path, data))
        if path in ("git/blobs", "git/trees", "git/commits"):
            return {"sha": path.split("/")[1]}
        if path == "git/refs":
            if self.race:
                raise RuntimeError("422 competing ref creation")
            assert not self.note
            assert data["ref"] == f"refs/{recovery.NOTE_REF}"
            self.note = True
            return {}
        if path == "releases":
            assert self.release is None
            assert data["target_commitish"] == recovery.SHA
            assert data["make_latest"] == "false"
            self.release = data
            return data
        raise AssertionError(path)


def test_preview_never_writes():
    api = FakeAPI()
    assert recovery.recover(api)["mode"] == "preview"
    assert not api.writes


@pytest.mark.parametrize(
    "note,release", [(False, False), (True, False), (False, True), (True, True)]
)
def test_all_partial_states_reconcile_without_overwrite(note, release):
    api = FakeAPI(note, release)
    assert recovery.recover(api, apply=True)["release_present"]
    paths = [p for p, _ in api.writes]
    assert paths.count("git/refs") == int(not note)
    assert paths.count("releases") == int(not release)
    assert not any(p.startswith("git/ref/tags") for p in paths)
    count = len(api.writes)
    recovery.recover(api, apply=True)
    assert len(api.writes) == count


def test_wrong_tag_stops_without_writes():
    api = FakeAPI()
    api.target = "0" * 40
    with pytest.raises(RuntimeError, match="target"):
        recovery.recover(api, apply=True)
    assert not api.writes


def test_conflicting_note_stops_without_writes():
    api = FakeAPI(note=True)
    api.note_payload = '{"channels":["other"]}'
    with pytest.raises(RuntimeError, match="channel"):
        recovery.recover(api, apply=True)
    assert not api.writes


def test_conflicting_release_stops_without_writes():
    api = FakeAPI(release=True)
    api.release["body"] = "unreviewed notes"
    with pytest.raises(RuntimeError, match="release"):
        recovery.recover(api, apply=True)
    assert not api.writes


def test_api_failure_is_not_absence():
    api = FakeAPI()
    api.fail_path = f"git/ref/{recovery.NOTE_REF}"
    with pytest.raises(RuntimeError, match="API failure"):
        recovery.recover(api, apply=True)
    assert not api.writes


def test_competing_ref_creation_never_forces_or_publishes():
    api = FakeAPI()
    api.race = True
    with pytest.raises(RuntimeError, match="422"):
        recovery.recover(api, apply=True)
    assert "releases" not in [p for p, _ in api.writes]


def test_failed_publication_preserves_note_and_can_resume():
    api = FakeAPI()
    api.fail_path = "releases"
    with pytest.raises(RuntimeError):
        recovery.recover(api, apply=True)
    assert api.note and api.release is None
    api.fail_path = None
    recovery.recover(api, apply=True)
    assert [p for p, _ in api.writes].count("git/refs") == 1


def test_notes_digest_binds_reviewed_body():
    with patch.object(recovery, "NOTES_DIGEST", "invalid"):
        with pytest.raises(RuntimeError, match="digest"):
            recovery.read_notes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("GITHUB_REPOSITORY", "other/repo"),
        ("GITHUB_EVENT_NAME", "push"),
        ("GITHUB_REF", "refs/heads/feature"),
        ("RECOVERY_APPLY", "yes"),
    ],
)
def test_entrypoint_rejects_untrusted_context(field, value):
    env = {
        "GITHUB_REPOSITORY": recovery.REPOSITORY,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "RECOVERY_APPLY": "false",
        "GH_TOKEN": "test",
    }
    env[field] = value
    with patch.dict(recovery.os.environ, env, clear=True):
        with pytest.raises(RuntimeError):
            recovery.main()


def test_workflow_has_manual_preview_and_narrow_serialized_apply():
    workflow = (PATH.parents[2] / ".github/workflows/recover-v2.1.4.yml").read_text()
    assert "workflow_dispatch:" in workflow and "default: false" in workflow
    assert "needs: preview" in workflow and "if: inputs.apply &&" in workflow
    assert "group: shared-semantic-release" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "contents: write" in workflow and "persist-credentials: false" in workflow
    assert "pull_request:" not in workflow


def test_uncertain_release_post_is_not_retried():
    api = FakeAPI(note=True)
    original = api.request

    def uncertain(method, path, data=None, optional=False):
        result = original(method, path, data, optional)
        if method == "POST" and path == "releases":
            raise RuntimeError("lost response after creation")
        return result

    api.request = uncertain
    with pytest.raises(RuntimeError, match="lost response"):
        recovery.recover(api, apply=True)
    assert [p for p, _ in api.writes].count("releases") == 1
    api.request = original
    recovery.recover(api, apply=True)
    assert [p for p, _ in api.writes].count("releases") == 1


def test_incomplete_note_tree_never_authorizes_writes():
    api = FakeAPI(note=True)
    original = api.request

    def incomplete(method, path, data=None, optional=False):
        result = original(method, path, data, optional)
        if path.startswith("git/trees/"):
            result["truncated"] = True
        return result

    api.request = incomplete
    with pytest.raises(RuntimeError, match="Incomplete"):
        recovery.recover(api, apply=True)
    assert not api.writes
