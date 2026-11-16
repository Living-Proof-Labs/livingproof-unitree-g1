# SPDX-License-Identifier: Apache-2.0
"""One-off, absent-only recovery of G1 v2.1.4. No ref update/delete operations."""

import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

REPOSITORY = "Living-Proof-Labs/livingproof-unitree-g1"
TAG = "v2.1.4"
SHA = "b5cfcb0d0268015aa5f9ce0914439c7ade67269d"
PREVIOUS_SHA = "dcfcf27e3803d83ebc7f1030ddbd99edf52a8a73"
NOTE_REF = f"notes/semantic-release-{TAG}"
NOTE = '{"channels":[null]}\n'
NOTES_PATH = Path(__file__).with_name("v2.1.4-notes.md")
NOTES_DIGEST = "cdf65299d4e9c2ca2d52926ea0fef4796c7f96cd4d1167b967c8702460c6c780"


class API:
    def __init__(self, token):
        self.token = token

    def request(self, method, path, data=None, optional=False):
        if method not in ("GET", "POST"):
            raise ValueError("Only read and create operations are permitted")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{REPOSITORY}/{path}",
            data=None if data is None else json.dumps(data).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if method == "GET" and optional and exc.code == 404:
                return None
            # Never echo a request/response which could contain credentials.
            raise RuntimeError(f"GitHub {method} {path}: HTTP {exc.code}") from None


def expect(value, message):
    if not value:
        raise RuntimeError(message)


def read_notes():
    raw = NOTES_PATH.read_bytes()
    expect(hashlib.sha256(raw).hexdigest() == NOTES_DIGEST, "Release notes digest mismatch")
    return raw.decode()


def inspect(api, notes):
    for tag, sha in ((TAG, SHA), ("v2.1.3", PREVIOUS_SHA)):
        ref = api.request("GET", f"git/ref/tags/{tag}")
        expect(
            ref["object"]["type"] == "commit" and ref["object"]["sha"] == sha,
            f"Unexpected {tag} target/type",
        )
    note = api.request("GET", f"git/ref/{NOTE_REF}", optional=True)
    if note is not None:
        expect(note["object"]["type"] == "commit", "Unexpected notes ref type")
        commit = api.request("GET", f"git/commits/{note['object']['sha']}")
        tree = api.request("GET", f"git/trees/{commit['tree']['sha']}?recursive=1")
        expect(tree.get("truncated") is False, "Incomplete note tree")
        leaves = [entry for entry in tree["tree"] if entry["type"] != "tree"]
        expect(len(leaves) == 1, "Unexpected additional/missing notes")
        leaf = leaves[0]
        expect(
            leaf["type"] == "blob"
            and leaf["mode"] == "100644"
            and leaf["path"].replace("/", "") == SHA,
            "Unexpected noted commit",
        )
        blob = api.request("GET", f"git/blobs/{leaf['sha']}")
        expect(blob["encoding"] == "base64", "Unexpected note encoding")
        expect(
            json.loads(base64.b64decode(blob["content"])) == {"channels": [None]},
            "Conflicting existing channel note",
        )
    release = api.request("GET", f"releases/tags/{TAG}", optional=True)
    if release is not None:
        expect(
            release["tag_name"] == TAG
            and release["name"] == TAG
            and release["draft"] is False
            and release["prerelease"] is False
            and release["body"] == notes,
            "Conflicting existing release",
        )
    return {"note_present": note is not None, "release_present": release is not None}


def recover(api, *, apply=False):
    notes = read_notes()
    state = inspect(api, notes)
    if not apply:
        return {"mode": "preview", "tag": TAG, "sha": SHA, **state}
    # Revalidate all bounds immediately before creating any state.
    state = inspect(api, notes)
    if not state["note_present"]:
        blob = api.request("POST", "git/blobs", {"content": NOTE, "encoding": "utf-8"})
        tree = api.request(
            "POST",
            "git/trees",
            {"tree": [{"path": SHA, "mode": "100644", "type": "blob", "sha": blob["sha"]}]},
        )
        commit = api.request(
            "POST",
            "git/commits",
            {
                "message": "Recover default-channel note for existing G1 v2.1.4",
                "tree": tree["sha"],
                "parents": [],
            },
        )
        # POST creates only; a competing writer causes 422, never an overwrite.
        # No retry after an uncertain POST. A later invocation reads state first.
        expect(not inspect(api, notes)["note_present"], "Note appeared during recovery")
        api.request("POST", "git/refs", {"ref": f"refs/{NOTE_REF}", "sha": commit["sha"]})
        expect(inspect(api, notes)["note_present"], "Note creation not verified")
    state = inspect(api, notes)
    if not state["release_present"]:
        api.request(
            "POST",
            "releases",
            {
                "tag_name": TAG,
                "target_commitish": SHA,
                "name": TAG,
                "body": notes,
                "draft": False,
                "prerelease": False,
                "make_latest": "false",
            },
        )
    final = inspect(api, notes)
    expect(final["note_present"] and final["release_present"], "Recovery readback incomplete")
    return {"mode": "applied", "tag": TAG, "sha": SHA, **final}


def main():
    expect(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "Wrong repository")
    expect(os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch", "Manual dispatch required")
    expect(os.environ.get("GITHUB_REF") == "refs/heads/main", "Trusted main required")
    mode = os.environ.get("RECOVERY_APPLY", "false")
    expect(mode in ("true", "false"), "Invalid apply value")
    token = os.environ.get("GH_TOKEN")
    expect(bool(token), "Workflow token required")
    print(json.dumps(recover(API(token), apply=mode == "true"), sort_keys=True))


if __name__ == "__main__":
    main()
