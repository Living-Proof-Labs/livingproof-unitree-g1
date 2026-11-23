# SPDX-License-Identifier: Apache-2.0
"""Read-only release identity and exact rendered-byte seals for docs delivery."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def published_release(tag, version, commit, resolved, release):
    if not re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag):
        raise ValueError("Expected an exact stable release tag")
    if tag != "v" + version or commit != resolved:
        raise ValueError("Release tag/version/source mismatch")
    if (
        release.get("tag_name") != tag
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not True
        or not release.get("published_at")
    ):
        raise ValueError("An existing published stable release is required")


def identity(root, repository, cached=False):
    commit = git(root, "rev-parse", "HEAD")
    dirty = bool(git(root, "status", "--porcelain"))
    tag = os.environ.get("RELEASE_TAG", "")
    version = os.environ.get("RELEASE_VERSION", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if bool(run_id) != bool(attempt) or any(
        not re.fullmatch(r"[1-9][0-9]*", x) for x in (run_id, attempt) if x
    ):
        raise ValueError("Run ID and attempt must be positive numbers supplied together")
    if os.environ.get("SOURCE_COMMIT", commit) != commit:
        raise ValueError("Source override differs from actual checkout")
    if os.environ.get("SOURCE_DIRTY", str(dirty).lower()) != str(dirty).lower():
        raise ValueError("Dirty override differs from actual checkout")
    result = dict(
        kind="development",
        source_commit=commit,
        source_dirty=dirty,
        run_id=run_id or None,
        run_attempt=attempt or None,
        release_tag=None,
        release_version=None,
        release_url=None,
    )
    if tag:
        if dirty or not run_id or os.environ.get("GITHUB_SHA") != commit:
            raise ValueError("Release requires clean exact event source and attempt")
        if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
            raise ValueError("Pull requests cannot build stable documentation")
        if os.environ.get("GITHUB_REF") != "refs/heads/main":
            raise ValueError("Release delivery is main-only")
        if not version and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            version = tag.removeprefix("v")
        # Validate syntax before passing the tag to Git or the API.
        published_release(
            tag,
            version,
            commit,
            commit,
            dict(
                tag_name=tag,
                draft=False,
                prerelease=False,
                immutable=True,
                published_at="syntax-only",
            ),
        )
        resolved = git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
        if cached:
            release = json.loads(Path(os.environ["RELEASE_METADATA"]).read_text())
        else:
            release = json.loads(
                subprocess.check_output(
                    ["gh", "api", f"repos/{repository}/releases/tags/{tag}"], text=True
                )
            )
        published_release(tag, version, commit, resolved, release)
        if not cached and os.environ.get("RELEASE_METADATA"):
            Path(os.environ["RELEASE_METADATA"]).write_text(json.dumps(release))
        result.update(
            kind="release",
            release_tag=tag,
            release_version=version,
            release_url=f"https://github.com/{repository}/releases/tag/{tag}",
        )
    elif version:
        raise ValueError("A version alone is not a release")
    result["label"] = tag or ("development" + (f"-{run_id}.{attempt}" if run_id else ""))
    result["artifact_identity"] = result["label"] + (f"-{run_id}.{attempt}" if tag else "")
    return result


def hashes(html):
    if any(p.is_symlink() for p in html.rglob("*")):
        raise ValueError("Served files must not be symlinks")
    return {
        p.relative_to(html).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(html.rglob("*"))
        if p.is_file()
    }


def seal(html, record):
    files = hashes(html)
    if not files or "build.json" not in files:
        raise ValueError("Missing rendered manual identity")
    path = html.parent / "seal.json"
    path.write_text(json.dumps(dict(identity=record, files=files), sort_keys=True, indent=2) + "\n")
    verify(html)


def verify(html):
    record = json.loads((html.parent / "seal.json").read_text())
    if record["files"] != hashes(html):
        raise ValueError("Rendered served bytes differ from tested seal")
    built = json.loads((html / "build.json").read_text())
    if any(built.get(k) != v for k, v in record["identity"].items()):
        raise ValueError("Rendered identity differs from seal")
    for key, env in [("run_id", "GITHUB_RUN_ID"), ("run_attempt", "GITHUB_RUN_ATTEMPT")]:
        if os.environ.get(env) and record["identity"][key] != os.environ[env]:
            raise ValueError("Stale attempt: rerun all jobs")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if sys.argv[1] == "verify":
        verify(root / sys.argv[2])
    else:
        record = identity(root, sys.argv[2])
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("artifact_identity=" + record["artifact_identity"] + "\n")
            output.write("run_attempt=" + str(record["run_attempt"]) + "\n")
