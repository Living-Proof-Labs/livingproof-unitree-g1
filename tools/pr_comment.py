# SPDX-License-Identifier: Apache-2.0
"""Post or update the capability-report comment on a pull request.

The body is the reporter's own markdown (``report.md`` per configuration, rendered by
``livingproof-capability-report render --format markdown``), unchanged, under a hidden
marker. An existing comment carrying the marker is updated in place; otherwise one is
created. ``--dry-run`` prints what would be sent and calls nothing that writes.

    python3 tools/pr_comment.py --repo OWNER/NAME --pr N reports/*/report.md [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

MARKER = "<!-- livingproof-capability-report -->"


def body(reports: list[Path]) -> str:
    parts = [MARKER]
    for p in sorted(reports):
        parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts) + "\n"


def gh(args: list[str], payload: dict | None = None) -> str:
    cmd = ["gh", "api", *args]
    if payload is not None:
        cmd += ["--input", "-"]
    data = None if payload is None else json.dumps(payload)
    return subprocess.run(cmd, input=data, text=True, capture_output=True, check=True).stdout


def existing(repo: str, pr: int, api=gh) -> int | None:
    out = api(["--paginate", "--slurp", f"repos/{repo}/issues/{pr}/comments"])
    for page in json.loads(out or "[]"):
        for c in page:
            if MARKER in (c.get("body") or ""):
                return int(c["id"])
    return None


def publish(repo: str, pr: int, text: str, api=gh, dry_run: bool = False) -> tuple[str, str]:
    cid = existing(repo, pr, api)
    if cid is None:
        method, path = "POST", f"repos/{repo}/issues/{pr}/comments"
    else:
        method, path = "PATCH", f"repos/{repo}/issues/comments/{cid}"
    if not dry_run:
        api(["--method", method, path], {"body": text})
    return method, path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--comments", type=Path, help="dry run: a JSON list of comments instead of gh")
    ap.add_argument("reports", nargs="+", type=Path)
    a = ap.parse_args(argv)
    text = body(a.reports)
    api = gh
    if a.comments is not None:
        if not a.dry_run:
            ap.error("--comments needs --dry-run")
        api = lambda args, payload=None: json.dumps([json.loads(a.comments.read_text())])  # noqa: E731
    method, path = publish(a.repo, a.pr, text, api, a.dry_run)
    print(f"{'would ' if a.dry_run else ''}{method} {path} ({len(text)} chars)")
    if a.dry_run:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
