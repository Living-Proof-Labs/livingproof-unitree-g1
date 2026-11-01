# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pr_comment", ROOT / "tools" / "pr_comment.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


class Fake:
    def __init__(self, comments):
        self.comments, self.calls = comments, []

    def __call__(self, args, payload=None):
        self.calls.append((args, payload))
        return json.dumps([self.comments]) if args[0] == "--paginate" else "{}"


def _reports(tmp_path):
    out = []
    for name in ("stock", "playground"):
        p = tmp_path / f"capability-report-{name}" / "report.md"
        p.parent.mkdir()
        p.write_text(f"# Capability report: {name}\n\nFAIL\n")
        out.append(p)
    return out


def test_body_is_the_reporter_markdown_under_the_marker(tmp_path):
    text = pc.body(_reports(tmp_path))
    assert text.startswith(pc.MARKER + "\n\n")
    assert text.index("playground") < text.index("stock")


def test_creates_when_no_marked_comment(tmp_path):
    api = Fake([{"id": 1, "body": "unrelated"}])
    assert pc.publish("o/r", 3, "x", api) == ("POST", "repos/o/r/issues/3/comments")
    assert api.calls[-1] == (["--method", "POST", "repos/o/r/issues/3/comments"], {"body": "x"})


def test_updates_the_marked_comment_in_place(tmp_path):
    api = Fake([{"id": 1, "body": "hi"}, {"id": 42, "body": pc.MARKER + "\nold"}])
    assert pc.publish("o/r", 3, "x", api) == ("PATCH", "repos/o/r/issues/comments/42")
    assert len(api.calls) == 2


def test_dry_run_writes_nothing(tmp_path):
    api = Fake([])
    pc.publish("o/r", 3, "x", api, dry_run=True)
    assert len(api.calls) == 1  # the read only
