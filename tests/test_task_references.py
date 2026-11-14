# SPDX-License-Identifier: Apache-2.0
"""Every `task <name>` the README, docs and profiles cite is a task in Taskfile.yml.

A citation is a code span starting with `task ` or a profile ``ref`` that starts with it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tasks() -> set[str]:
    text = (ROOT / "Taskfile.yml").read_text(encoding="utf-8")
    body = text.split("\ntasks:\n", 1)[1]
    return set(re.findall(r"^  ([\w:.-]+):\s*$", body, re.M))


def test_cited_tasks_exist():
    tasks = _tasks()
    sources = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    sources += sorted((ROOT / "configurations").glob("*.json"))
    missing = []
    for path in sources:
        for name in re.findall(
            r'(?:`|"ref": ")task ([a-z][\w:.-]*\w)', path.read_text(encoding="utf-8")
        ):
            if name not in tasks:
                missing.append(f"{path.relative_to(ROOT)}: task {name}")
    assert missing == []
