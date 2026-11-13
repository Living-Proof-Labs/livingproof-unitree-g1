# SPDX-License-Identifier: Apache-2.0
"""Stage canonical sources, generate task reference, and check the rendered manual."""

import hashlib
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/manual"
REPO = "https://github.com/Living-Proof-Labs/livingproof-unitree-g1"


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])


def main():
    source = OUT / "source"
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(ROOT / "docs", source)
    sha = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    identity = {"repository": REPO, "commit": sha, "dirty": dirty, "kind": "development"}
    guide = (ROOT / "README.md").read_text()
    guide = guide.replace("(docs/index.md)", "(index.md)")
    (source / "guide.md").write_text(guide)
    tasks = json.loads(run("task", "--list", "--json"))["tasks"]
    text = "# Tasks\n\nGenerated from the reviewed Taskfile with Task 3.54.0.\n\n"
    for task in tasks:
        text += f"## {task['name']}\n\n```text\n{run('task', '--summary', task['name'])}\n```\n\n"
    (source / "tasks.md").write_text(text)
    (source / "version.md").write_text(
        "# Version and source\n\n"
        f"Docs: development · [{sha[:7]}]({REPO}/tree/{sha})"
        f"{' (local modified preview)' if dirty else ''}.\n\n"
        f"Full source commit: `{sha}`.\n\n"
        "This manual describes this source tree, not an independently released package. "
        "No historical rendered manuals are maintained; "
        f"[release tags]({REPO}/tags) retain source history.\n\n"
        "Supported context: Python >=3.11, MuJoCo, and the model/controller and dependency "
        "releases recorded in the repository locks. See the guide for installation.\n"
    )
    brand = json.loads((ROOT / "docs-brand.lock.json").read_text())
    url = (
        f"https://raw.githubusercontent.com/{brand['repository']}/{brand['commit']}/{brand['path']}"
    )
    with urlopen(url, timeout=60) as response:
        css = response.read()
    if hashlib.sha256(css).hexdigest() != brand["sha256"]:
        raise ValueError("Shared brand checksum mismatch")
    (source / "assets").mkdir()
    (source / "assets/brand.css").write_bytes(css)
    config = yaml.safe_load((ROOT / "tools/docs/mkdocs.yml").read_text())
    config.update(docs_dir="source", site_dir="html")
    config["extra"]["source_commit"] = sha
    (OUT / "mkdocs.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    run(
        str(Path(__import__("sys").executable).with_name("zensical")),
        "build",
        "--clean",
        "-f",
        str(OUT / "mkdocs.yml"),
    )
    html = OUT / "html"
    (html / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
    pages = {p.resolve(): Page(p.read_text()) for p in html.rglob("*.html")}
    for path, page in pages.items():
        content = path.read_text()
        assert "assets/brand.css" in content, path
        assert 'data-md-color-primary="indigo"' not in content, path
        for href in page.links:
            link = urlsplit(href)
            if link.scheme or link.netloc or not link.path and not link.fragment:
                continue
            if link.path.startswith("/livingproof-unitree-g1/"):
                target = (
                    html / unquote(link.path.removeprefix("/livingproof-unitree-g1/"))
                ).resolve()
            else:
                target = (path.parent / unquote(link.path)).resolve() if link.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.exists(), (path, href)
            if link.fragment and target in pages:
                assert unquote(link.fragment) in pages[target].ids, (path, href)
    print(
        f"PASS: {len(pages)} rendered pages, local links/fragments and brand checks; source {sha}; dirty={dirty}"
    )


if __name__ == "__main__":
    main()
