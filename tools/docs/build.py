# SPDX-License-Identifier: Apache-2.0
"""Stage canonical sources, generate task reference, and check the rendered manual."""

import hashlib
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import urlopen

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_identity import identity as docs_identity, seal

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
        self.stylesheets = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.stylesheets.append(attrs["href"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])


def stage_brand(source, css, expected_digest):
    """Verify pinned bytes and version its URL by its complete content digest."""
    digest = hashlib.sha256(css).hexdigest()
    if digest != expected_digest:
        raise ValueError("Shared brand checksum mismatch")
    asset = f"assets/brand.css?v={digest}"
    (source / "assets").mkdir(exist_ok=True)
    # Keep a stable file so cached old HTML cannot reference a removed asset.
    (source / "assets/brand.css").write_bytes(css)
    return asset


def check_brand(html, page_path, page, asset, expected_digest):
    """Check the actual stylesheet reference and bytes, including nested pages."""
    matches = []
    for href in page.stylesheets:
        url = urlsplit(href)
        if url.scheme or url.netloc:
            continue
        if url.path.startswith("/livingproof-unitree-g1/"):
            target = (html / unquote(url.path.removeprefix("/livingproof-unitree-g1/"))).resolve()
        else:
            target = (page_path.parent / unquote(url.path)).resolve()
        if target == (html / urlsplit(asset).path).resolve() and parse_qs(url.query) == {
            "v": [expected_digest]
        }:
            matches.append(target)
    assert len(matches) == 1, (page_path, "missing or duplicate pinned brand stylesheet")
    assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == expected_digest


def main():
    release = docs_identity(ROOT, REPO.removeprefix("https://github.com/"), cached=True)
    label = release["label"]
    source = OUT / "source"
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(ROOT / "docs", source)
    sha = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    identity = {"repository": REPO, "commit": sha, "dirty": dirty, "kind": "development"}
    identity.update(release)
    identity["sources"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "README.md", ROOT / "Taskfile.yml", *sorted((ROOT / "docs").rglob("*.md"))]
    }
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
        f"Docs: {label} · [{sha[:7]}]({REPO}/tree/{sha})"
        f"{' (local modified preview)' if dirty else ''}.\n\n"
        f"Full source commit: `{sha}`.\n\n"
        "This manual describes this source tree, not an independently released package. "
        "No historical rendered manuals are maintained; "
        f"[release tags]({REPO}/tags) retain source history.\n\n"
        "Supported context: Python >=3.11, MuJoCo, and the model/controller and dependency "
        "releases recorded in the repository locks. See the guide for installation.\n"
    )
    with (source / "version.md").open("a") as version_page:
        version_page.write(
            (
                f"\n\n[Published source release: {label}]({release['release_url']})"
                if release["kind"] == "release"
                else ""
            )
            + "\n\n```json\n"
            + json.dumps(release, indent=2)
            + "\n```\n"
        )
    brand = json.loads((ROOT / "docs-brand.lock.json").read_text())
    url = (
        f"https://raw.githubusercontent.com/{brand['repository']}/{brand['commit']}/{brand['path']}"
    )
    with urlopen(url, timeout=60) as response:
        css = response.read()
    brand_asset = stage_brand(source, css, brand["sha256"])
    identity["brand_asset"] = brand_asset
    identity["brand_sha256"] = brand["sha256"]
    config = yaml.safe_load((ROOT / "tools/docs/mkdocs.yml").read_text())
    config.update(
        docs_dir="source",
        site_dir="html",
        repo_url=f"{REPO}/tree/{sha}",
        repo_name=f"Source · {label}",
    )
    config["extra_css"] = [
        brand_asset if p == "assets/brand.css" else p for p in config.get("extra_css", [])
    ]
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
        check_brand(html, path, page, brand_asset, brand["sha256"])
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
    seal(html, release)
    print(
        f"PASS: {len(pages)} rendered pages, local links/fragments and brand checks; source {sha}; dirty={dirty}"
    )


if __name__ == "__main__":
    main()
