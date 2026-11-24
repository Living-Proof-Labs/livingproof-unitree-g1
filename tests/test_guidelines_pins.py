# SPDX-License-Identifier: Apache-2.0
"""collaboration-guidelines bumps move every consumer pin together and keep the release interface.

Proposed by Vera in the capability-verifier#72 review (guidelines bump checklist items 2, 3 and 6),
adapted for this repository. SHARED_RELEASE_* is the floor the pinned semantic-release.yml
must still declare, including the commit output. LIVINGPROOF_NETWORK=1 reads that workflow at
the pinned tag and parses its full interface, and callers are checked against those parsed
names. Update SHARED_RELEASE_* when a guidelines release drops or renames a recorded name.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

import pytest

REPO = "Living-Proof-Labs/collaboration-guidelines"
TAG = r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
USES = re.compile(
    rf"^\s+uses:\s*{re.escape(REPO)}/\.github/workflows/"
    rf"(?P<file>[\w.-]+\.ya?ml)@(?P<ref>\S+)\s*$",
    re.MULTILINE,
)
PRESET = re.compile(rf"^github>{REPO}:renovate-preset#(?P<ref>.+)$")
# Floor the pinned semantic-release.yml must still declare. expected-version and
# expected-commit are parsed from that file at the pinned tag.
SHARED_RELEASE_INPUTS = {"dry-run", "draft-release"}
SHARED_RELEASE_OUTPUTS = {
    "draft-created",
    "published",
    "version",
    "git-tag",
    "commit",
    "release-url",
}
_PINNED_SHARED: str | None = None


def _root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "renovate.json").is_file():
            return candidate
    raise AssertionError("could not find the repository root (renovate.json)")


ROOT = _root()


def workflow_path() -> Path | None:
    for name in (".github/workflows/verify.yml", ".github/workflows/release.yml"):
        path = ROOT / name
        if path.is_file() and USES.search(path.read_text()):
            return path
    return None


def guideline_calls(text: str | None = None) -> dict[str, re.Match[str]]:
    path = workflow_path()
    if text is None:
        if path is None:
            return {}
        text = path.read_text()
    calls: dict[str, re.Match[str]] = {}
    for match in USES.finditer(text):
        headers = re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", text[: match.start()], re.MULTILINE)
        assert headers, "expected a job name above the guidelines workflow call"
        calls[headers[-1]] = match
    return calls


def preset_refs(config: dict | None = None) -> list[str]:
    config = config or json.loads((ROOT / "renovate.json").read_text())
    return [m.group("ref") for m in map(PRESET.match, config.get("extends", [])) if m]


def pinned_versions(text: str | None = None, config: dict | None = None) -> set[str]:
    refs = set(preset_refs(config))
    if text is not None or workflow_path() is not None:
        refs |= {match.group("ref") for match in guideline_calls(text).values()}
    return refs


def _job_block(text: str, job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:\n", text, re.MULTILINE)
    assert match, f"job {job} not found"
    start = match.start()
    end = len(text)
    for later in re.finditer(r"^  [A-Za-z0-9_-]+:\s*$", text[match.end() :], re.MULTILINE):
        end = match.end() + later.start()
        break
    return text[start:end]


def _mapping(block: str, key: str, indent: int = 4) -> dict[str, str]:
    pad = " " * indent
    inner = " " * (indent + 2)
    found = re.search(rf"^{pad}{re.escape(key)}:\n((?:{inner}.*\n)*)", block, re.MULTILINE)
    if not found:
        return {}
    return dict(re.findall(rf"^{inner}([A-Za-z0-9_-]+):\s*(\S+)", found.group(1), re.MULTILINE))


def _workflow_call_keys(shared: str, section: str) -> set[str]:
    call = re.search(r"^  workflow_call:\n((?:    .*\n)+)", shared, re.MULTILINE)
    assert call, "expected a workflow_call interface"
    body = call.group(1)
    found = re.search(rf"^    {re.escape(section)}:\n((?:      .*\n)+)", body, re.MULTILINE)
    if not found:
        return set()
    return set(re.findall(r"^      ([A-Za-z0-9_-]+):", found.group(1), re.MULTILINE))


def pinned_semantic_release() -> str:
    """The semantic-release.yml of the guidelines tag this repo pins.

    Same network gate as the rest of this module: callers skip this unless
    LIVINGPROOF_NETWORK is set.
    """
    global _PINNED_SHARED
    if _PINNED_SHARED is None:
        refs = {match.group("ref") for match in guideline_calls().values()}
        (ref,) = refs
        url = (
            f"https://raw.githubusercontent.com/{REPO}/{ref}/.github/workflows/semantic-release.yml"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            _PINNED_SHARED = response.read().decode()
    return _PINNED_SHARED


def parsed_release_interface(shared: str) -> tuple[set[str], set[str]]:
    """Inputs and outputs declared by a semantic-release workflow file.

    The recorded floor must be present. Extra keys belong to the parsed
    interface and are what callers are checked against.
    """
    inputs = _workflow_call_keys(shared, "inputs")
    outputs = _workflow_call_keys(shared, "outputs")
    missing_inputs = SHARED_RELEASE_INPUTS - inputs
    missing_outputs = SHARED_RELEASE_OUTPUTS - outputs
    assert not missing_inputs, sorted(missing_inputs)
    assert not missing_outputs, sorted(missing_outputs)
    return inputs, outputs


def release_interface(shared: str | None = None) -> tuple[set[str], set[str]]:
    """Interface callers must stay inside.

    A provided workflow file, or the pinned file when the network gate is
    open, is parsed. Otherwise the recorded floor is the interface.
    """
    if shared is not None:
        return parsed_release_interface(shared)
    if os.environ.get("LIVINGPROOF_NETWORK"):
        return parsed_release_interface(pinned_semantic_release())
    return set(SHARED_RELEASE_INPUTS), set(SHARED_RELEASE_OUTPUTS)


def _assert_release_call(text: str, inputs: set[str], outputs: set[str]) -> None:
    calls = guideline_calls(text)
    assert calls, "expected a job that uses the shared semantic-release workflow"
    for name, match in calls.items():
        assert match.group("file") == "semantic-release.yml"
        block = _job_block(text, name)
        used_inputs = set(_mapping(block, "with"))
        assert used_inputs <= inputs, sorted(used_inputs - inputs)
        assert _mapping(block, "permissions").get("contents") == "write"
        used = set(re.findall(rf"needs\.{re.escape(name)}\.outputs\.([\w-]+)", text))
        assert used <= outputs, sorted(used - outputs)


def _workflow_call_document(inputs: set[str], outputs: set[str]) -> str:
    input_lines = "\n".join(f"      {name}:\n        type: string" for name in sorted(inputs))
    output_lines = "\n".join(
        f"      {name}:\n        value: ${{{{ jobs.release.outputs.{name} }}}}"
        for name in sorted(outputs)
    )
    return f"on:\n  workflow_call:\n    inputs:\n{input_lines}\n    outputs:\n{output_lines}\n"


def _caller(with_keys: dict[str, str], output: str) -> str:
    with_body = "\n".join(f"      {key}: {value}" for key, value in with_keys.items())
    with_block = f"    with:\n{with_body}\n" if with_keys else ""
    return (
        "jobs:\n"
        "  release:\n"
        "    permissions:\n"
        "      contents: write\n"
        f"    uses: {REPO}/.github/workflows/semantic-release.yml@v9.9.11\n"
        f"{with_block}"
        "  after:\n"
        "    needs: release\n"
        f"    if: needs.release.outputs.{output} == 'x'\n"
    )


def test_every_guidelines_pin_is_a_full_release_tag():
    refs = pinned_versions()
    assert refs, "expected a Renovate preset pin and/or a release workflow call"
    for ref in refs:
        assert re.fullmatch(TAG, ref), f"{ref} is not a full vX.Y.Z release tag"


def test_workflow_call_and_renovate_preset_name_one_guidelines_release():
    # A partial bump (one pin moved, the other left behind) fails here.
    # Repos that only pin the preset still need exactly one release tag.
    assert len(pinned_versions()) == 1, sorted(pinned_versions())


@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_mismatched_pins_are_reported():
    text = workflow_path().read_text()
    mutated = USES.sub(lambda m: m.group(0).replace(m.group("ref"), "v2.2.0", 1), text, count=1)
    preset = {"extends": [f"github>{REPO}:renovate-preset#v9.2.0"]}
    assert len(pinned_versions(mutated, preset)) == 2


@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_release_call_uses_only_the_shared_interface():
    text = workflow_path().read_text()
    inputs, outputs = release_interface()
    _assert_release_call(text, inputs, outputs)


@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_an_unknown_release_output_is_reported():
    text = workflow_path().read_text()
    mutated = (
        text + "\n# needs.release.outputs.tag\n"
        if "needs.release.outputs" in text
        else text.replace(
            "permissions:", "needs.release.outputs.tag: placeholder\n  permissions:", 1
        )
    )
    found = set(re.findall(r"needs\.release\.outputs\.([\w-]+)", mutated))
    assert "tag" in found
    inputs, outputs = release_interface()
    assert found - outputs >= {"tag"}
    with pytest.raises(AssertionError):
        _assert_release_call(mutated, inputs, outputs)


def test_recorded_floor_keeps_commit_and_the_original_inputs():
    assert SHARED_RELEASE_INPUTS == {"dry-run", "draft-release"}
    assert "expected-version" not in SHARED_RELEASE_INPUTS
    assert "expected-commit" not in SHARED_RELEASE_INPUTS
    assert SHARED_RELEASE_OUTPUTS == {
        "draft-created",
        "published",
        "version",
        "git-tag",
        "commit",
        "release-url",
    }


@pytest.mark.parametrize(
    ("dropped_input", "dropped_output", "missing"),
    [
        ("draft-release", None, "draft-release"),
        (None, "commit", "commit"),
    ],
)
def test_parsed_interface_must_include_the_recorded_names(dropped_input, dropped_output, missing):
    inputs = set(SHARED_RELEASE_INPUTS)
    outputs = set(SHARED_RELEASE_OUTPUTS)
    if dropped_input:
        inputs.remove(dropped_input)
    if dropped_output:
        outputs.remove(dropped_output)
    shared = _workflow_call_document(inputs, outputs)
    with pytest.raises(AssertionError, match=re.escape(missing)):
        parsed_release_interface(shared)


def test_callers_are_checked_against_the_parsed_interface():
    inputs = set(SHARED_RELEASE_INPUTS) | {"expected-version", "expected-commit"}
    outputs = set(SHARED_RELEASE_OUTPUTS) | {"later-out"}
    parsed_inputs, parsed_outputs = parsed_release_interface(
        _workflow_call_document(inputs, outputs)
    )
    assert parsed_inputs == inputs
    assert parsed_outputs == outputs
    assert "expected-commit" not in SHARED_RELEASE_INPUTS
    assert "later-out" not in SHARED_RELEASE_OUTPUTS
    _assert_release_call(
        _caller({"expected-version": "1.2.3", "expected-commit": "abc"}, "later-out"),
        parsed_inputs,
        parsed_outputs,
    )
    with pytest.raises(AssertionError):
        _assert_release_call(
            _caller({"not-a-real-input": "true"}, "later-out"),
            parsed_inputs,
            parsed_outputs,
        )


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_recorded_interface_matches_the_pinned_shared_workflow():
    # The recorded names are a floor. Callers are held to the file at the pin.
    shared = pinned_semantic_release()
    parsed_inputs, parsed_outputs = parsed_release_interface(shared)
    inputs, outputs = release_interface()
    assert (inputs, outputs) == (parsed_inputs, parsed_outputs)
    assert SHARED_RELEASE_INPUTS <= inputs
    assert SHARED_RELEASE_OUTPUTS <= outputs
    assert {"expected-version", "expected-commit"} <= inputs
    assert "commit" in outputs
    path = workflow_path()
    assert path is not None
    _assert_release_call(path.read_text(), inputs, outputs)
