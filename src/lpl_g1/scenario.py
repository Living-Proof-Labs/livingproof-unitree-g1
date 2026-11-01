# SPDX-License-Identifier: Apache-2.0
"""walk-hallway@0.1.0 from robot-capability-specs v1.5.0: parameters, preconditions,
run manifest, metric evaluation from a recorded trajectory, grade and comparison."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import mujoco

from lpl_g1 import ROOT
from lpl_g1.fetch import load_pins, local, sha256
from lpl_g1.profiles import Unsupported, configuration, robot, validate
from lpl_g1.scene import WORLD, load_world
from lpl_g1.stats import wilson

SCENARIO = "walk-hallway@0.1.0"
COMMAND_SPEED = 0.5  # m/s, shared
WALL_CLEARANCE = {"humanoid": 0.05, "quadruped": 0.10}
TRIALS, SEEDS, REQUIRED_PASSES = 20, list(range(1, 21)), 18
SPAWN, EXIT = "spawn-entrance", "exit"
GRADE_ORDER = ["PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "UNSUPPORTED", "ERROR"]


def place(world: dict, name: str) -> dict:
    return next(p for p in world["places"] if p["name"] == name)


def parameters(name: str, root: Path = ROOT) -> dict:
    """Acceptance parameters resolved from the scenario, robot, configuration and world."""
    rob, (conf, _) = robot(root), configuration(name, root)
    world = load_world(root)
    route = place(world, EXIT)["reference"][0] - place(world, SPAWN)["position"][0]
    clear = world["dimensions"]["clear_width"]
    width = rob["dimensions"]["width"]["value"]
    return {
        "command_speed": COMMAND_SPEED,
        "route_length": route,
        "time_limit": 2 * route / COMMAND_SPEED + 5,
        "fall_height": conf["locomotion"]["fall_base_height"]["value"],
        "wall_clearance": WALL_CLEARANCE[rob["class"]],
        "lateral_limit": round(clear / 2 - width / 2, 6),
        "exit_x": place(world, EXIT)["reference"][0],
        "spawn": SPAWN,
    }


def preconditions(name: str, root: Path = ROOT) -> list[dict]:
    """Checked before any trial; a failed one makes the grade UNSUPPORTED."""
    rob = robot(root)
    world = load_world(root)
    width = rob["dimensions"]["width"]["value"]
    clear = world["dimensions"]["clear_width"]
    out = []
    need = width + 2 * WALL_CLEARANCE[rob["class"]]
    out.append(
        ("fits_width", need <= clear, f"{width} + 2 x clearance = {need:.2f} m vs {clear} m")
    )
    try:
        conf, _ = configuration(name, root)
        out.append(("configuration_offered", True, name))
        iface = conf["controller"]["interface"]
        out.append(("velocity_interface", "velocity_command" in iface, f"interface {iface}"))
    except Unsupported as e:
        out.append((e.precondition, False, e.detail))
    out.append(
        ("simulator_declared", "mujoco" in rob["simulators"], f"simulators {rob['simulators']}")
    )
    return [{"name": n, "met": bool(ok), "detail": d} for n, ok, d in out]


def git_head(path: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--", ".", ":!artifacts"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return head, dirty


def manifest(name: str, run_id: str, root: Path = ROOT, runner_root: Path | None = None) -> dict:
    """The run manifest, written before the first trial."""
    import livingproof_sim_mujoco

    rob, (conf, conf_sha) = robot(root), configuration(name, root)
    spec = load_pins(root)["groups"]["specs"]
    world = load_world(root)
    runner_root = runner_root or Path(livingproof_sim_mujoco.__file__).resolve().parents[2]
    adapter_commit, adapter_dirty = git_head(runner_root)
    robot_commit, robot_dirty = git_head(root)
    p = parameters(name, root)
    doc = {
        "format": "livingproof-run-manifest/0.1",
        "run_id": run_id,
        "specification": {"tag": spec["tag"], "commit": spec["commit"]},
        "scenario": SCENARIO,
        "world": {
            "id": world["id"],
            "version": world["content_version"],
            "sha256": sha256(local("worlds", WORLD, root)),
        },
        "simulator": {
            "engine": "mujoco",
            "version": mujoco.__version__,
            "adapter": "livingproof-sim-mujoco",
            "adapter_commit": adapter_commit,
        },
        "robot": {
            "id": rob["id"],
            "repo": "https://github.com/Living-Proof-Labs/livingproof-unitree-g1",
            "commit": robot_commit,
            "profile_sha256": sha256(root / "robot.json"),
        },
        "configuration": {"name": name, "version": conf["version"], "sha256": conf_sha},
        "parameters": p,
        "trials": {"count": TRIALS, "seeds": SEEDS, "required_passes": REQUIRED_PASSES},
    }
    validate("manifest", doc, root)
    trees = {"robot_repo_dirty": robot_dirty, "adapter_dirty": adapter_dirty}
    return doc, trees


def evaluate(samples: list[list[float]], p: dict) -> dict:
    """Metrics from a recorded trajectory of [t, x, y, z, wall_gap] samples."""
    exit_t = next((s[0] for s in samples if s[1] >= p["exit_x"]), None)
    contacts, prev = 0, False
    for s in samples:
        touching = s[4] <= 0.0
        contacts += touching and not prev
        prev = touching
    first = {
        "reached_exit": None if exit_t is not None else samples[-1][0],
        "time_to_exit": exit_t if exit_t is not None and exit_t > p["time_limit"] else None,
        "wall_contacts": next((s[0] for s in samples if s[4] <= 0.0), None),
        "min_base_height": next((s[0] for s in samples if s[3] < p["fall_height"]), None),
        "max_abs_lateral": next((s[0] for s in samples if abs(s[2]) > p["lateral_limit"]), None),
    }
    failing = sorted((t, m) for m, t in first.items() if t is not None)
    return {
        "metrics": {
            "reached_exit": exit_t is not None,
            "time_to_exit": exit_t,
            "wall_contacts": contacts,
            "min_base_height": round(min(s[3] for s in samples), 4),
            "max_abs_lateral": round(max(abs(s[2]) for s in samples), 4),
        },
        "outcome": "FAIL" if failing else "PASS",
        "first_failing_metric": failing[0][1] if failing else None,
        "first_failure_time": failing[0][0] if failing else None,
    }


def grade(outcomes: list[str], required: int = REQUIRED_PASSES) -> str:
    """PASS at >= required passes. ERROR trials make it INSUFFICIENT_EVIDENCE only when
    they could change the answer (passes + errors would reach the bar)."""
    passes, errors = outcomes.count("PASS"), outcomes.count("ERROR")
    if passes >= required:
        return "PASS"
    return "INSUFFICIENT_EVIDENCE" if passes + errors >= required else "FAIL"


def compare(results: list[dict]) -> dict:
    """Higher grade first, then non-overlapping Wilson intervals, else no clear winner."""
    rows = [
        {
            k: r[k]
            for k in (
                "configuration",
                "configuration_sha256",
                "passed",
                "trials",
                "pass_rate",
                "wilson95",
                "grade",
            )
        }
        for r in results
    ]
    rank = sorted(rows, key=lambda r: GRADE_ORDER.index(r["grade"]))
    winner, rule = "no clear winner", "no clear winner"
    a, b = rank[0], rank[1]
    if GRADE_ORDER.index(a["grade"]) < GRADE_ORDER.index(b["grade"]):
        winner, rule = a["configuration"], "higher grade"
    else:
        hi, lo = sorted(rows, key=lambda r: -r["pass_rate"])
        if hi["wilson95"][0] > lo["wilson95"][1]:
            winner, rule = hi["configuration"], "non-overlapping pass-rate intervals"
    return {
        "scenario": SCENARIO,
        "dimension": "configuration",
        "runs": rows,
        "better": winner,
        "rule": rule,
    }


def interval(k: int, n: int) -> list[float]:
    lo, hi = wilson(k, n)
    return [round(lo, 4), round(hi, 4)]


def dump(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
