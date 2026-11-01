# SPDX-License-Identifier: Apache-2.0
"""Robot and configuration profiles (robot-capability-specs v1.5.0), validated against
the pinned schemas, and the runner spec built from the selected configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from livingproof_sim_mujoco.walk import joint_dofs  # noqa: F401
import numpy as np
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from livingproof_sim_mujoco import adapters
from livingproof_sim_mujoco.policy import PolicySpec

from lpl_g1 import ROOT
from lpl_g1.fetch import load_pins, local, sha256
from lpl_g1.scene import Scene

SCHEMAS = {
    "robot": "specs/robots/robot-profile.schema.json",
    "configuration": "specs/robots/configuration-profile.schema.json",
    "manifest": "specs/robots/run-manifest.schema.json",
}


class Unsupported(Exception):
    """A scenario precondition is not met; ``precondition`` names it."""

    def __init__(self, precondition: str, detail: str):
        super().__init__(f"{precondition}: {detail}")
        self.precondition, self.detail = precondition, detail


def validate(kind: str, document: dict, root: Path = ROOT) -> None:
    loaded = {k: json.loads(local("specs", v, root).read_text()) for k, v in SCHEMAS.items()}
    registry = Registry().with_resources(
        (s["$id"], Resource.from_contents(s)) for s in loaded.values()
    )  # resolve cross-schema $refs from the pinned files, never the network
    validator = Draft202012Validator(loaded[kind], registry=registry)
    errors = sorted(validator.iter_errors(document), key=str)
    if errors:
        raise ValueError(
            f"{kind} does not match {SCHEMAS[kind]}: "
            + "; ".join(
                f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]
            )
        )


def robot(root: Path = ROOT) -> dict:
    doc = json.loads((root / "robot.json").read_text())
    validate("robot", doc, root)
    return doc


def configuration_path(name: str, root: Path = ROOT) -> Path:
    offered = {c["name"]: c["path"] for c in robot(root)["configurations"]}
    if name not in offered:
        raise Unsupported(
            "configuration_offered", f"{name!r} is not in robot.json {sorted(offered)}"
        )
    return root / offered[name]


def configuration(name: str, root: Path = ROOT) -> tuple[dict, str]:
    """The configuration profile and the sha256 of its file bytes."""
    path = configuration_path(name, root)
    doc = json.loads(path.read_text())
    validate("configuration", doc, root)
    if doc["name"] != name or doc["robot"] != robot(root)["id"]:
        raise ValueError(f"{path} names {doc['robot']}/{doc['name']}, expected {name}")
    return doc, sha256(path)


def artifact_path(pin: dict, root: Path = ROOT) -> Path:
    """The pins.json file matching a profile's pinned artifact; checks commit and sha256."""
    for group, g in load_pins(root)["groups"].items():
        if pin["repo"].endswith(g["repository"]) and pin["path"] in g["files"]:
            if g["commit"] != pin["commit"] or g["files"][pin["path"]]["sha256"] != pin["sha256"]:
                raise ValueError(f"{pin['path']}: profile pin differs from pins.json")
            path = local(group, pin["path"], root)
            if sha256(path) != pin["sha256"]:
                raise RuntimeError(f"{path} sha256 mismatch; run python -m lpl_g1.fetch")
            return path
    raise KeyError(f"{pin['repo']} {pin['path']} is not in pins.json")


@dataclass
class Controller:
    configuration: dict
    sha256: str
    policy_path: Path
    spec: PolicySpec
    initial_qpos: np.ndarray  # full qpos, base at the origin; spawn pose applied later


def controller(
    name: str, scene: Scene, root: Path = ROOT, actuation: str = "pd_torque"
) -> Controller:
    """Runner spec for one configuration; gains and limits come from the profile.

    ``actuation="position"`` (Playground only) drives the MJCF <position> actuators instead,
    used to show the in-step torque PD reproduces them."""
    conf, digest = configuration(name, root)
    s = conf["settings"]
    path = artifact_path(conf["controller"]["artifact"], root)
    model = scene.model
    qpos = model.key_qpos[model.key(s["initial_keyframe"]).id].copy()
    dt = float(model.opt.timestep)
    by_joint = {
        model.joint(model.actuator_trnid[i][0]).name: model.actuator(i).name
        for i in range(model.nu)
    }
    if s["adapter"] == "playground_joystick":
        joints = list(by_joint)
        g, lim = s["gains"], s["limits"]
        spec = adapters.playground_joystick(
            physics_dt=dt,
            ctrl_dt=s["ctrl_dt_s"],
            joints=joints,
            actuators=[by_joint[j] for j in joints],
            default_pose=[qpos[model.joint(j).qposadr[0]] for j in joints],
            action_scale=s["action_scale"],
            gait_hz=s["gait_hz"],
            gait_offsets=s["gait_offsets_rad"],
            linvel_sensor=s["observation"]["linvel_sensor"],
            gyro_sensor=s["observation"]["gyro_sensor"],
            imu_site=s["observation"]["imu_site"],
            actuation=actuation,
            kp=[g["kp"][j] for j in joints] if actuation == "pd_torque" else (),
            kd=[g["kd"][j] for j in joints] if actuation == "pd_torque" else (),
        )
        spec = PolicySpec(
            **{**spec.__dict__, "target_range": [lim["target_range_rad"][j] for j in joints]}
        )
    elif s["adapter"] == "rl_gym":
        if actuation != "pd_torque":
            raise ValueError("the rl_gym controller is a torque PD controller")
        legs, arms = s["leg_joints"], s["arm_waist_joints"]
        spec = adapters.rl_gym(
            s["deploy"],
            physics_dt=dt,
            joints=legs,
            actuators=[by_joint[j] for j in legs],
            hold_joints=arms,
            hold_actuators=[by_joint[j] for j in arms],
            gait_period=s["gait_period_s"],
        )
        spec = PolicySpec(**{**spec.__dict__, "first_tick": s["first_tick"]})
        for joint, target in zip(arms, s["deploy"]["arm_waist_target"]):
            qpos[model.joint(joint).qposadr[0]] = target
    else:
        raise ValueError(f"unknown adapter {s['adapter']!r}")
    qpos[0:2] = 0.0
    return Controller(conf, digest, path, spec, qpos)
