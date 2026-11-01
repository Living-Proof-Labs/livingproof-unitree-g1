# livingproof-unitree-g1

Living Proof Labs robot package: Unitree G1 humanoid. It pins an official model and a published, ready-made walking controller so the robot can be validated in the livingproof simulators (MuJoCo first, Gazebo where feasible) and in the [livingproof-worlds](https://github.com/Living-Proof-Labs/livingproof-worlds) test worlds.

Living Proof Labs does not train controllers. A robot is validated only with a walking controller or policy its maker or a maintained project has already published.

**Status: stub.** The layout below is the plan; content comes next.

## Planned layout

- `manifest.json`: one record with the fields `manufacturer`, `model`, `class` (humanoid or quadruped), `controller`, `source` and `license`, each pointing at an upstream release or commit.
- **Pinned references, not vendored:** the model and controller are listed by URL, version or commit, and sha256. They are downloaded and checksum-verified at run time into an ignored folder.
- **Per-simulator settings:** for example `simulators/mujoco/` and `simulators/gazebo/`, holding only what each engine needs (timestep, gains, control rate, observation mapping).
- Tests that check the manifest and the pinned checksums.

Planned sources: model from mujoco_menagerie `unitree_g1` (BSD-3-Clause); controller from Unitree's published walking policy (candidates: unitree_rl_lab / unitree_rl_mjlab `velocity/v0` ONNX, MuJoCo Playground `g1_policy.onnx`). Final pins are recorded in the manifest.

## Rules

- No generated files are committed: no downloaded models or policies, no run output, no evidence. They are produced by tasks and published as CI artifacts or release assets.
- Upstream licenses are recorded in the manifest and honoured. This repository's own code is Apache-2.0.
