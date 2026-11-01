# livingproof-unitree-g1

Living Proof Labs robot package for the Unitree G1 humanoid, in the [robot-capability-specs](https://github.com/Living-Proof-Labs/robot-capability-specs) v1.5.0 format (`specs/robots/`). It pins an official model and two published, ready-made walking controllers, and runs scenario `walk-hallway@0.1.0` in MuJoCo.

Living Proof Labs does not train controllers. A robot is checked only with a controller its maker or a maintained project has already published, run unchanged.

## Files

- `robot.json`: robot profile (`livingproof-robot-profile/0.1`). Covers the model source pinned by commit and sha256, the dimensions (each `measured`, `published` or `chosen`), the offered configurations, and `simulators: [mujoco]`.
- `configurations/<name>.json`: configuration profiles (`livingproof-configuration-profile/0.1`), version 1.0.0. Each holds its controller artifact, gains, limits and locomotion heights. Tests check the gains and limits against the pinned upstream files.
- `pins.json`: every upstream file by repository, commit and sha256, including the specs schemas and scenario at v1.5.0 (`1df6eaa`). `task fetch` downloads them into `.cache/` and verifies each one. Nothing is vendored.
- `runner.json`: runner settings that are not part of the spec, all marked `chosen`: the initial-state noise, the sample rate and the stand length.
- `src/lpl_g1/`: profiles and schema validation, the scenario (parameters, preconditions, manifest, evaluation, grade, comparison), the scene, wall clearance and the walk.

| Configuration | Controller | Publisher | License |
|---|---|---|---|
| `stock` | unitree_rl_gym `deploy/pre_train/g1/motion.pt` @ 276801e (TorchScript LSTM, torch CPU), with gains from `deploy_real/configs/g1.yaml` | Unitree Robotics | BSD-3-Clause |
| `playground` (default) | mujoco_playground `g1_policy.onnx` @ 12ad7f3 (onnxruntime), with gains from the Playground G1 MJCF | Google DeepMind | Apache-2.0 |

The robot model is the Playground G1 MJCF (Apache-2.0) with menagerie `unitree_g1` meshes @ 1b86ece (BSD-3-Clause, © Unitree). The model is the same for both configurations; selecting a configuration is the toggle.

## Actuation

Both configurations use the same in-step PD. Every 0.002 s physics step runs `mj_step1`, writes `kp (q* - q) - kd dq` from the current state into motor actuators, then runs `mj_step2`. This is the fix from [unitree-foxglove-sdk](https://github.com/Living-Proof-Labs/unitree-foxglove-sdk) (`MotorFirmwareBridge`). With the Playground gains, a test shows it reproduces the MJCF `<position>` actuators to 1e-6. The runner is `OnnxPolicy` in livingproof-sim-mujoco.

## walk-hallway@0.1.0

Every parameter comes from the pinned scenario, the robot and configuration profiles, and the world:

| Parameter | Value |
|---|---|
| command | [0.5, 0, 0] m/s |
| `time_limit` | 2 × 5.5 / 0.5 + 5 = 27 s |
| exit | base crosses x = 3.0 |
| `fall_height` | `locomotion.fall_base_height` (0.5 m in both profiles) |
| `lateral_limit` | 1.5 / 2 − 0.52 / 2 = 0.49 m |
| trials | seeds 1–20, no retries |
| `required_passes` | 18 |

- **Preconditions:** `fits_width`, `configuration_offered`, `velocity_interface` and `simulator_declared` are checked first. If any is unmet, the run is graded `UNSUPPORTED` and no trial runs.
- **Run manifest:** written to `run-manifest.json` before the first trial and validated against the pinned schema. It records the configuration's name, version and file sha256. The runner refuses to start with uncommitted changes, so the robot commit it cites is exact.
- **Evaluation:** each trial records its trajectory at 50 Hz (`trajectories/seed-NN.json`), and the metrics are evaluated from that recording afterwards.
- **Wall contacts:** the policy model collides only at its feet, so the walls are visual only. A wall contact is a sample whose world-AABB gap between any robot geom and a wall box is ≤ 0. This is conservative.
- **Trial outcomes:** a trial that cannot complete is `ERROR`, never `FAIL`.
- **Grade:** `PASS` with at least 18 passes. `INSUFFICIENT_EVIDENCE` applies only when `ERROR` trials could change the answer; otherwise the grade is `FAIL`.
- **Comparison:** the higher grade wins first; otherwise non-overlapping Wilson 95% intervals decide; otherwise the result is "no clear winner".

## Tasks

Needs [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev) and OSMesa (`libosmesa6`) for recordings. Until the runner is released, it is referenced by path from `../livingproof-sim-mujoco-onnx` (branch `feat/onnx-policy-runner`).

```sh
task setup                         # uv sync + fetch
task check                         # ruff + profiles against the pinned schemas
task test                          # schemas, gains vs upstream, parameters, evaluation, grade, PD
task walk CONFIGURATION=stock      # one configuration, seeds 1-20, recordings
task stand CONFIGURATION=playground  # zero-command standing check, 10 s
task walk:all                      # both, then the comparison
task profile:measure               # width, length, mass for robot.json
```

Results go to `artifacts/<configuration>/`:

- `run-manifest.json`
- `result.json`: the grade, k/N, Wilson 95% interval and every trial (seed, outcome, first failing metric, metrics). Each trial echoes the configuration name and sha256.
- `seed-NN.mp4` and `trajectories/seed-NN.json` for each seed
- `stand.json`

The comparison goes to `artifacts/comparison.json`. Artifacts are never committed.
