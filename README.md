# livingproof-unitree-g1

**Start here: documentation** — [manual source/preview](docs/index.md).
The deployment workflow is proposed; the live manual is not yet verified.

Living Proof Labs robot package for the Unitree G1 humanoid, in the [robot-capability-specs](https://github.com/Living-Proof-Labs/robot-capability-specs) v2.0.2 format (`specs/robots/`). It pins an official model and two published, ready-made walking controllers, and exposes them through a MuJoCo adapter.

Living Proof Labs does not train controllers. A robot is checked only with a controller its maker or a maintained project has already published, run unchanged.

This package publishes the adapter and the configuration profiles. It does not select a world or a scenario. [livingproof-verification](https://github.com/Living-Proof-Labs/livingproof-verification) pins a released tag of this repository and owns those integration pins.

## Files

- `robot.json`: robot profile (`livingproof-robot-profile/0.1`). Covers the model source pinned by commit and sha256, the dimensions (each `measured`, `published` or `chosen`), the offered configurations, and `simulators: [mujoco]`.
- `configurations/<name>.json`: configuration profiles (`livingproof-configuration-profile/0.1`). `stock` is version 1.1.0 and `playground` is version 1.0.0. Each holds its controller artifact, gains, limits and locomotion heights. Tests check the gains and limits against the pinned upstream files.
- `pins.json`: every upstream file by repository, commit and sha256, including the robot and configuration schemas at v2.0.2 (`16524cb`). The pytest session (`livingproof_sim_mujoco.pins.fetch`) downloads them into `.cache/` and verifies each one. Nothing is vendored.
- `runner.json`: runner settings that are not part of the spec, marked `chosen`: the sample rate and the stand length. A trial this adapter starts itself uses the configuration's nominal pose. There is no seed and no noise.
- `src/lpl_g1/`: the adapter, profile and schema validation, the robot scene, and the publisher reference check. Nothing here grades, and nothing here loads a world.
- `pyproject.toml` pins `livingproof-sim-mujoco` at tag `v2.0.3`. The adapter tests import that release. The pin is a tag, not a branch.

| Configuration | Controller | Publisher | License |
|---|---|---|---|
| `stock` | unitree_rl_gym `deploy/pre_train/g1/motion.pt` @ 276801e (TorchScript LSTM, torch CPU), with gains from `deploy_real/configs/g1.yaml` | Unitree Robotics | BSD-3-Clause |
| `playground` (default) | mujoco_playground `g1_policy.onnx` @ 12ad7f3 (onnxruntime), with gains from the Playground G1 MJCF | Google DeepMind | Apache-2.0 |

The robot model is the Playground G1 MJCF (Apache-2.0) with menagerie `unitree_g1` meshes @ 1b86ece (BSD-3-Clause, © Unitree). The model is the same for both configurations; selecting a configuration is the toggle.

## Actuation

Both configurations use the same in-step PD. Every 0.002 s physics step runs `mj_step1`, writes `kp (q* - q) - kd dq` from the current state into motor actuators, then runs `mj_step2`. This is the fix from [unitree-foxglove-sdk](https://github.com/Living-Proof-Labs/unitree-foxglove-sdk) (`MotorFirmwareBridge`). With the Playground gains, a test shows it reproduces the MJCF `<position>` actuators to 1e-6. The runner is `OnnxPolicy` in livingproof-sim-mujoco.

## Tasks

Needs [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev) and OSMesa (`libosmesa6`).

```sh
uv sync && uv run pytest tests                           # uv sync + tests (the session fetches and verifies every pin)
uv run ruff check . && uv run ruff format --check .     # ruff
uv run pytest tests                                      # schemas, gains vs upstream, nominal pose, PD
task reference                                           # tick-by-tick match against the publishers' loops
task profile:measure                                     # width, length, mass for robot.json
```

The adapter entry point is `lpl_g1.adapter:ROBOT` (`[tool.livingproof] adapter`).

GitHub Actions runs that pytest command and ruff on `ubuntu-24.04` and `ubuntu-24.04-arm` for pull requests and pushes to `main`.
