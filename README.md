# livingproof-unitree-g1

Living Proof Labs robot package for the Unitree G1 humanoid, in the [robot-capability-specs](https://github.com/Living-Proof-Labs/robot-capability-specs) v2.0.2 format (`specs/robots/`). It pins an official model and two published, ready-made walking controllers, and runs scenario `walk-hallway@0.1.0` in MuJoCo.

Living Proof Labs does not train controllers. A robot is checked only with a controller its maker or a maintained project has already published, run unchanged.

## Files

- `robot.json`: robot profile (`livingproof-robot-profile/0.1`). Covers the model source pinned by commit and sha256, the dimensions (each `measured`, `published` or `chosen`), the offered configurations, and `simulators: [mujoco]`.
- `configurations/<name>.json`: configuration profiles (`livingproof-configuration-profile/0.1`), version 1.0.0. Each holds its controller artifact, gains, limits and locomotion heights. Tests check the gains and limits against the pinned upstream files.
- `pins.json`: every upstream file by repository, commit and sha256, including the specs schemas and scenario at v2.0.2 (`16524cb`). the pytest session (livingproof_sim_mujoco.pins.fetch) downloads them into `.cache/` and verifies each one. Nothing is vendored.
- `runner.json`: runner settings that are not part of the spec, all marked `chosen`: the sample rate and the stand length. Trials start at the exact nominal pose. There is no seed and no noise.
- `verification.lock.json`: what a verification run uses. Its `releases` block is the one place for the specs, runner, verifier and reporter refs (tag and commit). It also holds the world, each configuration's name, version and sha256, and the trial plan (`trials.count` 3, 0.5 m/s, no seeds and no `required_passes`). Both `stock` and `playground` are in the lock, so this repository's verification runs both.
- `src/lpl_g1/`: profiles and schema validation, scenario parameters and preconditions, the scene, the standing check and `evidence` (sim observations for the verifier). Nothing here grades.
- `task repin` (capability-verifier's `capability_verifier.repin`, at the locked verifier commit): fills `PENDING_RELEASE` refs from real release tags and checks that none remain.
- `.github/workflows/verify.yml`: sim evidence → capability verifier → capability report.

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
| trials | `trials.count` 3 identical trials from the exact nominal pose |
| start | exact nominal pose at `spawn-entrance` (a `start_offset` would be one declared comparison change) |

The specification's default count is 1. This package uses 3 so a published run can show that the identical trials agree. Evidence 2.0.0 has no seeds and no `required_passes`. The run passes when all `trials.count` trials succeed.

Trial plan, scenario commit and release refs are read from `verification.lock.json`.

- **Evidence only.** `python -m livingproof_sim_mujoco.evidence lpl_g1.adapter:ROBOT <configuration> <out>` runs `trials.count` identical trials through the runner's
  `observations.observe` and writes one observation per trial (`trials/trial-NN.json`: base trace, maximum lateral offset,
  minimum base height, exit time and real-geometry wall contacts). It also writes a
  `livingproof-hallway-evidence/2.0.0` manifest that echoes the lock and records the
  preconditions, and a `livingproof-run-manifest/0.1` before the first trial. No thresholds are applied here.
  Exit 4 means the identical trials disagreed (a test-system reproducibility finding, not a robot grade).
- **Assessment:** the capability verifier's hallway evaluators (exit reached, lateral limit,
  wall contact, fall) assess each trial with structured observed/limit/time failures.
- **Grade:** the capability report applies `all-required-v1` per trial. The run passes when all
  `trials.count` trials succeed.

## Tasks

Needs [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev) and OSMesa (`libosmesa6`).

```sh
uv sync && uv run pytest tests                           # uv sync + tests (the session fetches and verifies every pin)
uv run ruff check . && uv run ruff format --check .                           # ruff + profiles against the pinned schemas
uv run pytest tests                            # schemas, gains vs upstream, parameters, lock, PD
task reference                       # tick-by-tick match against the publishers' loops
uv run python -m livingproof_sim_mujoco.evidence lpl_g1.adapter:ROBOT stock artifacts/evidence/stock  # sim evidence, trials.count from the lock (CI's command)
task stand CONFIGURATION=playground  # zero-command standing check, 10 s
task repin:check                     # fails while any release ref is PENDING_RELEASE
task repin -- runner=vX.Y.Z verifier=vX.Y.Z reporter=vX.Y.Z   # e.g. runner=v2.0.3, the current pin
task profile:measure                 # width, length, mass for robot.json
```

Every release ref in `verification.lock.json` is currently pinned (specs v2.0.2, runner v2.0.3,
verifier v1.16.3, reporter v3.4.0). The evidence command refuses if a ref is ever `PENDING_RELEASE`; for a local run
before a release exists, pass `--allow-pending` and the manifest lists what was pending.
Evidence goes to `artifacts/evidence/<configuration>/` (`manifest.json`, `run-manifest.json`, `trials/trial-NN.json`).
Artifacts are never committed.

The runner pin is the pyproject tag on the `livingproof-sim-mujoco` source,
`verification.lock.json` `releases.runner.tag` and `releases.runner.commit`, then `uv lock`.
`task repin -- runner=vX.Y.Z` rewrites that tag and the lock.

## Verification workflow

`verify.yml` reads every ref from `verification.lock.json`. It checks that none is
`PENDING_RELEASE`, records evidence per configuration, assesses it with the locked verifier
release and grades it with the locked capability-report image. It uploads `evidence-*`,
`verified-*` and `capability-report-*` artifacts and writes the markdown report into the job
summary. A capability `FAIL` is the result, not a CI failure.

