# G1 collision and wall-measurement coverage

## Scope and source

This audit preserves the publisher model. It does **not** add collision shapes,
change contact pairs, or claim new robot trial results. Both `stock` and
`playground` use the same compiled Playground model; they differ in controller,
gains and initial pose, not in collision coverage.

The audited model is `g1_mjx_feetonly.xml` from
`google-deepmind/mujoco_playground` commit
`12ad7f3842791ced39f27e5ac96d066bbe361653`, SHA-256
`fae73d06ae6957ab38e59c67cac5b3665db6de27cd741602dd93951e3f82b72c`.
Mesh assets are pinned to Menagerie commit
`1b86ece576591213e2b666ebf59508454200ca97`. The runner used for this audit is
`livingproof-sim-mujoco` v3.4.3. These are audit inputs, not new integration pins.

## Three different meanings of coverage

1. **Physics contacts:** all robot geoms have zero `contype` and `conaffinity`.
   Only the model's explicit pairs enter the contact solver. They pair each foot
   with the floor, each hand with its thigh, and the two feet with each other.
2. **Wall measurement:** the runner selects collision-class group 3, including
   shins that are not in active pairs. For a body with no such geometry, it
   measures that body's visual geoms and records a `visual_fallback_bodies`
   warning. Straight-hallway `Reach` and corner `RouteReach` use this same
   selection. A visual-only torso or shoulder is therefore **not invisible**
   to current wall-intrusion measurement.
3. **Physical wall response:** the simulator's world walls are deliberately
   visual-only measurement boundaries. They exert no wall-contact force.
   Adding robot collision primitives alone would not change that.

The original issue's assertion that only base drift could detect torso or arm
intrusion predates the runner's visual fallback. It is not true of v3.4.3.
A measured geometric intrusion is not proof of a simulated physical collision.

## Body-by-body census and proposed exclusions

`left/right` below lists each side explicitly as two bodies. Every one of the
30 robot bodies has geometry selected for measurement. Eight have collision-class
primitives; the other 22 use visual fallback.

| Bodies | Measurement representation | Proposed disposition |
| --- | --- | --- |
| `pelvis` | Three visual geoms | Keep visual fallback; retain publisher physics. |
| `left_hip_pitch_link`, `right_hip_pitch_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_hip_roll_link`, `right_hip_roll_link` | Thigh capsule | Keep publisher primitive; visual geoms excluded by the runner. |
| `left_hip_yaw_link`, `right_hip_yaw_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_knee_link`, `right_knee_link` | Shin capsule | Keep publisher primitive; not currently in an explicit contact pair. |
| `left_ankle_pitch_link`, `right_ankle_pitch_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_ankle_roll_link`, `right_ankle_roll_link` | Foot box | Keep publisher primitive and foot contact pairs. |
| `waist_yaw_link`, `waist_roll_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `torso_link` | Six visual geoms, including head geometry | Keep visual fallback; upstream torso capsules are commented out. |
| `left_shoulder_pitch_link`, `right_shoulder_pitch_link` | Visual mesh and cylinder | Keep visual fallback; retain publisher physics. |
| `left_shoulder_roll_link`, `right_shoulder_roll_link` | Visual mesh and cylinder | Keep visual fallback; retain publisher physics. |
| `left_shoulder_yaw_link`, `right_shoulder_yaw_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_elbow_link`, `right_elbow_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_wrist_roll_link`, `right_wrist_roll_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_wrist_pitch_link`, `right_wrist_pitch_link` | Visual geoms | Keep visual fallback; retain publisher physics. |
| `left_wrist_yaw_link`, `right_wrist_yaw_link` | Hand capsule | Keep publisher primitive; wrist and hand visual meshes are excluded. |

These are proposed exclusions from **new collision physics**, not from geometric
measurement. The reason is to preserve the pinned publisher's trained-model
contact assumptions while explicitly measuring and reporting the fallback.
The upstream files establish what is disabled; they do not establish the
publisher's reason for every individual omission.

## Remaining limits and decision

- Collision primitives approximate a body's visible surface. Coverage by a
  capsule does not prove that every part of its visual mesh is enclosed. In
  particular, the wrist-yaw body shares the hand capsule, so there is no separate
  guarantee for the wrist mesh. The same limitation applies to thigh, shin and foot
  approximations. Visual fallback does not augment bodies that already have a
  collision primitive.
- Neither scenario tests realistic wall force, friction or recovery from contact.
  A change to that contract needs separate world/runner and robot configuration
  work, with review of controller assumptions.
- The existing warning remains meaningful: measurement uses mixed approximate
  collision primitives and visual meshes, not complete physical collision coverage.
- There is no model change in this audit, hence no changed grades to compare.
  Seed-by-seed stock and Playground trials remain required if an approved follow-up
  changes geometry or selection. No new full hallway/corner runs are claimed here.

**Recommendation for Aaron and Rivet:** accept the documented exclusions for the
current measurement-only worlds, retain all thresholds and publisher geometry,
and keep the regression tests proving fallback coverage. If full-envelope
coverage or physical-contact behavior is wanted, specify it separately rather
than silently making the published controller's model different.

## Reproduce

From a dependency-synced checkout with the pinned files fetched (the test session
fetches and verifies them):

```sh
uv run pytest tests/test_collision_coverage.py
uv run python -m lpl_g1.collision_audit
```

The census is machine-readable and lists each body's measurement mode, geometry
count, direct contact eligibility, explicit pairs and fallback warnings. Tests
check both configuration keyframes, straight/corner selection parity, a
visual-only torso penetrating a measurement wall, and that the audit itself does
not alter collision masks or pairs.
