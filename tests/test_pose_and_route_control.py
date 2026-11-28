# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the G1 Pose API and configurable route control (scenario-neutral)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from lpl_g1 import route
from lpl_g1.adapter import ROBOT
from lpl_g1.route import (
    PoseProvider,
    PoseSample,
    PoseUnavailableError,
    RouteControlConfig,
    get_pose,
    parse_route_control,
    polyline_turn_direction,
    set_pose,
    synthesize_open_loop_schedule,
    validate_pose_sample,
)


class _RecordingPolicy:
    def __init__(self):
        self.commands: list[list[float]] = []

    def step(self, data, command):
        del data
        self.commands.append([float(v) for v in command])


def test_adapter_pose_api_set_and_get_round_trip():
    scene = ROBOT.scene("stock")
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    provider = ROBOT.pose_provider(model, "stock", source="unit-test")
    assert isinstance(provider, PoseProvider)

    placed = ROBOT.set_pose(model, data, x=-1.25, y=0.6, z=0.82, yaw=math.pi / 4)
    assert placed.x == pytest.approx(-1.25)
    assert placed.y == pytest.approx(0.6)
    assert placed.z == pytest.approx(0.82)
    assert placed.yaw == pytest.approx(math.pi / 4)
    assert placed.frame_id == "world"
    assert data.qvel[:6] == pytest.approx([0.0] * 6)

    queried = ROBOT.get_pose(model, data)
    assert queried.xy_yaw == pytest.approx((-1.25, 0.6, math.pi / 4))
    assert provider.read(data).xy_yaw == pytest.approx((-1.25, 0.6, math.pi / 4))

    # Omitting z preserves current z.
    moved = ROBOT.set_pose(model, data, x=0.4, y=-0.3, yaw=-math.pi / 2)
    assert moved.z == pytest.approx(0.82)
    assert moved.yaw == pytest.approx(-math.pi / 2)


def test_pose_validation_rejects_missing_wrong_frame_stale_and_non_finite():
    scene = ROBOT.scene("stock")
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with pytest.raises(PoseUnavailableError, match="does not exist"):
        get_pose(model, data, "missing_body")
    with pytest.raises(PoseUnavailableError, match="None"):
        get_pose(model, None, ROBOT.base_body("stock"))
    with pytest.raises(PoseUnavailableError, match="non-finite"):
        set_pose(model, data, ROBOT.base_body("stock"), x=float("nan"), y=0.0)
    with pytest.raises(PoseUnavailableError, match="returned None"):
        validate_pose_sample(None)
    with pytest.raises(PoseUnavailableError, match="frame_id"):
        validate_pose_sample(
            PoseSample.from_yaw(0.0, 0.0, 0.8, 0.0, frame_id="odom"), expected_frame="world"
        )
    with pytest.raises(PoseUnavailableError, match="stale"):
        validate_pose_sample(
            PoseSample.from_yaw(0.0, 0.0, 0.8, 0.0, timestamp_s=1.0),
            max_age_s=0.1,
            now_s=1.5,
        )


def test_route_control_config_parsing_turn_selection_and_validation():
    left_route = {"points": [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]}
    right_route = {"points": [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]}

    assert polyline_turn_direction(left_route["points"]) == "left"
    assert polyline_turn_direction(right_route["points"]) == "right"
    assert polyline_turn_direction([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) == "straight"

    places = {"route": left_route, "routes": {"forward": left_route, "reverse": right_route}}
    policy = _RecordingPolicy()
    scene = ROBOT.scene("stock")
    model = scene.model

    # turn_direction="right" picks right_route from places["routes"].
    step_right = ROBOT.drive(
        "stock",
        model,
        None,
        policy,
        places,
        route_control={"mode": "feedback", "turn_direction": "right"},
        pose_provider=type(
            "StartSouth",
            (),
            {
                "frame_id": "world",
                "source": "fake",
                "read": lambda self, _d: PoseSample.from_yaw(0.0, 0.1, 0.8, -math.pi / 2),
            },
        )(),
    )
    step_right(mujoco.MjData(model))
    assert policy.commands[-1][2] < 0.0

    # Conflicting turn_direction against explicit single route fails loudly.
    with pytest.raises(ValueError, match="conflicts"):
        ROBOT.drive(
            "stock",
            model,
            None,
            policy,
            {"route": left_route},
            route_control={"turn_direction": "right"},
        )

    # Custom waypoints override places['route'].
    cfg = parse_route_control(
        {"mode": "feedback", "waypoints": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], "speed": 0.4}
    )
    assert isinstance(cfg, RouteControlConfig)
    assert cfg.speed == pytest.approx(0.4)

    with pytest.raises(ValueError, match="Unsupported route control mode"):
        parse_route_control({"mode": "invalid"})
    with pytest.raises(ValueError, match="Unsupported turn_direction"):
        parse_route_control({"turn_direction": "backwards"})
    with pytest.raises(ValueError, match="Unknown route_control"):
        parse_route_control({"unknown_field": 1})


def test_interchangeable_fake_pose_providers_produce_identical_commands():
    scene = ROBOT.scene("stock")
    model = scene.model
    data = mujoco.MjData(model)
    places = {"route": {"points": [[-2.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0]]}}
    poses = [
        (0.0, -2.5, 0.0, 0.8, 0.0),
        (1.0, -0.1, 0.05, 0.8, 0.2),
        (2.0, 0.0, 1.8, 0.8, math.pi / 2),
    ]

    class ProviderA:
        frame_id = "world"
        source = "fake-a"

        def __init__(self):
            self.idx = 0

        def read(self, _data=None):
            t, x, y, z, yaw = poses[self.idx]
            self.idx += 1
            return PoseSample.from_yaw(x, y, z, yaw, timestamp_s=t, source=self.source)

    class ProviderB:
        frame_id = "world"
        source = "fake-b"

        def __init__(self):
            self.idx = 0

        def read(self, _data=None):
            t, x, y, z, yaw = poses[self.idx]
            self.idx += 1
            return type(
                "DuckSample",
                (),
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": yaw,
                    "frame_id": "world",
                    "timestamp_s": t,
                    "source": self.source,
                },
            )()

    pol_a, pol_b = _RecordingPolicy(), _RecordingPolicy()
    step_a = ROBOT.drive("stock", model, None, pol_a, places, pose_provider=ProviderA())
    step_b = ROBOT.drive("stock", model, None, pol_b, places, pose_provider=ProviderB())

    for t, *_ in poses:
        data.time = t
        step_a(data)
        step_b(data)

    assert np.allclose(pol_a.commands, pol_b.commands)
    assert pol_a.commands[0][0] == pytest.approx(route.SPEED)
    assert pol_a.commands[1][2] > 0.0


def test_open_loop_mode_never_reads_pose_provider_and_feedback_fails_cleanly():
    scene = ROBOT.scene("stock")
    model = scene.model
    data = mujoco.MjData(model)
    places = {"route": {"points": [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]}}

    class FailingProvider:
        frame_id = "world"
        source = "broken"
        calls = 0

        def read(self, _data=None):
            self.calls += 1
            raise PoseUnavailableError("pose unavailable")

    broken = FailingProvider()
    policy = _RecordingPolicy()

    dt = float(model.opt.timestep)
    schedule = [(2 * dt, [0.4, 0.0, 0.0]), (2 * dt, [0.0, 0.0, 0.6])]
    step_ol = ROBOT.drive(
        "stock",
        model,
        None,
        policy,
        places,
        route_control={"mode": "open_loop", "schedule": schedule},
        pose_provider=broken,
    )
    for _ in range(4):
        step_ol(data)

    assert broken.calls == 0
    assert policy.commands == [
        [0.4, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 0.6],
    ]

    # Synthesized open-loop schedule also skips pose feedback cleanly.
    synthesized = synthesize_open_loop_schedule(
        places["route"]["points"], speed=0.5, yaw_rate_max=1.0
    )
    assert len(synthesized) == 4
    assert synthesized[1][1][2] == pytest.approx(1.0)

    # Feedback mode with failing provider raises PoseUnavailableError.
    step_fb = ROBOT.drive(
        "stock",
        model,
        None,
        policy,
        places,
        route_control={"mode": "feedback"},
        pose_provider=broken,
    )
    with pytest.raises(PoseUnavailableError, match="pose unavailable"):
        step_fb(data)
    assert broken.calls == 1


def test_engine_free_route_controller_operates_without_mujoco_objects():
    pts = [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    sample = PoseSample.from_yaw(
        -2.0, 0.0, 0.8, 0.0, timestamp_s=0.5, source="external-tracking", clock="monotonic"
    )
    assert sample.clock == "monotonic"
    assert sample.to_dict()["clock"] == "monotonic"

    provider = type(
        "NoEngineProvider",
        (),
        {"frame_id": "world", "source": "external-tracking", "read": lambda self, _d=None: sample},
    )()
    ctrl = route.RouteController(
        pts,
        route_control={"mode": "feedback", "max_pose_age_s": 0.2},
        pose_provider=provider,
    )
    cmd = ctrl.next_command(now_s=0.6)
    assert cmd == pytest.approx([route.SPEED, 0.0, 0.0])
    with pytest.raises(PoseUnavailableError, match="stale"):
        ctrl.next_command(now_s=1.0)
