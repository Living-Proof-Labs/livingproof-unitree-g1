# SPDX-License-Identifier: Apache-2.0
"""Configurable route control and Pose API for the Unitree G1 adapter.

Supports:
- Explicit Pose API (``get_pose``, ``set_pose``, ``MujocoPoseProvider``, ``PoseSample``,
  ``PoseProvider``, ``PoseUnavailableError``) with world-frame provenance and validation.
- Closed-loop ``feedback`` route pursuit reading from an injected ``PoseProvider`` and
  ``open_loop`` deterministic timed velocity schedules that never query pose feedback.
- Configurable ``turn_direction`` (``"auto"``, ``"left"``, ``"right"``), custom
  ``waypoints``, and explicit steering parameters while keeping the existing controller
  and 5-argument ``drive(...)`` contract unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

SPEED = 0.5  # m/s, the speed the verification plans command on a straight route
LOOKAHEAD = 0.7  # m along the route
YAW_GAIN = 2.5  # rad/s per rad of heading error
YAW_RATE_MAX = 1.0  # rad/s
SLOW_ERROR = 0.4  # rad; forward speed scales with cos(error) and stops past this
CROSS_GAIN = 1.0  # m/s sideways per metre off the route, in the robot's frame
CROSS_MAX = 0.3  # m/s

DEFAULT_FRAME_ID = "world"
DEFAULT_SOURCE = "simulator-ideal"
DEFAULT_CLOCK = "simulation"

try:
    from livingproof_sim_mujoco.pose import (
        MujocoPoseProvider,
        PoseProvider,
        PoseSample,
        PoseUnavailableError,
        get_pose,
        normalize_quat,
        quat_to_yaw,
        set_pose,
        yaw_to_quat,
    )
except ImportError:

    class PoseUnavailableError(RuntimeError):
        """Raised when a required pose is missing, stale, wrong-frame, or non-finite."""

    def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
        """World yaw (rad, about +z) from quaternion ``(w, x, y, z)``."""
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
        """Normalized upright ``(w, x, y, z)`` quaternion for heading ``yaw`` (rad)."""
        if not math.isfinite(yaw):
            raise PoseUnavailableError(f"Requested yaw is non-finite ({yaw!r}).")
        half = 0.5 * float(yaw)
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def normalize_quat(
        quat: tuple[float, float, float, float] | list[float],
    ) -> tuple[float, float, float, float]:
        """Normalize ``(w, x, y, z)`` quaternion; raise ``PoseUnavailableError`` if invalid."""
        if len(quat) != 4:
            raise PoseUnavailableError(
                f"Quaternion must have 4 components (w, x, y, z), got length {len(quat)}."
            )
        vals = tuple(float(v) for v in quat)
        if not all(math.isfinite(v) for v in vals):
            raise PoseUnavailableError(f"Quaternion contains non-finite values: {vals!r}.")
        norm = math.sqrt(sum(v * v for v in vals))
        if norm <= 0.0:
            raise PoseUnavailableError(f"Quaternion has zero norm: {vals!r}.")
        return (vals[0] / norm, vals[1] / norm, vals[2] / norm, vals[3] / norm)

    @dataclass(frozen=True)
    class PoseSample:
        """Typed world-frame pose with orientation and provenance metadata."""

        x: float
        y: float
        z: float
        yaw: float
        qw: float = 1.0
        qx: float = 0.0
        qy: float = 0.0
        qz: float = 0.0
        frame_id: str = DEFAULT_FRAME_ID
        timestamp_s: float = 0.0
        source: str = DEFAULT_SOURCE
        clock: str = DEFAULT_CLOCK

        def __post_init__(self) -> None:
            numeric = (
                self.x,
                self.y,
                self.z,
                self.yaw,
                self.qw,
                self.qx,
                self.qy,
                self.qz,
                self.timestamp_s,
            )
            if not all(math.isfinite(float(v)) for v in numeric):
                raise PoseUnavailableError(
                    f"PoseSample contains non-finite numeric fields from source {self.source!r}."
                )
            if not isinstance(self.frame_id, str) or not self.frame_id:
                raise PoseUnavailableError("PoseSample.frame_id must be a non-empty string.")
            if not isinstance(self.source, str) or not self.source:
                raise PoseUnavailableError("PoseSample.source must be a non-empty string.")
            if not isinstance(self.clock, str) or not self.clock:
                raise PoseUnavailableError("PoseSample.clock must be a non-empty string.")

        @property
        def xy(self) -> tuple[float, float]:
            return (self.x, self.y)

        @property
        def position(self) -> tuple[float, float, float]:
            return (self.x, self.y, self.z)

        @property
        def xy_yaw(self) -> tuple[float, float, float]:
            return (self.x, self.y, self.yaw)

        @property
        def orientation(self) -> tuple[float, float, float, float]:
            return (self.qw, self.qx, self.qy, self.qz)

        @classmethod
        def from_quaternion(
            cls,
            x: float,
            y: float,
            z: float,
            qw: float,
            qx: float,
            qy: float,
            qz: float,
            *,
            frame_id: str = DEFAULT_FRAME_ID,
            timestamp_s: float = 0.0,
            source: str = DEFAULT_SOURCE,
            clock: str = DEFAULT_CLOCK,
        ) -> PoseSample:
            w, vx, vy, vz = normalize_quat((qw, qx, qy, qz))
            return cls(
                x=float(x),
                y=float(y),
                z=float(z),
                yaw=quat_to_yaw(w, vx, vy, vz),
                qw=w,
                qx=vx,
                qy=vy,
                qz=vz,
                frame_id=frame_id,
                timestamp_s=float(timestamp_s),
                source=source,
                clock=clock,
            )

        @classmethod
        def from_yaw(
            cls,
            x: float,
            y: float,
            z: float = 0.0,
            yaw: float = 0.0,
            *,
            frame_id: str = DEFAULT_FRAME_ID,
            timestamp_s: float = 0.0,
            source: str = DEFAULT_SOURCE,
            clock: str = DEFAULT_CLOCK,
        ) -> PoseSample:
            w, vx, vy, vz = yaw_to_quat(yaw)
            return cls(
                x=float(x),
                y=float(y),
                z=float(z),
                yaw=float(yaw),
                qw=w,
                qx=vx,
                qy=vy,
                qz=vz,
                frame_id=frame_id,
                timestamp_s=float(timestamp_s),
                source=source,
                clock=clock,
            )

        def to_dict(self) -> dict[str, Any]:
            return {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "yaw": self.yaw,
                "orientation": {"w": self.qw, "x": self.qx, "y": self.qy, "z": self.qz},
                "frame_id": self.frame_id,
                "timestamp_s": self.timestamp_s,
                "source": self.source,
                "clock": self.clock,
            }

    @runtime_checkable
    class PoseProvider(Protocol):
        """Protocol for an engine-backed or synthetic pose source."""

        source: str
        frame_id: str

        def read(self, data: Any = None) -> PoseSample: ...

    def _resolve_body_id(model: mujoco.MjModel, body: str | int) -> int:
        if isinstance(body, (int, np.integer)):
            body_id = int(body)
            if 0 <= body_id < model.nbody:
                return body_id
            raise PoseUnavailableError(f"Body id {body_id} is out of range [0, {model.nbody}).")
        if isinstance(body, str):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id >= 0:
                return int(body_id)
            raise PoseUnavailableError(f"Body {body!r} does not exist in compiled MuJoCo model.")
        raise PoseUnavailableError(f"Body must be a name or id, got {type(body).__name__}.")

    def _freejoint_addresses(model: mujoco.MjModel, body_id: int) -> tuple[int, int]:
        jnt_start = int(model.body_jntadr[body_id])
        jnt_count = int(model.body_jntnum[body_id])
        for offset in range(jnt_count):
            jnt_id = jnt_start + offset
            if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[jnt_id]), int(model.jnt_dofadr[jnt_id])
        body_name = model.body(body_id).name
        raise PoseUnavailableError(
            f"Body {body_name!r} has no free joint; world pose cannot be set via qpos."
        )

    def get_pose(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body: str | int,
        *,
        frame_id: str = DEFAULT_FRAME_ID,
        source: str = DEFAULT_SOURCE,
    ) -> PoseSample:
        """Query the current world-frame pose ``(x, y, z, yaw, quat)`` of ``body``."""
        if data is None:
            raise PoseUnavailableError("MuJoCo data is None; cannot query body pose.")
        body_id = _resolve_body_id(model, body)
        pos = np.asarray(data.xpos[body_id], dtype=float)
        quat = np.asarray(data.xquat[body_id], dtype=float)
        t = float(data.time)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat)) and math.isfinite(t)):
            body_name = model.body(body_id).name
            raise PoseUnavailableError(
                f"Non-finite state while reading pose for body {body_name!r} at t={t!r}."
            )
        return PoseSample.from_quaternion(
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
            frame_id=frame_id,
            timestamp_s=t,
            source=source,
        )

    def set_pose(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body: str | int,
        *,
        x: float,
        y: float,
        z: float | None = None,
        yaw: float = 0.0,
        quat: tuple[float, float, float, float] | list[float] | None = None,
        zero_vel: bool = True,
        forward: bool = True,
        frame_id: str = DEFAULT_FRAME_ID,
        source: str = DEFAULT_SOURCE,
    ) -> PoseSample:
        """Teleport ``body`` via its free joint to ``(x, y, z, yaw/quat)``."""
        if data is None:
            raise PoseUnavailableError("MuJoCo data is None; cannot set body pose.")
        body_id = _resolve_body_id(model, body)
        qadr, dofadr = _freejoint_addresses(model, body_id)

        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise PoseUnavailableError(f"Requested pose position (x={x!r}, y={y!r}) is non-finite.")
        zf = float(data.qpos[qadr + 2]) if z is None else float(z)
        if not math.isfinite(zf):
            raise PoseUnavailableError(f"Pose height z={zf!r} is non-finite.")

        w, vx, vy, vz = normalize_quat(quat) if quat is not None else yaw_to_quat(float(yaw))
        data.qpos[qadr : qadr + 3] = [xf, yf, zf]
        data.qpos[qadr + 3 : qadr + 7] = [w, vx, vy, vz]
        if zero_vel:
            data.qvel[dofadr : dofadr + 6] = 0.0
        if forward:
            mujoco.mj_forward(model, data)
            return get_pose(model, data, body_id, frame_id=frame_id, source=source)
        return PoseSample.from_quaternion(
            xf,
            yf,
            zf,
            w,
            vx,
            vy,
            vz,
            frame_id=frame_id,
            timestamp_s=float(data.time),
            source=source,
        )

    class MujocoPoseProvider:
        """MuJoCo-backed pose provider with explicit provenance and get/set helpers."""

        def __init__(
            self,
            model: mujoco.MjModel,
            base_body: str | int,
            *,
            source: str = DEFAULT_SOURCE,
            frame_id: str = DEFAULT_FRAME_ID,
            max_age_s: float | None = None,
        ) -> None:
            self.model = model
            self.body_id = _resolve_body_id(model, base_body)
            self.base_body = model.body(self.body_id).name
            self.source = source
            self.frame_id = frame_id
            self.max_age_s = max_age_s

        def read(self, data: mujoco.MjData) -> PoseSample:
            sample = self.get_pose(data)
            return validate_pose_sample(
                sample,
                expected_frame=self.frame_id,
                max_age_s=self.max_age_s,
                now_s=float(data.time),
            )

        def get_pose(self, data: mujoco.MjData) -> PoseSample:
            return get_pose(
                self.model,
                data,
                self.body_id,
                frame_id=self.frame_id,
                source=self.source,
            )

        def set_pose(
            self,
            data: mujoco.MjData,
            *,
            x: float,
            y: float,
            z: float | None = None,
            yaw: float = 0.0,
            quat: tuple[float, float, float, float] | list[float] | None = None,
            zero_vel: bool = True,
            forward: bool = True,
        ) -> PoseSample:
            return set_pose(
                self.model,
                data,
                self.body_id,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                quat=quat,
                zero_vel=zero_vel,
                forward=forward,
                frame_id=self.frame_id,
                source=self.source,
            )


def validate_pose_sample(
    sample: Any,
    *,
    expected_frame: str = DEFAULT_FRAME_ID,
    max_age_s: float | None = None,
    now_s: float | None = None,
) -> Any:
    """Validate duck-typed or dataclass pose sample for presence, finiteness, frame, and freshness."""
    if sample is None:
        raise PoseUnavailableError("Pose provider returned None; no pose sample is available.")
    try:
        x = float(sample.x)
        y = float(sample.y)
        z = float(sample.z)
        yaw = float(sample.yaw)
        frame_id = sample.frame_id
        timestamp_s = float(sample.timestamp_s)
        source = getattr(sample, "source", "unknown")
    except (AttributeError, TypeError, ValueError) as exc:
        raise PoseUnavailableError(f"Invalid pose sample from provider: {exc}") from exc
    if not all(math.isfinite(v) for v in (x, y, z, yaw, timestamp_s)):
        raise PoseUnavailableError(f"Non-finite pose fields from source {source!r}.")
    if frame_id != expected_frame:
        raise PoseUnavailableError(
            f"Pose sample frame_id {frame_id!r} does not match expected frame "
            f"{expected_frame!r} (source={source!r})."
        )
    if max_age_s is not None and now_s is not None:
        age = float(now_s) - timestamp_s
        if age > float(max_age_s):
            raise PoseUnavailableError(
                f"Pose sample from {source!r} is stale (age {age:.4f}s > "
                f"max_age_s={float(max_age_s):.4f}s)."
            )
    return sample


@dataclass(frozen=True)
class RouteControlConfig:
    """Explicit configuration for closed-loop feedback or open-loop route execution."""

    mode: str = "feedback"
    turn_direction: str = "auto"
    waypoints: tuple[tuple[float, ...], ...] | None = None
    schedule: tuple[tuple[float, tuple[float, float, float]], ...] | None = None
    speed: float = SPEED
    lookahead: float = LOOKAHEAD
    yaw_gain: float = YAW_GAIN
    yaw_rate_max: float = YAW_RATE_MAX
    slow_error: float = SLOW_ERROR
    cross_gain: float = CROSS_GAIN
    cross_max: float = CROSS_MAX
    expected_frame: str = DEFAULT_FRAME_ID
    max_pose_age_s: float | None = None
    turn_forward_speed: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"feedback", "open_loop"}:
            raise ValueError(
                f"Unsupported route control mode {self.mode!r}; expected 'feedback' or 'open_loop'."
            )
        if self.turn_direction not in {"auto", "left", "right"}:
            raise ValueError(
                f"Unsupported turn_direction {self.turn_direction!r}; "
                "expected 'auto', 'left', or 'right'."
            )
        for name in (
            "speed",
            "lookahead",
            "yaw_gain",
            "yaw_rate_max",
            "slow_error",
            "cross_gain",
            "cross_max",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(
                    f"RouteControlConfig.{name} must be positive and finite, got {val!r}."
                )
        if not math.isfinite(float(self.turn_forward_speed)) or self.turn_forward_speed < 0.0:
            raise ValueError(
                f"RouteControlConfig.turn_forward_speed must be non-negative and finite, "
                f"got {self.turn_forward_speed!r}."
            )
        if self.max_pose_age_s is not None and (
            not math.isfinite(float(self.max_pose_age_s)) or self.max_pose_age_s <= 0.0
        ):
            raise ValueError(
                f"RouteControlConfig.max_pose_age_s must be positive, got {self.max_pose_age_s!r}."
            )
        if self.waypoints is not None:
            if len(self.waypoints) < 2:
                raise ValueError("RouteControlConfig.waypoints requires at least 2 points.")
            for pt in self.waypoints:
                if len(pt) < 2 or not all(math.isfinite(float(v)) for v in pt):
                    raise ValueError(f"Invalid waypoint coordinate: {pt!r}.")
        if self.schedule is not None:
            if not self.schedule:
                raise ValueError("RouteControlConfig.schedule cannot be empty.")
            for duration_s, cmd in self.schedule:
                if not math.isfinite(float(duration_s)) or float(duration_s) <= 0.0:
                    raise ValueError(f"Open-loop segment duration must be > 0, got {duration_s!r}.")
                if len(cmd) != 3 or not all(math.isfinite(float(v)) for v in cmd):
                    raise ValueError(f"Open-loop segment command must have 3 floats, got {cmd!r}.")


def parse_route_control(payload: RouteControlConfig | dict[str, Any] | None) -> RouteControlConfig:
    """Parse None, dict, or ``RouteControlConfig`` into a validated ``RouteControlConfig``."""
    if payload is None:
        return RouteControlConfig()
    if isinstance(payload, RouteControlConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            f"route_control must be a dict, RouteControlConfig, or None; got {type(payload).__name__}."
        )
    allowed = {
        "mode",
        "turn_direction",
        "waypoints",
        "schedule",
        "speed",
        "lookahead",
        "yaw_gain",
        "yaw_rate_max",
        "slow_error",
        "cross_gain",
        "cross_max",
        "expected_frame",
        "max_pose_age_s",
        "turn_forward_speed",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown route_control field(s): {sorted(unknown)}.")
    data = dict(payload)
    if data.get("waypoints") is not None:
        data["waypoints"] = tuple(tuple(float(v) for v in pt) for pt in data["waypoints"])
    if data.get("schedule") is not None:
        normalized: list[tuple[float, tuple[float, float, float]]] = []
        for item in data["schedule"]:
            dur, cmd = item
            normalized.append((float(dur), (float(cmd[0]), float(cmd[1]), float(cmd[2]))))
        data["schedule"] = tuple(normalized)
    return RouteControlConfig(**data)


def polyline_turn_direction(points: list | tuple) -> str:
    """Classify cumulative planar turn sign of ``points`` as ``'left'``, ``'right'``, or ``'straight'``."""
    xy = [np.array(pt[:2], dtype=float) for pt in points]
    if len(xy) < 3:
        return "straight"
    total_angle = 0.0
    prev_dir: np.ndarray | None = None
    for a, b in zip(xy[:-1], xy[1:]):
        seg = b - a
        norm = float(np.linalg.norm(seg))
        if norm <= 1e-12:
            continue
        cur_dir = seg / norm
        if prev_dir is not None:
            cross = float(prev_dir[0] * cur_dir[1] - prev_dir[1] * cur_dir[0])
            dot = float(np.clip(np.dot(prev_dir, cur_dir), -1.0, 1.0))
            total_angle += math.atan2(cross, dot)
        prev_dir = cur_dir
    if total_angle > 1e-3:
        return "left"
    if total_angle < -1e-3:
        return "right"
    return "straight"


def select_route_points(places: dict, config: RouteControlConfig) -> list[list[float]]:
    """Resolve active route points from ``waypoints`` override or ``places`` + ``turn_direction``."""
    if config.waypoints is not None:
        points = [list(pt) for pt in config.waypoints]
    else:
        candidates = places.get("routes")
        if config.turn_direction in {"left", "right"} and candidates:
            candidate_list = (
                list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
            )
            matched = [
                c
                for c in candidate_list
                if polyline_turn_direction(c["points"]) == config.turn_direction
            ]
            if not matched:
                raise ValueError(
                    f"No candidate route in places matches turn_direction={config.turn_direction!r}."
                )
            points = [list(pt) for pt in matched[0]["points"]]
        else:
            points = [list(pt) for pt in places["route"]["points"]]

    if config.turn_direction in {"left", "right"}:
        actual = polyline_turn_direction(points)
        if actual in {"left", "right"} and actual != config.turn_direction:
            raise ValueError(
                f"Configured turn_direction={config.turn_direction!r} conflicts with "
                f"route geometry turning {actual!r}."
            )
    return points


def synthesize_open_loop_schedule(
    points: list | tuple,
    *,
    speed: float = SPEED,
    yaw_rate_max: float = YAW_RATE_MAX,
    turn_forward_speed: float = 0.0,
    extend_distance: float = 1.5,
) -> tuple[tuple[float, tuple[float, float, float]], ...]:
    """Build a deterministic timed ``(duration_s, (vx, vy, wz))`` sequence along ``points``."""
    xy = [np.array(pt[:2], dtype=float) for pt in points]
    if len(xy) < 2:
        raise ValueError("Route polyline requires at least 2 points.")
    segments: list[tuple[float, tuple[float, float, float]]] = []
    dirs: list[tuple[float, np.ndarray]] = []
    for a, b in zip(xy[:-1], xy[1:]):
        vec = b - a
        dist = float(np.linalg.norm(vec))
        if dist > 1e-12:
            dirs.append((dist, vec / dist))
    if not dirs:
        raise ValueError("Route polyline has zero length.")
    for idx, (dist, unit) in enumerate(dirs):
        segments.append((dist / float(speed), (float(speed), 0.0, 0.0)))
        if idx + 1 < len(dirs):
            next_unit = dirs[idx + 1][1]
            cross = float(unit[0] * next_unit[1] - unit[1] * next_unit[0])
            dot = float(np.clip(np.dot(unit, next_unit), -1.0, 1.0))
            dtheta = math.atan2(cross, dot)
            if abs(dtheta) > 1e-3:
                dur = abs(dtheta) / float(yaw_rate_max)
                wz = math.copysign(float(yaw_rate_max), dtheta)
                segments.append((dur, (float(turn_forward_speed), 0.0, wz)))
    if extend_distance > 0.0:
        segments.append((float(extend_distance) / float(speed), (float(speed), 0.0, 0.0)))
    return tuple(segments)


def _route(points: list) -> tuple[np.ndarray, np.ndarray]:
    """Route vertices in the plane, extended one segment past the exit, and arc lengths."""
    xy = np.array([p[:2] for p in points], dtype=float)
    last = xy[-1] + (xy[-1] - xy[-2])
    xy = np.vstack([xy, last])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return xy, arc


def _nearest(xy: np.ndarray, arc: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Arc length and point of the route nearest ``p``."""
    best, best_s, best_q = math.inf, 0.0, xy[0]
    for i in range(len(xy) - 1):
        a, b = xy[i], xy[i + 1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(q - p))
        if d < best:
            best, best_s, best_q = d, arc[i] + t * float(np.linalg.norm(ab)), q
    return best_s, best_q


def _point_at(xy: np.ndarray, arc: np.ndarray, s: float) -> np.ndarray:
    s = min(max(s, 0.0), arc[-1])
    i = int(np.searchsorted(arc, s, side="right") - 1)
    i = min(i, len(xy) - 2)
    t = (s - arc[i]) / max(arc[i + 1] - arc[i], 1e-12)
    return xy[i] + t * (xy[i + 1] - xy[i])


def command(
    xy,
    arc,
    position: np.ndarray,
    yaw: float,
    *,
    config: RouteControlConfig | None = None,
) -> list[float]:
    """``[vx, vy, yaw_rate]`` toward the look-ahead point from ``position`` at ``yaw``."""
    speed = SPEED if config is None else config.speed
    lookahead = LOOKAHEAD if config is None else config.lookahead
    yaw_gain = YAW_GAIN if config is None else config.yaw_gain
    yaw_rate_max = YAW_RATE_MAX if config is None else config.yaw_rate_max
    slow_error = SLOW_ERROR if config is None else config.slow_error
    cross_gain = CROSS_GAIN if config is None else config.cross_gain
    cross_max = CROSS_MAX if config is None else config.cross_max

    along, nearest = _nearest(xy, arc, position)
    target = _point_at(xy, arc, along + lookahead)
    dx, dy = target - position
    heading = math.atan2(dy, dx)
    error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
    yaw_rate = float(np.clip(yaw_gain * error, -yaw_rate_max, yaw_rate_max))
    forward = speed * max(0.0, math.cos(min(abs(error) / slow_error, 1.0) * math.pi / 2))
    # Sideways toward the route, in the robot's frame (+y is the robot's left).
    ox, oy = nearest - position
    left = -math.sin(yaw) * ox + math.cos(yaw) * oy
    sideways = float(np.clip(cross_gain * left, -cross_max, cross_max))
    return [forward, sideways, yaw_rate]


class RouteController:
    """Engine-independent route controller consuming a ``PoseProvider`` or open-loop schedule.

    Holds no simulator engine objects (``MjModel``/``MjData``) and produces body-frame
    velocity commands ``[vx, vy, yaw_rate]`` in ``(m/s, m/s, rad/s)``.
    """

    def __init__(
        self,
        route: dict | list,
        *,
        route_control: RouteControlConfig | dict[str, Any] | None = None,
        pose_provider: PoseProvider | None = None,
    ) -> None:
        self.config = parse_route_control(route_control)
        self.points = (
            [list(pt) for pt in self.config.waypoints]
            if self.config.waypoints is not None
            else (
                [list(pt) for pt in route["points"]]
                if isinstance(route, dict)
                else [list(pt) for pt in route]
            )
        )
        if self.config.turn_direction in {"left", "right"}:
            actual = polyline_turn_direction(self.points)
            if actual in {"left", "right"} and actual != self.config.turn_direction:
                raise ValueError(
                    f"Configured turn_direction={self.config.turn_direction!r} conflicts with "
                    f"route geometry turning {actual!r}."
                )
        self.pose_provider = pose_provider
        self._ticks = 0
        if self.config.mode == "open_loop":
            self.schedule: tuple[tuple[float, tuple[float, float, float]], ...] | None = (
                self.config.schedule
                if self.config.schedule is not None
                else synthesize_open_loop_schedule(
                    self.points,
                    speed=self.config.speed,
                    yaw_rate_max=self.config.yaw_rate_max,
                    turn_forward_speed=self.config.turn_forward_speed,
                )
            )
            self.xy, self.arc = _route(self.points)
        else:
            self.schedule = None
            self.xy, self.arc = _route(self.points)

    def command_at_time(self, elapsed_s: float) -> list[float]:
        """Return ``[vx, vy, yaw_rate]`` at ``elapsed_s`` along the open-loop schedule."""
        assert self.schedule is not None
        acc = 0.0
        cmd = self.schedule[-1][1]
        for duration_s, seg_cmd in self.schedule:
            acc += duration_s
            if elapsed_s < acc - 1e-12:
                cmd = seg_cmd
                break
        return [float(cmd[0]), float(cmd[1]), float(cmd[2])]

    def command_for_pose(self, sample: Any, *, now_s: float | None = None) -> list[float]:
        """Validate ``sample`` and compute closed-loop ``[vx, vy, yaw_rate]``."""
        validated = validate_pose_sample(
            sample,
            expected_frame=self.config.expected_frame,
            max_age_s=self.config.max_pose_age_s,
            now_s=now_s,
        )
        position = np.array([float(validated.x), float(validated.y)], dtype=float)
        return command(self.xy, self.arc, position, float(validated.yaw), config=self.config)

    def next_command(
        self,
        *,
        data: Any = None,
        now_s: float | None = None,
        dt_s: float | None = None,
    ) -> list[float]:
        """Advance one control tick: open-loop schedule (no pose read) or feedback via ``PoseProvider``."""
        if self.config.mode == "open_loop":
            t = (
                self._ticks * float(dt_s)
                if dt_s is not None
                else (float(now_s) if now_s is not None else 0.0)
            )
            self._ticks += 1
            return self.command_at_time(t)
        if self.pose_provider is None:
            raise PoseUnavailableError("Feedback route control requires a PoseProvider.")
        sample = self.pose_provider.read(data)
        return self.command_for_pose(sample, now_s=now_s)


def follow_route(
    model,
    policy,
    base_body: str,
    route: dict | list,
    *,
    route_control: RouteControlConfig | dict[str, Any] | None = None,
    pose_provider: PoseProvider | None = None,
):
    """``step(data)``: MuJoCo bridge connecting ``RouteController`` to ``policy.step``."""
    cfg = parse_route_control(route_control)
    provider: Any = pose_provider
    if provider is None and cfg.mode == "feedback":
        provider = MujocoPoseProvider(model, base_body, frame_id=cfg.expected_frame)
    controller = RouteController(route, route_control=cfg, pose_provider=provider)
    dt = float(model.opt.timestep)

    if cfg.mode == "open_loop":

        def open_loop_step(data) -> None:
            policy.step(data, controller.next_command(dt_s=dt))

        return open_loop_step

    def feedback_step(data) -> None:
        policy.step(data, controller.next_command(data=data, now_s=float(data.time)))

    return feedback_step


def drive(
    robot,
    name: str,
    model,
    ctrl,
    policy,
    places: dict,
    *,
    route_control: RouteControlConfig | dict[str, Any] | None = None,
    pose_provider: PoseProvider | None = None,
):
    """The adapter's ``drive``: follow ``places['route']`` or configured waypoints/mode.

    ``places`` is the simulator's place dictionary. By default this reads closed-loop
    world pose via ``MujocoPoseProvider`` and follows ``places['route']``; callers may
    pass ``route_control`` (``feedback`` / ``open_loop``, ``turn_direction``,
    ``waypoints``, ``schedule``, gains) and/or an alternate ``pose_provider``.
    """
    del ctrl
    cfg = parse_route_control(route_control)
    points = select_route_points(places, cfg)
    return follow_route(
        model,
        policy,
        robot.base_body(name),
        {"points": points},
        route_control=cfg,
        pose_provider=pose_provider,
    )
