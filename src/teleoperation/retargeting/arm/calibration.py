# 类 ArmCalibration[to_dict/load/from_dict],CameraCalibrationConfig,NeutralPoseCollector[ready/add/calibration]（封装数据/流程）；函数 _finite/_positive/_rotation/_quaternion（内部辅助计算）；tron2a_reference（业务处理）；常量 ARM_DOF/SIDES（配置/索引）；链路：上肢观测→校准/目标位姿/IK→关节目标
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ARM_DOF = 7
from teleoperation.data.calibration import read_calibration
@dataclass(frozen=True)
class ArmCalibration:
    rotation: np.ndarray
    translation: np.ndarray
    scale: np.ndarray
    human_neutral_wrist: dict[str, np.ndarray]
    robot_neutral_pose: dict[str, np.ndarray]
    tool_offset_quat_xyzw: dict[str, np.ndarray]
    safe_q: dict[str, np.ndarray]
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_acceleration: float
    tracking_timeout: float
    hand_mount_pose: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        _rotation(self.rotation)
        for name in ("translation", "scale", "workspace_min", "workspace_max"):
            _finite(getattr(self, name), (3,), name)
        if np.any(self.scale <= 0):
            raise ValueError("scale must be positive")
        if np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace_min must be below workspace_max")
        for name in ("max_acceleration", "tracking_timeout"):
            _positive(getattr(self, name), name)
        for side in SIDES:
            _finite(self.human_neutral_wrist[side], (3,), f"{side} human neutral")
            _finite(self.safe_q[side], (ARM_DOF,), f"{side} safe_q")
            _quaternion(self.tool_offset_quat_xyzw[side], f"{side} tool offset")
            for name in ("robot_neutral_pose", "hand_mount_pose"):
                pose = _finite(getattr(self, name)[side], (7,), f"{side} {name}")
                _quaternion(pose[3:], f"{side} {name}")

    def to_dict(self):
        return {
            name: ({side: value.tolist() for side, value in item.items()}
                   if isinstance(item, dict) and name != "metadata"
                   else item.tolist() if isinstance(item, np.ndarray) else item)
            for name, item in vars(self).items()
        }

    @classmethod
    def load(cls, path: str | Path) -> "ArmCalibration":
        return cls.from_dict(read_calibration(path))

    @classmethod
    def from_dict(cls, data) -> "ArmCalibration":
        def vector(name, size):
            value = np.asarray(data[name], dtype=np.float64)
            if value.shape != (size,):
                raise ValueError(f"calibration.{name} must have shape ({size},)")
            return value
        rotation = np.asarray(data["rotation"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("calibration.rotation must have shape (3, 3)")
        scale = np.asarray(data["scale"], dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(3, float(scale))
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("calibration.scale must be positive scalar or shape (3,)")
        sides = ("left", "right")
        return cls(
            rotation=rotation, translation=vector("translation", 3), scale=scale,
            human_neutral_wrist={side: np.asarray(data["human_neutral_wrist"][side], dtype=np.float64) for side in sides},
            robot_neutral_pose={side: np.asarray(data["robot_neutral_pose"][side], dtype=np.float64) for side in sides},
            tool_offset_quat_xyzw={side: np.asarray(data["tool_offset_quat_xyzw"][side], dtype=np.float64) for side in sides},
            safe_q={side: np.asarray(data["safe_q"][side], dtype=np.float64) for side in sides},
            workspace_min=vector("workspace_min", 3), workspace_max=vector("workspace_max", 3),
            max_acceleration=float(data.get("max_acceleration", 20.0)),
            tracking_timeout=float(data.get("tracking_timeout", 0.5)),
            hand_mount_pose={side: np.asarray(data.get("hand_mount_pose", {}).get(side, [0, 0, 0, 0, 0, 0, 1]), dtype=np.float64) for side in sides},
            metadata=dict(data.get("metadata", {})),
        )


SIDES = ("left", "right")


def _finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _rotation(value):
    matrix = _finite(value, (3, 3), "rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("rotation must be a proper orthonormal matrix")
    return matrix


def _quaternion(value, name):
    quat = _finite(value, (4,), name)
    if not np.isclose(np.linalg.norm(quat), 1, atol=1e-6):
        raise ValueError(f"{name} quaternion must have unit norm")


@dataclass(frozen=True)
class CameraCalibrationConfig:
    """Explicit mapping and collection criteria, not estimated extrinsics."""
    rotation: np.ndarray
    scale: np.ndarray
    safe_q: dict[str, np.ndarray]
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sample_count: int = 60
    min_visibility: float = 0.7
    stability_tolerance: float = 0.015
    min_elbow_bend_degrees: float = 10.0
    max_elbow_bend_degrees: float = 100.0
    max_sample_gap: float = 0.5
    max_acceleration: float = 20.0
    tracking_timeout: float = 0.5

    def __post_init__(self):
        _rotation(self.rotation)
        for name in ("scale", "workspace_min", "workspace_max", "translation"):
            _finite(getattr(self, name), (3,), name)
        if np.any(self.scale <= 0) or np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("positive scale and ordered workspace bounds required")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 3:
            raise ValueError("sample_count must be an integer >= 3")
        if not np.isfinite(self.min_visibility) or not 0 <= self.min_visibility <= 1:
            raise ValueError("min_visibility must be in [0,1]")
        for name in ("stability_tolerance", "max_sample_gap", "max_acceleration", "tracking_timeout"):
            _positive(getattr(self, name), name)
        if not (np.isfinite(self.min_elbow_bend_degrees) and np.isfinite(self.max_elbow_bend_degrees)
                and 0 < self.min_elbow_bend_degrees < self.max_elbow_bend_degrees < 180):
            raise ValueError("elbow bend bounds must satisfy 0 < min < max < 180 degrees")
        for side in SIDES:
            _finite(self.safe_q[side], (ARM_DOF,), f"{side} safe_q")


def tron2a_reference(config, urdf_path):
    """URDF-only FK; meshes and robot/simulation execution are not needed."""
    from .ik import load_arm_specification

    poses, limits = {}, {}
    for side in SIDES:
        spec = load_arm_specification(urdf_path, side)
        for name in ("lower", "upper", "velocity"):
            _finite(getattr(spec, name), (ARM_DOF,), f"{side} {name}")
        if np.any(spec.lower >= spec.upper) or np.any(spec.velocity <= 0):
            raise ValueError(f"Invalid {side} URDF joint limits")
        q = config.safe_q[side]
        if np.any(q < spec.lower) or np.any(q > spec.upper):
            raise ValueError(f"{side} safe_q is outside URDF joint limits")
        matrix = _finite(spec.robot.fkine(q).A, (4, 4), f"{side} FK")
        _rotation(matrix[:3, :3])
        position = matrix[:3, 3]
        if np.any(position + config.translation < config.workspace_min) or np.any(position + config.translation > config.workspace_max):
            raise ValueError(f"{side} robot neutral target is outside workspace")
        poses[side] = np.r_[position, Rotation.from_matrix(matrix[:3, :3]).as_quat()]
        limits[side] = {"lower": spec.lower.tolist(), "upper": spec.upper.tolist(),
                        "velocity": spec.velocity.tolist(), "joint_names": list(spec.joint_names),
                        "ee_link": spec.ee_link}
    return poses, limits


class NeutralPoseCollector:
    """Collect a consecutive stable bilateral window in shoulder-relative metres.

    A missing/low-quality/degenerate sample clears the window. Excess motion
    starts a new window at the current sample. Stability checks both elbow and
    wrist relative to each side's own shoulder, preserving posture as well as
    wrist location. No interpolation or filtering is used.
    """
    def __init__(self, config):
        self.config = config
        self.samples = []
        self.trace = []
        self._last_time = None
        self._last_frame = None
        self._last_detection = None
        self.status = "waiting for both arms"

    @property
    def ready(self):
        return len(self.samples) >= self.config.sample_count

    def _clear(self, reason):
        self.samples.clear()
        self.trace.clear()
        self.status = reason

    def add(self, observation, control_seconds):
        from teleoperation.contracts.observations import RawTeleopObservation
        if not isinstance(observation, RawTeleopObservation):
            raise TypeError("Expected RawTeleopObservation")
        if not np.isfinite(control_seconds) or control_seconds < 0:
            raise ValueError("control time must be finite relative seconds")
        if self._last_time is not None and control_seconds <= self._last_time:
            raise ValueError("control time must increase monotonically")
        body, metadata = observation.upper_body, observation.hands.metadata
        frame = metadata.get("frame_index")
        detection = body.timestamp
        if (not isinstance(frame, (int, np.integer)) or frame < 0
                or detection is None or not np.isfinite(detection)
                or detection != observation.hands.timestamp):
            raise ValueError("matching raw timestamps and frame index required")
        if metadata.get("timestamp_unit") != "unix_ms" or metadata.get("arm_world_space") != "mediapipe_pose_world_meters":
            raise ValueError("Expected MediaPipe Pose world metres and Unix ms detection time")
        if self._last_time is not None and (
            control_seconds - self._last_time > self.config.max_sample_gap
            or frame != self._last_frame + 1 or detection <= self._last_detection
        ):
            self._clear("capture gap; restarting")
        self._last_time, self._last_frame, self._last_detection = control_seconds, frame, detection
        visibility = np.asarray(metadata.get("pose_visibility"), dtype=float)
        if visibility.shape != (6,):
            self._clear("missing pose visibility")
            return False
        relative = {}
        for side, indices in (("left", [0, 2, 4]), ("right", [1, 3, 5])):
            points = None if body.world_arms is None else body.world_arms[side]
            confidence = visibility[indices]
            if (not body.valid[side] or points is None or np.shape(points) != (3, 3)
                    or not np.isfinite(points).all() or not np.isfinite(confidence).all()
                    or np.any(confidence < self.config.min_visibility) or np.any(confidence > 1)):
                self._clear(f"{side} arm invalid/low visibility")
                return False
            points = np.asarray(points, dtype=float)
            upper, forearm = points[1] - points[0], points[2] - points[1]
            lengths = np.linalg.norm(upper), np.linalg.norm(forearm)
            if min(lengths) < 1e-6:
                self._clear(f"{side} degenerate arm")
                return False
            bend = np.degrees(np.arccos(np.clip(np.dot(upper, forearm) / np.prod(lengths), -1, 1)))
            if not self.config.min_elbow_bend_degrees <= bend <= self.config.max_elbow_bend_degrees:
                self._clear(f"{side} bend elbow slightly")
                return False
            relative[side] = points - points[0]
        candidate = self.samples + [relative]
        for side in SIDES:
            stack = np.stack([sample[side] for sample in candidate])
            if np.max(np.linalg.norm(stack - stack.mean(axis=0), axis=2)) > self.config.stability_tolerance:
                self._clear("motion exceeds stability tolerance")
                break
        self.samples.append(relative)
        self.trace.append({"frame_index": int(frame), "detection_unix_ms": int(detection),
                           "control_relative_seconds": float(control_seconds)})
        if len(self.samples) > self.config.sample_count:
            self.samples.pop(0)
            self.trace.pop(0)
        self.status = f"stable samples {len(self.samples)}/{self.config.sample_count}"
        return self.ready

    def calibration(self, urdf_path):
        if not self.ready:
            raise ValueError("Insufficient consecutive stable neutral samples")
        poses, limits = tron2a_reference(self.config, urdf_path)
        neutral, offsets = {}, {}
        from .retargeter import _forearm_rotation
        for side in SIDES:
            stack = np.stack([sample[side] for sample in self.samples])
            mean = stack.mean(axis=0)
            if np.max(np.linalg.norm(stack - mean, axis=2)) > self.config.stability_tolerance:
                raise ValueError("Unstable neutral samples")
            orientation = _forearm_rotation(mean)
            if orientation is None:
                raise ValueError(f"{side} neutral forearm orientation is degenerate")
            neutral[side] = mean[2]
            # At neutral, configured mapping + tool offset matches reference FK.
            robot_rotation = Rotation.from_quat(poses[side][3:]).as_matrix()
            offsets[side] = Rotation.from_matrix((self.config.rotation @ orientation).T @ robot_rotation).as_quat()
        return ArmCalibration(
            self.config.rotation, self.config.translation, self.config.scale,
            neutral, poses, offsets, self.config.safe_q,
            self.config.workspace_min, self.config.workspace_max,
            self.config.max_acceleration, self.config.tracking_timeout,
            {side: np.array([0, 0, 0, 0, 0, 0, 1.], dtype=float) for side in SIDES},
            metadata={
                "robot": "tron2a", "urdf": str(Path(urdf_path).resolve()),
                "coordinate_source": "mediapipe_pose_world_meters", "units": "m",
                "human_reference": "each_side_own_shoulder",
                "point_order": ["shoulder", "elbow", "wrist"],
                "robot_reference_frame": "base_Link",
                "position_formula": "robot_neutral + translation + rotation @ (scale * ((wrist - shoulder) - human_neutral_wrist))",
                "extrinsics_calibrated": False, "rotation_source": "explicit_configuration",
                "scale_source": "explicit_configuration",
                "tool_offset_source": "neutral_forearm_alignment_to_reference_FK",
                "hand_mount_source": "identity_placeholder_not_calibrated",
                "notice": "Neutral capture estimates reference positions only; camera extrinsics and scale are not automatically calibrated. Shoulder-relative inputs are required before future retargeting.",
                "sample_count": len(self.samples), "min_visibility": self.config.min_visibility,
                "stability_tolerance_m": self.config.stability_tolerance,
                "elbow_bend_degrees": [self.config.min_elbow_bend_degrees, self.config.max_elbow_bend_degrees],
                "max_sample_gap_seconds": self.config.max_sample_gap,
                "joint_limits": limits, "capture_trace": self.trace.copy(),
            },
        )
