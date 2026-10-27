"""Native dex Vector pipeline, separate from the L21 model-result interface.

Camera and VR profiles are explicit. The VR extraction follows the local
TransHandR reference; its existing model pipeline is never modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np

from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.contracts.hand import NativeHandResult
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.retargeting.hand.coordinates import build_palm_basis, OPERATOR_TO_CANONICAL
from teleoperation.retargeting.hand.interface import NativeHandRetargeter

SIDES = ("left", "right")
PROFILES = {"camera": "dex_camera_world_v1", "visionpro": "dex_visionpro_palm_v2"}
VECTOR_PARAMETERS = {"upstream_version": "0.5.0", "type": "vector", "scaling": 1.,
    "norm_delta": 0., "huber_delta": .02, "ftol_abs": 1e-6, "xtol_abs": 1e-6,
    "maxeval": 100, "internal_lowpass": False, "native_limit_epsilon": 0.,
    "recovery": "G1 deterministic multi-start with verified planar feasibility bound"}


class DexInputAdapter:
    def __init__(self, source):
        if source not in PROFILES:
            raise ValueError("Dex input must be camera or visionpro")
        self.source = source
        if source == "camera":
            from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
            self.processor = MediaPipePalmLocalProcessor("world")

    def adapt(self, observation):
        if not isinstance(observation, RawHandFrame):
            raise TypeError("Dex input requires RawHandFrame")
        if observation.timestamp is None or not np.isfinite(observation.timestamp):
            raise ValueError("Dex timestamps must explicitly be finite seconds")
        if "coordinate_alignment" in observation.metadata:
            raise ValueError("Dex adapter requires raw observations, not aligned points")
        if self.source == "camera":
            if observation.source not in ("mediapipe", "mediapipe_approx"):
                raise ValueError("Camera dex adapter requires MediaPipe observations")
            result = self.processor.process_frame(observation)
            hands, reasons = result.hands, dict(result.invalid_reasons)
        else:
            from teleoperation.applications.visionpro_processing import transhandr_vr_points
            if observation.source != "visionpro":
                raise ValueError("VR dex adapter requires source=visionpro")
            hands, reasons = {}, {}
            for side in SIDES:
                value = observation.hands[side]
                hands[side], reasons[side] = None, "missing"
                if value is None:
                    continue
                if observation.geometry_encoding.get(side) != "transforms25":
                    raise ValueError("VR dex requires explicit transforms25 encoding")
                try:
                    relative = transhandr_vr_points(value, side)
                    # Upstream's left-to-right L21 remap has determinant -1.
                    # The native left URDF needs anatomical handedness restored
                    # BEFORE fitting a proper palm rotation. This belongs only
                    # to the new dex profile, never the upstream model route.
                    if side == "left":
                        relative = relative @ np.diag([-1., 1., 1.])
                    basis = build_palm_basis(relative, (0, 6, 11))
                    hands[side] = (relative @ basis @ OPERATOR_TO_CANONICAL[side]).astype(np.float32)
                    reasons[side] = None
                except (ValueError, np.linalg.LinAlgError) as error:
                    reasons[side] = str(error)
        metadata = {"dex_input_profile": PROFILES[self.source], "units": "m", "origin": "wrist",
                    "coordinate_frame": "dex_canonical_hand", "invalid_reasons": reasons,
                    "timestamp_unit": "seconds"}
        return CanonicalHandFrame(hands, float(observation.timestamp), self.source, metadata)


@dataclass(frozen=True)
class DexCalibration:
    profile: str
    palm_lengths: dict
    segment: tuple
    source_sha256: str = ""

    def __post_init__(self):
        if self.profile not in PROFILES.values() or set(self.palm_lengths) != set(SIDES):
            raise ValueError("Calibration profile or hand sides are invalid")
        if not all(np.isfinite(v) and v > 1e-4 for v in self.palm_lengths.values()):
            raise ValueError("Calibration palm lengths must be positive metres")
        if len(self.segment) != 2 or not np.isfinite(self.segment).all() or self.segment[1] <= self.segment[0]:
            raise ValueError("Calibration must declare its excluded time segment")

    @staticmethod
    def is_open(points):
        if points is None:
            return False
        for mcp in (6, 11, 16, 21):
            bones = np.diff(points[mcp:mcp + 4], axis=0)
            lengths = np.linalg.norm(bones, axis=1)
            if np.any(lengths < 1e-6):
                return False
            axes = bones / lengths[:, None]
            if np.any(np.sum(axes[:-1] * axes[1:], axis=1) < .85):
                return False
        return True

    @classmethod
    def fit(cls, frames, start, duration=2., source_sha256=""):
        if duration < 2 or not np.isfinite([start, duration]).all():
            raise ValueError("Open-hand calibration requires at least two seconds")
        frames = list(frames)
        selected = [f for f in frames if start <= f.timestamp <= start + duration]
        if not selected or selected[-1].timestamp - selected[0].timestamp < 1.9:
            raise ValueError("Calibration recording does not cover two seconds")
        profile = selected[0].metadata.get("dex_input_profile")
        if any(f.metadata.get("dex_input_profile") != profile for f in selected):
            raise ValueError("Mixed calibration source profiles")
        lengths = {}
        for side in SIDES:
            valid = [f for f in selected if cls.is_open(f.hands[side])]
            if len(valid) < 10 or valid[-1].timestamp - valid[0].timestamp < 1.9:
                raise ValueError(f"Need two seconds and at least ten open-hand samples: {side}")
            lengths[side] = float(np.median([np.linalg.norm(f.hands[side][11]) for f in valid]))
        return cls(profile, lengths, (float(start), float(start + duration)), source_sha256)

    def scale_for(self, model):
        return model.palm_reference()[1] / self.palm_lengths[model.side]

    def save(self, path):
        payload = {"schema": "dex_native_calibration_v1", "units": "m", "profile": self.profile,
                   "palm_lengths": self.palm_lengths, "segment": list(self.segment),
                   "source_sha256": self.source_sha256}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "dex_native_calibration_v1" or payload.get("units") != "m":
            raise ValueError("Unknown dex calibration schema or units")
        return cls(payload["profile"], payload["palm_lengths"], tuple(payload["segment"]), payload["source_sha256"])


class NativeVectorSolver:
    """Unmodified upstream VectorOptimizer/SeqRetargeting with native constraints."""

    def __init__(self, model):
        try:
            from dex_retargeting.optimizer import VectorOptimizer
            from dex_retargeting.seq_retarget import SeqRetargeting
            from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
        except ImportError as error:
            raise ImportError("Install the dex-native environment from docs/dex-native.md") from error
        self.model = model
        robot = model.kinematics
        for name, bounds in zip(model.independent_names, model.bounds):
            index = robot.dof_joint_names.index(name)
            robot.model.lowerPositionLimit[index], robot.model.upperPositionLimit[index] = bounds
        optimizer = VectorOptimizer(robot, list(model.independent_names),
            [model.root_link] * len(model.fingers), list(model.tip_frames),
            np.array([np.zeros(len(model.fingers), dtype=int), model.human_tip_indices]),
            scaling=VECTOR_PARAMETERS["scaling"], norm_delta=VECTOR_PARAMETERS["norm_delta"],
            huber_delta=VECTOR_PARAMETERS["huber_delta"])
        mimic_names = [n for n in model.joint_names if n not in model.independent_names]
        if mimic_names:
            optimizer.set_kinematic_adaptor(MimicJointKinematicAdaptor(
                robot, list(model.independent_names),
                source_joint_names=[model.relations[n][0] for n in mimic_names],
                mimic_joint_names=mimic_names,
                multipliers=[model.relations[n][1] for n in mimic_names],
                offsets=[model.relations[n][2] for n in mimic_names]))
        self.sequence = SeqRetargeting(optimizer, has_joint_limits=True, lp_filter=None)
        optimizer.set_joint_limit(model.bounds, epsilon=0.)
        optimizer.opt.set_ftol_abs(VECTOR_PARAMETERS["ftol_abs"])
        optimizer.opt.set_xtol_abs(VECTOR_PARAMETERS["xtol_abs"])
        optimizer.opt.set_maxeval(VECTOR_PARAMETERS["maxeval"])
        self.motion_plane_z = None
        if model.robot_type == "g1":
            import xml.etree.ElementTree as ET
            root = ET.parse(model.urdf_path).getroot()
            joints = {j.get("name"): j for j in root.findall("joint")}
            plane_verified = True
            for finger in ("index", "middle"):
                for index in (0, 1):
                    joint = joints[f"{model.side}_hand_{finger}_{index}_joint"]
                    plane_verified &= np.allclose(np.fromstring(joint.find("axis").get("xyz"), sep=" "), [0,0,1])
                    plane_verified &= np.allclose(np.fromstring(joint.find("origin").get("rpy", "0 0 0"), sep=" "), 0)
            if plane_verified:
                self.motion_plane_z = model.positions(np.clip(np.zeros(len(model.independent_names)), *model.bounds.T))[1:3, 2]
        self.reset()

    def reset(self):
        self.sequence.reset()
        self.sequence.last_qpos = np.clip(np.zeros(len(self.model.independent_names)), *self.model.bounds.T)
        self.previous_targets = None

    def solve(self, targets, timestamp):
        started = time.perf_counter()
        previous = self.sequence.last_qpos.copy()
        model = self.model
        values, error = None, None
        try:
            targets = np.asarray(targets, dtype=float)
            if targets.shape != (len(model.fingers), 3) or not np.isfinite(targets).all():
                raise ValueError("Native Vector targets must be finite fingertip vectors")
            def candidate(seed=None):
                if seed is not None:
                    self.sequence.last_qpos = seed.copy()
                qpos = self.sequence.retarget(targets)
                status = self.sequence.optimizer.opt.last_optimize_result()
                if status < 0:
                    raise RuntimeError(f"NLopt failed with status {status}")
                named = dict(zip(self.sequence.joint_names, qpos))
                raw = np.asarray([named[n] for n in model.independent_names])
                if not np.isfinite(raw).all() or np.any(raw < model.bounds[:, 0] - 1e-6) or np.any(raw > model.bounds[:, 1] + 1e-6):
                    raise ValueError("Optimizer violated effective native limits")
                values = np.clip(raw, *model.bounds.T)
                residual = np.linalg.norm(model.positions(values) - targets, axis=1)
                return values, float(residual.max())
            independent, residual = candidate()
            # A sudden target jump can put G1's thumb in a different local branch.
            # Retry deterministic initial guesses with the SAME upstream optimizer.
            jumped = self.previous_targets is None or np.max(np.linalg.norm(targets - self.previous_targets, axis=1)) > .02
            # A target outside G1's verified planar finger motion cannot reach
            # the 5 mm recovery goal under ANY initialization. Avoid futile
            # multi-starts there; the first upstream optimum is still returned.
            plane_reachable = (self.motion_plane_z is None or
                               np.max(np.abs(targets[1:3, 2] - self.motion_plane_z)) <= .005)
            if model.robot_type == "g1" and residual > .005 and jumped and plane_reachable:
                seeds = [np.clip(np.zeros(len(model.independent_names)), *model.bounds.T),
                         model.bounds.mean(1), model.bounds[:, 0], model.bounds[:, 1]]
                for seed in seeds:
                    try:
                        proposal, distance = candidate(seed)
                        if distance < residual:
                            independent, residual = proposal, distance
                        if residual <= .005:
                            break
                    except RuntimeError:
                        continue
            # float32 conversion in upstream can put a boundary a few ulps out.
            if np.any(independent < model.bounds[:, 0] - 1e-6) or np.any(independent > model.bounds[:, 1] + 1e-6):
                raise ValueError("Optimizer violated effective native limits")
            independent = np.clip(independent, *model.bounds.T)
            expanded = model.expand(independent)
            values = np.array([expanded[n] for n in model.joint_names])
            self.sequence.last_qpos = independent.copy()
            self.previous_targets = targets.copy()
        except (ValueError, RuntimeError, ArithmeticError) as failure:
            self.sequence.last_qpos = previous
            error = f"{type(failure).__name__}: {failure}"
        return NativeHandResult(model.robot_type, model.side, model.joint_names, values,
                                float(timestamp), model.config_hash, values is not None, error,
                                (time.perf_counter() - started) * 1000)


class DexNativeRetargeter(NativeHandRetargeter):
    def __init__(self, calibration, robots=("h1_2", "gr1_t2", "g1"), sides=SIDES):
        from teleoperation.robots.native_hand import NativeHandModel
        if not isinstance(calibration, DexCalibration):
            raise TypeError("Dex native retargeting requires a fixed DexCalibration")
        self.calibration = calibration
        self.models = {(r, s): NativeHandModel(r, s) for r in robots for s in sides}
        self.solvers = {key: NativeVectorSolver(model) for key, model in self.models.items()}
        self.transforms, self.scales = {}, {}
        for key, model in self.models.items():
            basis, _ = model.palm_reference()
            self.transforms[key] = OPERATOR_TO_CANONICAL[model.side].T @ basis.T
            self.scales[key] = calibration.scale_for(model)

    def reset(self):
        for solver in self.solvers.values():
            solver.reset()

    def targets(self, frame, key):
        points = frame.hands[key[1]]
        if points is None:
            return None
        return points[self.models[key].human_tip_indices] @ self.transforms[key] * self.scales[key]

    def retarget(self, frame):
        if not isinstance(frame, CanonicalHandFrame):
            raise TypeError("Native dex retargeting requires a canonical frame")
        if frame.metadata.get("dex_input_profile") != self.calibration.profile or frame.units != "m":
            raise ValueError("Input and calibration profiles or units differ")
        result = {}
        for key, solver in self.solvers.items():
            targets = self.targets(frame, key)
            if targets is None:
                solver.reset()
                model = solver.model
                result[key] = NativeHandResult(*key, model.joint_names, None, frame.timestamp,
                                              model.config_hash, False, "missing/invalid input")
            else:
                result[key] = solver.solve(targets, frame.timestamp)
        return result
