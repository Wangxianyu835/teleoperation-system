"""Neutral alignment of source-space wrist motion, without hand reflections."""
import numpy as np
from scipy.spatial.transform import Rotation

# avp_stream already applies YUP2ZUP: SDK +X=right,+Y=forward,+Z=up.
VR_TO_BASE = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


class WristTracker:
    """Per-side alignment and SO(3) smoothing; output is a base-space rotation delta.

    delta = C R(t) R(0)^T C^T. Robot target = delta * robot_initial.
    Source local axes cancel, so SDK wrist axes and SMPL-X wrist axes need not
    match and neither is passed through the hand model's left reflection.
    """
    def __init__(self, camera_to_base, calibration_frames=10):
        self.camera_to_base = np.asarray(camera_to_base, dtype=float)
        if self.camera_to_base.shape != (3, 3) or not np.isfinite(self.camera_to_base).all() or not np.allclose(self.camera_to_base.T @ self.camera_to_base, np.eye(3), atol=1e-6):
            raise ValueError("Camera axis map must be orthogonal")
        self.calibration_frames = max(1, int(calibration_frames))
        self.signature = None
        self.samples = []
        self.neutral = None
        self.neutral_position = None
        self.delta = Rotation.identity()
        self.displacement = np.zeros(3)
        self.status = "uncalibrated"
        self.reason = "no wrist observation"
        self.generation = 0
        self.elapsed_missing = 0.
        self.last_timestamp = None
        self.last_observed = None
        self.spike_frames = 0

    def update(self, observation, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1/30
        elapsed = dt
        dt = min(dt, .1)
        if observation is None or not observation.orientation_valid:
            self.elapsed_missing += elapsed
            self.status = "uncalibrated" if self.neutral is None else ("lost_hold" if self.elapsed_missing >= .5 else "hold")
            self.reason = "missing wrist" if observation is None else observation.reason
            pending_body = (observation is not None and observation.source == 'smplerx'
                            and observation.reason == 'no new body result' and self.elapsed_missing < .5)
            if self.neutral is None and not pending_body:
                self.samples.clear()
            return
        if (observation.source == 'camera_palm'
                or observation.reference_frame == 'mediapipe_hand_world_axes'):
            self.update(None, elapsed)
            self.reason = "unsupported wrist reference frame"
            return
        signature = (observation.source, observation.reference_frame, observation.local_frame)
        if signature != self.signature:
            self.signature = signature
            self.samples.clear()
            self.neutral = None
            self.neutral_position = None
            self.delta = Rotation.identity()
            self.displacement = np.zeros(3)
            self.last_timestamp = None
            self.last_observed = None
            self.elapsed_missing = 0.
            self.spike_frames = 0
        if observation.timestamp is not None:
            stamp = float(observation.timestamp)
            if self.last_timestamp is not None and stamp == self.last_timestamp:
                # Cached SDK packets are waiting, not additional calibration samples.
                self.elapsed_missing += elapsed
                if self.neutral is None:
                    if self.elapsed_missing >= .5:
                        self.samples.clear()
                    self.status = "uncalibrated" if self.elapsed_missing >= .5 else "calibrating"
                else:
                    self.status = "lost_hold" if self.elapsed_missing >= .5 else "hold"
                self.reason = "waiting for a new wrist sample"
                return
            if not np.isfinite(stamp) or (self.last_timestamp is not None and stamp <= self.last_timestamp):
                self.update(None, elapsed)
                self.reason = "stale or nonmonotonic wrist sample"
                return
            self.last_timestamp = stamp
        if observation.source == 'smplerx':
            self.elapsed_missing = 0.
        if observation.reference_frame == "visionpro_sdk_zup":
            c = VR_TO_BASE
        elif observation.reference_frame == "smplx_camera_axes":
            c = self.camera_to_base
        else:
            self.status, self.reason = "uncalibrated", "unsupported wrist reference frame"
            return
        current = Rotation.from_quat(observation.orientation_xyzw)
        if self.neutral is None:
            self.elapsed_missing = 0.
            if self.samples and (current * self.samples[0][0].inv()).magnitude() > np.deg2rad(8):
                self.samples.clear()
            if self.samples and observation.position_valid and self.samples[0][1] is not None:
                if np.linalg.norm(np.asarray(observation.position) - self.samples[0][1]) > .03:
                    self.samples.clear()
            self.samples.append((current, observation.position if observation.position_valid else None))
            self.status, self.reason = "calibrating", "hold a natural palm pose briefly"
            if len(self.samples) < self.calibration_frames:
                return
            self.neutral = Rotation.from_quat([r.as_quat() for r, _ in self.samples]).mean()
            if all(p is not None for _, p in self.samples):
                self.neutral_position = np.mean([p for _, p in self.samples], axis=0)
            self.delta = Rotation.identity()
            self.generation += 1
            self.last_observed = current
            self.status, self.reason = "aligned", ""
            self.elapsed_missing = 0.
            return
        # Reject transient tracking flips, but accept a persistent new estimate
        # after three frames and approach it through the rate limiter.
        jump = (current * self.last_observed.inv()).magnitude() if self.last_observed is not None else 0.
        if jump > max(np.deg2rad(60.), np.deg2rad(360.) * dt) and not self.elapsed_missing:
            self.spike_frames += 1
            if self.spike_frames < 3:
                self.status, self.reason = "hold", "wrist rotation spike"
                return
        self.spike_frames = 0
        self.last_observed = current
        raw = c @ (current * self.neutral.inv()).as_matrix() @ c.T
        wanted = Rotation.from_matrix(raw)
        error = wanted * self.delta.inv()
        angle = error.magnitude()
        fraction = min(1. - np.exp(-dt / .08), np.deg2rad(180.) * dt / max(angle, 1e-9))
        self.delta = Rotation.from_rotvec(error.as_rotvec() * fraction) * self.delta
        if observation.position_valid and self.neutral_position is not None:
            requested = c @ (np.asarray(observation.position) - self.neutral_position)
            # Relative VR reach is bounded to 20 cm from the alignment pose.
            requested *= min(1., .2 / max(np.linalg.norm(requested), 1e-9))
            step = requested - self.displacement
            self.displacement += step * min(1., .3 * dt / max(np.linalg.norm(step), 1e-9))
        self.status = "reacquiring" if self.elapsed_missing else "tracking"
        self.reason = ""
        self.elapsed_missing = 0.

    def summary(self):
        return {"state": self.status, "reason": self.reason, "source_frames": self.signature,
                "calibration_samples": len(self.samples), "generation": self.generation,
                "delta_xyzw": self.delta.as_quat().tolist(),
                "displacement_m": self.displacement.tolist(),
                "neutral_xyzw": None if self.neutral is None else self.neutral.as_quat().tolist(),
                "neutral_position_m": None if self.neutral_position is None else self.neutral_position.tolist(),
                "alignment_version": "source_spatial_delta_v1"}
