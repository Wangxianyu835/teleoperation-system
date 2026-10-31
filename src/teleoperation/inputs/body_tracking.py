"""Temporal validity and recovery for SMPLer-X body results."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.contracts.body import SmplerXResult
from teleoperation.contracts.observations import UpperBodyObservation
from teleoperation.contracts.wrist import WristObservation
from teleoperation.inputs.smplerx_geometry import smplerx_observations


SIDES = ("left", "right")


@dataclass(frozen=True)
class TrackedBody:
    body: UpperBodyObservation
    wrists: dict[str, WristObservation]
    state_by_side: dict[str, str]
    reason_by_side: dict[str, str]
    sample_id: dict[str, int | None]
    age_s: dict[str, float]
    fresh: dict[str, bool]


def _invalid_wrist(reason: str) -> WristObservation:
    return WristObservation(None, "smplx_camera_axes", "smplx_wrist_local", "smplerx",
                            orientation_valid=False, position_valid=False, reason=reason)


class BodyTracker:
    """Hold reliable body samples and accept a jump only after recovery samples."""

    def __init__(self, *, max_age_s=.5, recovery_samples=3,
                 position_jump_m=.15, rotation_jump_rad=1.2):
        for name, value in (("max_age_s", max_age_s), ("position_jump_m", position_jump_m),
                            ("rotation_jump_rad", rotation_jump_rad)):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if int(recovery_samples) != recovery_samples or recovery_samples < 1:
            raise ValueError("recovery_samples must be a positive integer")
        self.max_age_s = float(max_age_s)
        self.recovery_samples = int(recovery_samples)
        self.position_jump_m = float(position_jump_m)
        self.rotation_jump_rad = float(rotation_jump_rad)
        self._session_id = None
        self._last_sequence = -1
        self._metadata = {'source': 'smplerx', 'source_coordinate_frame': 'smplx_camera_root_relative_m',
                          'source_units': 'm'}
        self._accepted_arms = {side: None for side in SIDES}
        self._accepted_world = {side: None for side in SIDES}
        self._accepted_wrists = {side: None for side in SIDES}
        self._accepted_time = {side: None for side in SIDES}
        self._accepted_sequence = {side: None for side in SIDES}
        self._candidate = {side: None for side in SIDES}
        self._candidate_count = {side: 0 for side in SIDES}
        self._candidate_time = {side: None for side in SIDES}
        self._recovery_required = {side: False for side in SIDES}
        self._state = {side: "calibrating" for side in SIDES}
        self._reason = {side: "no body sample" for side in SIDES}

    def reset(self, session_id: str | None = None) -> None:
        self.__init__(max_age_s=self.max_age_s, recovery_samples=self.recovery_samples,
                      position_jump_m=self.position_jump_m,
                      rotation_jump_rad=self.rotation_jump_rad)
        self._session_id = session_id

    def _empty_body(self, timestamp=None) -> UpperBodyObservation:
        return UpperBodyObservation(
            {side: self._accepted_arms[side] for side in SIDES},
            {side: self._accepted_arms[side] is not None and self._state[side] == 'tracking' for side in SIDES},
            timestamp=timestamp, coordinate_frame="smplx_image_normalized", units="normalized",
            world_arms={side: self._accepted_world[side] for side in SIDES},
            metadata={**self._metadata, 'tracking': True, 'tracking_state': dict(self._state)},
        )

    def _emit(self, now_s: float, *, timestamp=None, fresh=None) -> TrackedBody:
        ages = {}
        for side in SIDES:
            stamp = self._accepted_time[side]
            ages[side] = float("inf") if stamp is None else max(0., float(now_s) - stamp)
            if stamp is not None and ages[side] > self.max_age_s:
                self._recovery_required[side] = True
                self._state[side] = "lost_hold"
                self._reason[side] = "body sample expired"
        body = self._empty_body(timestamp if timestamp is not None else max(
            (stamp for stamp in self._accepted_time.values() if stamp is not None), default=None))
        body.metadata.update(body_sample_ids=dict(self._accepted_sequence),
                             body_age_s={side: age if math.isfinite(age) else None for side, age in ages.items()},
                             body_fresh={side: bool((fresh or {}).get(side, False)) for side in SIDES},
                             body_reason=dict(self._reason))
        return TrackedBody(
            body,
            {side: (self._accepted_wrists[side] if self._state[side] == 'tracking'
                    and self._accepted_wrists[side] is not None else _invalid_wrist(self._reason[side])) for side in SIDES},
            dict(self._state), dict(self._reason), dict(self._accepted_sequence), ages,
            {side: bool((fresh or {}).get(side, False)) for side in SIDES},
        )

    def _jump(self, side: str, points: np.ndarray, wrist: WristObservation) -> bool:
        previous = self._accepted_world[side]
        if previous is None:
            return False
        if np.linalg.norm(np.asarray(points) - previous) > self.position_jump_m:
            return True
        old = self._accepted_wrists[side]
        if old is not None and old.orientation_valid and wrist.orientation_valid:
            angle = (Rotation.from_quat(wrist.orientation_xyzw)
                     * Rotation.from_quat(old.orientation_xyzw).inv()).magnitude()
            if angle > self.rotation_jump_rad:
                return True
        return False

    def _candidate_is_consistent(self, side: str, points: np.ndarray, wrist: WristObservation) -> bool:
        candidate = self._candidate[side]
        if candidate is None:
            return False
        old_points, old_wrist = candidate
        if np.linalg.norm(np.asarray(points) - old_points) > self.position_jump_m * .25:
            return False
        if old_wrist is not None and old_wrist.orientation_valid and wrist.orientation_valid:
            angle = (Rotation.from_quat(wrist.orientation_xyzw)
                     * Rotation.from_quat(old_wrist.orientation_xyzw).inv()).magnitude()
            if angle > self.rotation_jump_rad * .25:
                return False
        return True

    def _accept(self, side, arms, world, wrist, stamp, sequence):
        self._accepted_arms[side] = np.asarray(arms, dtype=float).copy()
        self._accepted_world[side] = np.asarray(world, dtype=float).copy()
        self._accepted_wrists[side] = wrist
        self._accepted_time[side] = float(stamp)
        self._accepted_sequence[side] = int(sequence)
        self._candidate[side] = None
        self._candidate_count[side] = 0
        self._candidate_time[side] = None
        self._recovery_required[side] = False
        self._state[side] = "tracking"
        self._reason[side] = ""

    def update(self, result: SmplerXResult | None, *, now_s: float) -> TrackedBody:
        if not math.isfinite(float(now_s)):
            raise ValueError("now_s must be finite")
        if result is None:
            for side in SIDES:
                self._state[side] = "hold" if self._accepted_time[side] is not None else "calibrating"
                self._reason[side] = "no new body result"
            return self._emit(now_s, fresh={})
        session = result.frame_id.session_id
        if self._session_id != session:
            self.reset(session)
        if result.frame_id.sequence <= self._last_sequence:
            for side in SIDES:
                if self._state[side] == 'tracking':
                    self._state[side] = 'hold'
                    self._reason[side] = 'duplicate or nonmonotonic body sample'
            return self._emit(now_s, timestamp=result.frame_id.capture_time_s, fresh={})
        self._last_sequence = result.frame_id.sequence
        try:
            body, wrists = smplerx_observations(result)
            self._metadata = dict(body.metadata)
        except (KeyError, TypeError, ValueError) as error:
            for side in SIDES:
                self._candidate[side] = None
                self._candidate_count[side] = 0
                self._recovery_required[side] = self._accepted_time[side] is not None
                self._state[side] = "fault"
                self._reason[side] = str(error)
            return self._emit(now_s, timestamp=result.frame_id.capture_time_s, fresh={})
        fresh = {}
        for side in SIDES:
            points = body.world_arms[side]
            wrist = wrists[side]
            stamp = result.frame_id.capture_time_s
            age = float(now_s) - stamp
            if age < 0 or age > self.max_age_s:
                self._candidate[side] = None
                self._candidate_count[side] = 0
                self._recovery_required[side] = self._accepted_time[side] is not None
                self._state[side] = "lost_hold" if self._accepted_time[side] is not None else "calibrating"
                self._reason[side] = "body sample expired or from the future"
                fresh[side] = False
                continue
            if not body.valid[side] or points is None or not wrist.orientation_valid:
                self._candidate[side] = None
                self._candidate_count[side] = 0
                self._recovery_required[side] = self._accepted_time[side] is not None
                self._state[side] = "hold" if self._accepted_time[side] is not None else "calibrating"
                self._reason[side] = result.reason or wrist.reason or "invalid body side"
                fresh[side] = False
                continue
            if not self._jump(side, points, wrist) and not self._recovery_required[side]:
                self._accept(side, body.arms[side], points, wrist, stamp, result.frame_id.sequence)
                fresh[side] = True
                continue
            candidate_time = self._candidate_time[side]
            if (candidate_time is not None and 0 < stamp - candidate_time <= self.max_age_s
                    and self._candidate_is_consistent(side, points, wrist)):
                self._candidate_count[side] += 1
            else:
                self._candidate[side] = (np.asarray(points, dtype=float).copy(), wrist)
                self._candidate_count[side] = 1
            self._candidate_time[side] = stamp
            if self._candidate_count[side] >= self.recovery_samples:
                self._accept(side, body.arms[side], points, wrist, stamp, result.frame_id.sequence)
                fresh[side] = True
            else:
                self._state[side] = "reacquiring"
                self._reason[side] = "body jump; waiting for consistent samples"
                fresh[side] = False
        emitted = self._emit(now_s, timestamp=result.frame_id.capture_time_s, fresh=fresh)
        return emitted
