"""Source-specific dex adapters -> existing native Vector IK for the viewer.

Only fingers are solved here. Wrist orientation never enters the optimizer.
The model runs at physical URDF scale; visualization enlargement is independent.
"""
from collections import deque
from dataclasses import replace
import numpy as np


class WristFingerIK:
    def __init__(self, source, sides):
        from teleoperation.retargeting.hand.dex_native import DexInputAdapter, NativeVectorSolver, _SpecVectorModel
        from teleoperation.robots.hand_spec import load_hand_spec
        self.adapter = DexInputAdapter(source)
        self.source = source
        self.specs = {side: load_hand_spec("l21", side) for side in sides}
        self.solvers = {side: NativeVectorSolver(_SpecVectorModel(spec)) for side, spec in self.specs.items()}
        self.lengths = {side: deque(maxlen=3) for side in sides}
        self.last_timestamp = None
        self.last_positions = {side: np.clip(np.zeros(spec.dof), *spec.bounds.T) for side, spec in self.specs.items()}
        self.states = {side: {"state": "waiting", "valid_frames": 0, "reason": "no hand input"} for side in sides}

    def update(self, frame):
        if frame is None:
            for side in self.states:
                self.states[side].update(state="hold", reason="missing input")
            return {}
        # MediaPipe live timestamps are milliseconds, VR/replay are seconds.
        factor = .001 if frame.metadata.get("timestamp_unit") in ("unix_ms", "unix_milliseconds") else 1.
        timestamp = float(frame.timestamp) * factor
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return {}
        dt = 1/30 if self.last_timestamp is None else min(.1, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        canonical = self.adapter.adapt(replace(frame, timestamp=timestamp))
        commands = {}
        for side, spec in self.specs.items():
            state = self.states[side]
            wrist = (frame.wrists or {}).get(side)
            points = canonical.hands.get(side)
            if points is None or wrist is None or not wrist.orientation_valid:
                state.update(state="hold", reason="missing hand or wrist tracking")
                self.lengths[side].clear()
                continue
            points = np.asarray(points, dtype=float)
            length = float(np.linalg.norm(points[11] - points[0]))
            if not np.isfinite(length) or length <= 1e-6:
                state.update(state="hold", reason="degenerate palm length")
                continue
            self.lengths[side].append(length)
            scale = spec.reference_palm_length_m / float(np.median(self.lengths[side]))
            vectors = (points[list(spec.human_tip_indices)] - points[0]) @ spec.native_to_canonical.T * scale
            result = self.solvers[side].solve(vectors, timestamp)
            if not result.valid:
                state.update(state="hold", reason=result.error)
                continue
            # Reuse URDF velocity semantics, without mixing finger filtering into
            # the unit-gain wrist orientation display.
            speed = np.minimum(spec.velocity_limits, 2.) * max(dt, 1e-4)
            previous = self.last_positions[side]
            q = np.clip(result.positions, previous - speed, previous + speed)
            self.last_positions[side] = q
            commands[side] = spec.expand(q)
            state.update(state="tracking", reason="", solve_ms=result.solve_ms,
                         palm_length_m=length, positions_rad=q.tolist())
            state["valid_frames"] += 1
        return commands

    def summary(self):
        return {"backend": "dex-retargeting Vector IK", "hand_model": "l21",
                "source_adapter": self.source, "scaling": "reference_palm_length / median_source_palm_length_3",
                "sides": self.states}
