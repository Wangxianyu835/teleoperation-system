"""Native dex Vector pipeline, separate from the L21 model-result interface.

Camera and VR profiles are explicit. The VR extraction follows the local
TransHandR reference; its existing model pipeline is never modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.retargeting.hand.coordinates import build_palm_basis, OPERATOR_TO_CANONICAL

SIDES = ("left", "right")
PROFILES = {"camera": "dex_camera_world_v1", "visionpro": "dex_visionpro_palm_v1"}


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
