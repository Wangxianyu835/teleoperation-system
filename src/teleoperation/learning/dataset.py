"""H5 dataset utilities for shared left/right hand retargeting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.data.hand_h5 import load_twohand_h5, read_coordinate_alignment
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT, coordinate_frame_for_alignment, is_world_alignment, PALM_LENGTH_PROFILE, PALM_LENGTH_ALIGNMENT


HAND_SIDES = ("left", "right")
_SIDE_KEYS = {
    "left": "left_hand_keypoints",
    "right": "right_hand_keypoints",
}


class TwoHandH5Dataset:
    """Create masked three-frame samples from a two-hand H5 recording.

    The current camera recorder writes datasets at the H5 root:
    ``left_hand_keypoints`` and ``right_hand_keypoints`` with shape
    ``(frames, 25, 3)`` after explicit world alignment. Grouped glove and
    image-normalized training recordings are rejected.
    """

    def __init__(
        self,
        path: str | Path,
        receptive_field: int = 3,
        scale_factor: float = 1.0,
        frame_start: int = 0,
        frame_end: int | None = None,
        reset_on_gaps: bool = True,
        track_identity: bool = True,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
        input_profile: str | None = None,
    ):
        self.path = Path(path)
        if input_profile not in (None, PALM_LENGTH_PROFILE):
            raise ValueError("Unknown training input profile")
        self.input_profile = input_profile
        self.receptive_field = int(receptive_field)
        self.scale_factor = float(scale_factor)
        self.frame_start = int(frame_start)
        self.reset_on_gaps = bool(reset_on_gaps)
        self.track_identity = bool(track_identity)
        self.max_center_displacement = float(max_center_displacement)
        self.max_shape_rmse = float(max_shape_rmse)
        if self.receptive_field < 1:
            raise ValueError("receptive_field must be positive")

        self.coordinate_alignment = read_coordinate_alignment(self.path)
        self.is_world = is_world_alignment(self.coordinate_alignment)
        self.is_canonical = self.coordinate_alignment == WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT
        if not self.is_canonical:
            raise ValueError("Palm-length training requires canonical world metre H5")
        self.source_coordinate_alignment = self.coordinate_alignment
        self.units = "1" if input_profile == PALM_LENGTH_PROFILE else ("m" if self.is_world else "unknown")
        if self.is_world:
            if self.scale_factor != 1.0:
                raise ValueError("World Dataset requires scale_factor=1.0; preserve human metre dimensions")
            self.track_identity = False
        self.coordinate_frame = coordinate_frame_for_alignment(self.coordinate_alignment)
        with h5py.File(self.path, "r") as source:
            self.coordinate_metadata = dict(source.attrs)
            space = source.attrs.get("source_landmark_space")
            self.source_landmark_space = space.decode("utf-8") if isinstance(space, bytes) else space

        (
            self.frame_ids,
            self.timestamps,
            self._raw_hands,
        ) = load_twohand_h5(self.path)
        self.frame_count = int(self.frame_ids.shape[0])

        if self.frame_start < 0 or self.frame_start > self.frame_count:
            raise ValueError(
                f"frame_start must be in [0, {self.frame_count}], "
                f"got {self.frame_start}"
            )

        requested_end = self.frame_count if frame_end is None else int(frame_end)
        if requested_end < self.frame_start or requested_end > self.frame_count:
            raise ValueError(
                f"frame_end must be in [{self.frame_start}, {self.frame_count}], "
                f"got {requested_end}"
            )
        self.frame_end = requested_end

        self.samples = self._build_samples()
        if input_profile == PALM_LENGTH_PROFILE:
            self.coordinate_alignment = PALM_LENGTH_ALIGNMENT

    def __len__(self) -> int:
        return len(self.samples)

    def side_counts(self) -> dict[str, int]:
        return {
            side: sum(1 for sample in self.samples if sample[f"{side}_valid"])
            for side in HAND_SIDES
        }

    def _build_samples(self) -> list[dict]:
        samples: list[dict] = []
        previous_frame_id = None
        buffer = TemporalBuffer(self.receptive_field)
        for frame_index in range(self.frame_start, self.frame_end):
            frame_id = _scalar_or_none(self.frame_ids[frame_index])
            if (
                self.reset_on_gaps
                and previous_frame_id is not None
                and frame_id is not None
                and frame_id != previous_frame_id + 1
            ):
                buffer.reset()
            previous_frame_id = frame_id

            raw_hands = {
                side: self._raw_hands[side][frame_index]
                for side in HAND_SIDES
            }
            timestamp = _scalar_or_none(self.timestamps[frame_index])
            hands = {}
            for side, points in raw_hands.items():
                if points.shape != (25, 3):
                    raise ValueError('Aligned world Dataset requires (25,3) hands')
                hands[side] = points if np.isfinite(points).all() and np.any(points) else None
            canonical = CanonicalHandFrame(hands, timestamp, 'h5', self.coordinate_metadata)
            if self.input_profile == PALM_LENGTH_PROFILE:
                from teleoperation.retargeting.hand.normalization import validate_metric_hand_frame
                canonical = validate_metric_hand_frame(canonical)
            window = buffer.append(canonical)
            if window is not None and self.input_profile == PALM_LENGTH_PROFILE:
                from teleoperation.retargeting.hand.normalization import normalize_hand_window
                ready = {side:window.hands[side] is not None for side in HAND_SIDES}
                window = normalize_hand_window(window)
                for side in HAND_SIDES:
                    if ready[side] and window.hands[side] is None:
                        buffer.reset_side(side)
            payload = None if window is None else window.to_payload()
            current = {side: points for side, points in canonical.hands.items() if points is not None}
            if self.input_profile == PALM_LENGTH_PROFILE:
                current = {} if window is None else {side: points[-1] for side, points in window.hands.items() if points is not None}
            for side, hand in current.items():
                if not np.isfinite(hand).all():
                    raise ValueError(
                        f"Converted {side} hand contains NaN or Inf at "
                        f"frame {frame_index}"
                    )

            valid_sides = {
                side: payload is not None
                and payload["hands"][side] is not None
                for side in HAND_SIDES
            }
            # Keep one sample for every frame that has enough history to be a
            # target. A frame may have neither hand valid; the zero tensors and
            # masks let callers preserve the original timeline and skip its
            # loss contribution without treating missing data as real points.
            if frame_index < self.frame_start + self.receptive_field - 1:
                continue

            sample = {
                "frame_index": frame_index,
                "frame_id": frame_id if frame_id is not None else frame_index,
                "timestamp": _scalar_or_none(self.timestamps[frame_index]),
                "window_metadata": {} if window is None else dict(window.metadata),
            }
            for side in HAND_SIDES:
                if valid_sides[side]:
                    sample[f"{side}_input"] = payload["hands"][side]
                    sample[f"{side}_target"] = current[side][None, :, :]
                else:
                    sample[f"{side}_input"] = np.zeros(
                        (self.receptive_field, 25, 3),
                        dtype=np.float32,
                    )
                    sample[f"{side}_target"] = np.zeros(
                        (1, 25, 3),
                        dtype=np.float32,
                    )
                sample[f"{side}_valid"] = valid_sides[side]
            samples.append(sample)

        return samples

    def iter_samples(self) -> Iterator[dict]:
        """Iterate samples in chronological order without shuffling."""
        yield from self.samples


def _scalar_or_none(value):
    if value is None:
        return None
    scalar = np.asarray(value).reshape(-1)
    return None if scalar.size == 0 else scalar[0].item()
