"""H5 dataset utilities for shared left/right hand retargeting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from teleoperation.retargeting.hand.processing import CanonicalHandProcessor
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.data.hand_h5 import load_twohand_h5, read_coordinate_alignment
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.contracts.coordinates import COORDINATE_FRAME, validate_coordinate_alignment


HAND_SIDES = ("left", "right")
_SIDE_KEYS = {
    "left": "left_hand_keypoints",
    "right": "right_hand_keypoints",
}


class TwoHandH5Dataset:
    """Create masked three-frame samples from a two-hand H5 recording.

    The current camera recorder writes datasets at the H5 root:
    ``left_hand_keypoints`` and ``right_hand_keypoints`` with shape
    ``(frames, 21, 3)``. The loader also accepts the older grouped schema
    containing ``l_glove_pos`` and ``r_glove_pos``.
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
    ):
        self.path = Path(path)
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
        self.coordinate_frame = COORDINATE_FRAME
        with h5py.File(self.path, "r") as source:
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
        processor = CanonicalHandProcessor(
            scale_factor=self.scale_factor,
            track_identity=self.track_identity,
            max_center_displacement=self.max_center_displacement,
            max_shape_rmse=self.max_shape_rmse,
        )

        buffer = TemporalBuffer(self.receptive_field)
        for frame_index in range(self.frame_start, self.frame_end):
            frame_id = _scalar_or_none(self.frame_ids[frame_index])
            if (
                self.reset_on_gaps
                and previous_frame_id is not None
                and frame_id is not None
                and frame_id != previous_frame_id + 1
            ):
                processor.reset()
                buffer.reset()
            previous_frame_id = frame_id

            raw_hands = {
                side: self._raw_hands[side][frame_index]
                for side in HAND_SIDES
            }
            canonical = processor.process_frame(RawHandFrame(raw_hands, _scalar_or_none(self.timestamps[frame_index]), "h5", {"frame_index": frame_index}))
            window = buffer.append(canonical)
            payload = None if window is None else window.to_payload()
            current = {side: points for side, points in canonical.hands.items() if points is not None}
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


class TwoHandH5ChunkedGenerator:
    """Batch generator for :class:`TwoHandH5Dataset`."""

    def __init__(
        self,
        dataset: TwoHandH5Dataset,
        batch_size: int,
        shuffle: bool = True,
        random_seed: int = 1234,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.shuffle = bool(shuffle)
        self.random = np.random.RandomState(random_seed)
        self.num_batches = (
            len(dataset) + self.batch_size - 1
        ) // self.batch_size

    def num_frames(self) -> int:
        return len(self.dataset)

    def next_epoch(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            samples = [self.dataset.samples[int(i)] for i in batch_indices]
            yield {
                "left_input": _stack(samples, "left_input"),
                "right_input": _stack(samples, "right_input"),
                "left_target": _stack(samples, "left_target"),
                "right_target": _stack(samples, "right_target"),
                "left_valid": _mask(samples, "left_valid"),
                "right_valid": _mask(samples, "right_valid"),
                "frame_index": np.asarray(
                    [sample["frame_index"] for sample in samples],
                    dtype=np.int64,
                ),
                "frame_id": np.asarray(
                    [sample["frame_id"] for sample in samples],
                ),
                "timestamp": np.asarray(
                    [sample["timestamp"] for sample in samples],
                    dtype=np.float64,
                ),
            }


def _is_valid_hand_frame(points: np.ndarray | None) -> bool:
    if points is None:
        return False
    points = np.asarray(points)
    return bool(np.isfinite(points).all() and not np.all(points == 0))


def _scalar_or_none(value):
    if value is None:
        return None
    scalar = np.asarray(value).reshape(-1)
    return None if scalar.size == 0 else scalar[0].item()


def _stack(samples: list[dict], key: str) -> np.ndarray:
    return np.stack([sample[key] for sample in samples], axis=0).astype(
        np.float32,
        copy=False,
    )


def _mask(samples: list[dict], key: str) -> np.ndarray:
    return np.asarray([sample[key] for sample in samples], dtype=bool)
