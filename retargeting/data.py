"""H5 dataset utilities for shared left/right hand retargeting."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from retargeting.tracking import (
    DEFAULT_MAX_CENTER_DISPLACEMENT,
    DEFAULT_MAX_SHAPE_RMSE,
    HandIdentityTracker,
    ensure_hand25,
)
from retargeting.coordinates import COORDINATE_FRAME


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
        buffers = {
            side: deque(maxlen=self.receptive_field) for side in HAND_SIDES
        }
        samples: list[dict] = []
        previous_frame_id = None
        identity_tracker = HandIdentityTracker(
            max_center_displacement=self.max_center_displacement,
            max_shape_rmse=self.max_shape_rmse,
        )

        for frame_index in range(self.frame_start, self.frame_end):
            frame_id = _scalar_or_none(self.frame_ids[frame_index])
            if (
                self.reset_on_gaps
                and previous_frame_id is not None
                and frame_id is not None
                and frame_id != previous_frame_id + 1
            ):
                for buffer in buffers.values():
                    buffer.clear()
                identity_tracker.reset()
            previous_frame_id = frame_id

            raw_hands = {
                side: self._raw_hands[side][frame_index]
                for side in HAND_SIDES
            }
            if self.track_identity:
                tracked_hands = identity_tracker.update(
                    left_hand=raw_hands["left"],
                    right_hand=raw_hands["right"],
                )
            else:
                tracked_hands = raw_hands

            current: dict[str, np.ndarray] = {}
            for side in HAND_SIDES:
                raw_points = tracked_hands[side]
                if not _is_valid_hand_frame(raw_points):
                    buffers[side].clear()
                    continue

                hand = ensure_hand25(
                    raw_points,
                    scale_factor=self.scale_factor,
                )
                if not np.isfinite(hand).all():
                    raise ValueError(
                        f"Converted {side} hand contains NaN or Inf at "
                        f"frame {frame_index}"
                    )
                buffers[side].append(hand)
                current[side] = hand

            valid_sides = {
                side: side in current and len(buffers[side]) == self.receptive_field
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
                    sample[f"{side}_input"] = np.stack(
                        tuple(buffers[side]),
                        axis=0,
                    ).astype(np.float32, copy=False)
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


def load_twohand_h5(
    path: str | Path,
    require_aligned: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load and validate the supported two-hand H5 layouts."""
    h5_path = Path(path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"H5 input was not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        if require_aligned:
            coordinate_frame = h5_file.attrs.get("coordinate_frame")
            if isinstance(coordinate_frame, bytes):
                coordinate_frame = coordinate_frame.decode("utf-8")
            if coordinate_frame != COORDINATE_FRAME:
                raise ValueError(
                    "H5 input must be pre-aligned and declare "
                    f"coordinate_frame={COORDINATE_FRAME!r}; "
                    f"got {coordinate_frame!r}. Run the align_h5 command first."
                )
        if all(key in h5_file for key in _SIDE_KEYS.values()):
            left = np.asarray(h5_file[_SIDE_KEYS["left"]][:])
            right = np.asarray(h5_file[_SIDE_KEYS["right"]][:])
            frame_ids = _read_optional_vector(
                h5_file,
                "frame_ids",
                left.shape[0],
                dtype=np.int64,
            )
            timestamps = _read_optional_vector(
                h5_file,
                "timestamps",
                left.shape[0],
                dtype=np.float64,
            )
        else:
            groups = [
                key
                for key in h5_file.keys()
                if isinstance(h5_file[key], h5py.Group)
                and "l_glove_pos" in h5_file[key]
                and "r_glove_pos" in h5_file[key]
            ]
            if len(groups) != 1:
                raise ValueError(
                    "Expected root datasets "
                    "'left_hand_keypoints'/'right_hand_keypoints' or "
                    "exactly one group with 'l_glove_pos'/'r_glove_pos'"
                )
            group = h5_file[groups[0]]
            left = np.asarray(group["l_glove_pos"][:])
            right = np.asarray(group["r_glove_pos"][:])
            frame_ids = _read_optional_vector(
                group,
                "frame_ids",
                left.shape[0],
                dtype=np.int64,
            )
            timestamps = _read_optional_vector(
                group,
                "timestamps",
                left.shape[0],
                dtype=np.float64,
            )

    _validate_hand_array(left, "left")
    _validate_hand_array(right, "right")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "Left/right frame counts do not match: "
            f"{left.shape[0]} != {right.shape[0]}"
        )
    if frame_ids.shape[0] != left.shape[0]:
        raise ValueError("frame_ids length does not match hand data")
    if timestamps.shape[0] != left.shape[0]:
        raise ValueError("timestamps length does not match hand data")
    if not np.isfinite(frame_ids).all() or not np.isfinite(timestamps).all():
        raise ValueError("frame_ids and timestamps must be finite")

    return (
        frame_ids,
        timestamps,
        {"left": left.astype(np.float32, copy=False), "right": right.astype(np.float32, copy=False)},
    )


def _read_optional_vector(
    container,
    name: str,
    length: int,
    dtype,
) -> np.ndarray:
    if name not in container:
        return np.arange(length, dtype=dtype)
    values = np.asarray(container[name][:], dtype=dtype).reshape(-1)
    if values.shape[0] != length:
        raise ValueError(f"{name} length does not match hand data")
    return values


def _validate_hand_array(points: np.ndarray, side: str) -> None:
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            f"{side} hand data must have shape (frames, joints, 3), "
            f"got {points.shape}"
        )
    if points.shape[1] not in (21, 25):
        raise ValueError(
            f"{side} hand data must contain 21 or 25 points, "
            f"got {points.shape[1]}"
        )
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError(f"{side} hand data must be numeric")


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
