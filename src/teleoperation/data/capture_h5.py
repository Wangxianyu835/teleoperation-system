"""原始 21 点抓取容器：把一帧帧 MediaPipe 结果落成一个可回放的 H5。

为什么有这个文件
----------------
``hand align``（``apps/hand_align.py``）与 ``--input-h5`` 回放
（``devtools/replay_capture_to_l21.py``）都要求输入是"根数据集
``left_hand_keypoints`` / ``right_hand_keypoints``，形状 ``(T,21,3)``"的原始抓取
H5。队友原来的采集脚本（产物形如 ``input/visual_hand_data_20260912_112108.h5``）
没有随仓库入库，本机也已不存在，所以"摄像头 -> H5"这一环缺一个入口。本模块
只提供这块拼图：数据集名、形状、dtype 与队友的读取器逐字段对齐，并且不做任何
坐标变换（对齐是 ``hand align`` 的职责）。

沿用既有约定：没检出的手写全零
------------------------------
``retargeting.hand.alignment.align_recording`` 会把整行全零的帧统计成
``zero_source_frames``，``devtools/replay_capture_to_l21.py`` 会把全零行还原成
``None``（当作"没手"，而不是"退化的手"）。本模块写全零行，并额外写一份
``left_valid`` / ``right_valid``（bool）让"这一帧到底有没有检出"可被直接读取；
队友的读取器忽略未知数据集，所以多这一列不影响 ``hand align`` 与回放。

落盘布局（h5py）
----------------
    /left_hand_keypoints  (T,21,3) float32   全零 = 未检出
    /right_hand_keypoints (T,21,3) float32
    /left_valid           (T,)     bool
    /right_valid          (T,)     bool
    /frame_ids            (T,)     int64     取自 MediaPipe metadata.frame_index
    /timestamps           (T,)     float64   相对第一帧的秒（第一帧 0.0）
    /unix_ms              (T,)     int64     MediaPipe VIDEO 时间戳（Unix 毫秒）

属性（attrs）里 ``source_landmark_space='mediapipe_normalized'`` 与
``timestamp_unit='relative_seconds'`` 都是队友 ``inputs/mediapipe.py`` 与
``datasets/raw/*.h5`` 已在用的口径，``hand align`` 的
``_validate_source_landmark_space`` 只接受 ``mediapipe_normalized`` 或未声明。

写入是增量的（``maxshape=(None,21,3)``）：异常或 Ctrl+C 中断时已写入的帧仍在
文件里，收尾属性由 ``close()`` 补写；调用方必须保证 ``close()`` 一定会被调用。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from teleoperation.contracts.constants import HAND_SIDES

CAPTURE_SIDE_KEYS = {"left": "left_hand_keypoints", "right": "right_hand_keypoints"}
VALID_SIDE_KEYS = {"left": "left_valid", "right": "right_valid"}
LANDMARK_COUNT = 21
POINT_DIM = 3
TIMESTAMP_UNIT = "relative_seconds"
SOURCE_LANDMARK_SPACE = "mediapipe_normalized"
MISSING_HAND_ENCODING = "zeros"
RECORDER = "teleoperation.data.capture_h5"


def _utc_now() -> str:
    """ISO-8601 UTC timestamp with a trailing Z, e.g. 2026-10-10T09:31:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def landmarks_or_zeros(value) -> tuple[np.ndarray, bool]:
    """Return one frame's ``(21,3)`` float32 landmarks plus its validity flag.

    ``None`` means the side was not detected: that becomes an all-zero row, the
    convention ``align_recording`` and the offline replay already rely on.
    A present-but-malformed side raises instead of silently becoming "missing",
    so a broken detector can never masquerade as an empty scene.
    """
    if value is None:
        return np.zeros((LANDMARK_COUNT, POINT_DIM), dtype=np.float32), False
    points = np.asarray(value, dtype=np.float32)
    if points.shape != (LANDMARK_COUNT, POINT_DIM):
        raise ValueError(
            f"landmarks must have shape ({LANDMARK_COUNT}, {POINT_DIM}), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("landmarks must be finite; pass None for a missing hand")
    return points, True


class CaptureH5Writer:
    """Append raw ``next_frame()`` dicts to one capture H5; call ``close()``.

    Counters and attributes are finalised in ``close()``; ``summary()`` may be
    called at any time and reports what has been written so far.
    """

    def __init__(
        self,
        path,
        *,
        camera_index: int = 0,
        image_width: int = 0,
        image_height: int = 0,
        fps: int = 0,
        model_asset_path: str = "",
        source: str = "mediapipe",
        attributes: dict | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._hands = {}
        self._valid = {}
        for side in HAND_SIDES:
            self._hands[side] = self._file.create_dataset(
                CAPTURE_SIDE_KEYS[side], shape=(0, LANDMARK_COUNT, POINT_DIM),
                maxshape=(None, LANDMARK_COUNT, POINT_DIM), dtype=np.float32,
                chunks=(1, LANDMARK_COUNT, POINT_DIM),
            )
            self._valid[side] = self._file.create_dataset(
                VALID_SIDE_KEYS[side], shape=(0,), maxshape=(None,), dtype=bool,
                chunks=(1024,),
            )
        self._frame_ids = self._file.create_dataset(
            "frame_ids", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        self._timestamps = self._file.create_dataset(
            "timestamps", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        self._unix_ms = self._file.create_dataset(
            "unix_ms", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        self._file.attrs.update({
            "recorder": RECORDER,
            "source": str(source),
            "source_landmark_space": SOURCE_LANDMARK_SPACE,
            "missing_hand_encoding": MISSING_HAND_ENCODING,
            "landmark_count": LANDMARK_COUNT,
            "camera_index": int(camera_index),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "fps": int(fps),
            "model_asset_path": str(model_asset_path),
            "timestamp_unit": TIMESTAMP_UNIT,
            "unix_ms_origin": "unix_epoch",
            "created_at_utc": _utc_now(),
        })
        if attributes:
            self._file.attrs.update({str(key): value for key, value in attributes.items()})
        self._frames = 0
        self._valid_frames = {side: 0 for side in HAND_SIDES}
        self._any_hand_frames = 0
        self._first_unix_ms: int | None = None
        self._last_unix_ms: int | None = None
        self._closed = False

    def append(self, frame: dict) -> int:
        """Store one frame; returns the row index it was written to.

        ``frame`` is exactly what ``MediaPipeCameraInput.next_frame()`` returns:
        ``{'left': ndarray | None, 'right': ndarray | None, 'timestamp': unix_ms,
        'metadata': {...}}``. Timestamps must increase strictly, which the camera
        input already guarantees (it bumps a repeated clock by 1 ms).
        """
        if self._closed:
            raise ValueError("capture writer is closed")
        timestamp = frame.get("timestamp")
        if timestamp is None:
            raise ValueError("frame has no timestamp")
        unix_ms = int(timestamp)
        if self._last_unix_ms is not None and unix_ms <= self._last_unix_ms:
            raise ValueError(
                f"timestamps must increase strictly: {unix_ms} <= {self._last_unix_ms}"
            )
        if self._first_unix_ms is None:
            self._first_unix_ms = unix_ms
        metadata = frame.get("metadata") or {}
        frame_id = int(metadata.get("frame_index", self._frames))
        index = self._frames
        detected = False
        for side in HAND_SIDES:
            points, valid = landmarks_or_zeros(frame.get(side))
            self._hands[side].resize(index + 1, axis=0)
            self._hands[side][index] = points
            self._valid[side].resize(index + 1, axis=0)
            self._valid[side][index] = valid
            self._valid_frames[side] += int(valid)
            detected = detected or valid
        elapsed = (unix_ms - self._first_unix_ms) / 1000.0
        for dataset, value in (
            (self._frame_ids, frame_id),
            (self._timestamps, elapsed),
            (self._unix_ms, unix_ms),
        ):
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self._frames += 1
        self._any_hand_frames += int(detected)
        self._last_unix_ms = unix_ms
        return index

    def summary(self) -> dict:
        """What has been written so far; also the ``close()`` return value."""
        duration = 0.0
        if self._first_unix_ms is not None and self._last_unix_ms is not None:
            duration = (self._last_unix_ms - self._first_unix_ms) / 1000.0
        return {
            "path": str(self.path),
            "frames": int(self._frames),
            "left_valid_frames": int(self._valid_frames["left"]),
            "right_valid_frames": int(self._valid_frames["right"]),
            "any_hand_frames": int(self._any_hand_frames),
            "duration_seconds": round(float(duration), 6),
            "measured_fps": round(self._frames / duration, 3) if duration > 0 else 0.0,
            "closed": bool(self._closed),
        }

    def close(self) -> dict:
        """Write the final attributes, close the file, and return ``summary()``."""
        if self._closed:
            return self.summary()
        summary = self.summary()
        self._file.attrs.update({
            "frames": summary["frames"],
            "left_valid_frames": summary["left_valid_frames"],
            "right_valid_frames": summary["right_valid_frames"],
            "any_hand_frames": summary["any_hand_frames"],
            "duration_seconds": summary["duration_seconds"],
            "measured_fps": summary["measured_fps"],
            "finished_at_utc": _utc_now(),
        })
        self._file.flush()
        self._file.close()
        self._closed = True
        # Re-read the counters: the attributes above had to be written while the
        # file was still open, so the returned summary is the post-close one.
        return self.summary()

    def update_attributes(self, mapping: dict) -> None:
        """Merge extra attributes before ``close()`` (e.g. observed frame size)."""
        if self._closed:
            raise ValueError("capture writer is closed")
        self._file.attrs.update({str(key): value for key, value in mapping.items()})

    def __enter__(self) -> "CaptureH5Writer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_capture_summary(path) -> dict:
    """Read back the layout a capture H5 must have (used by the verify scripts)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"capture H5 was not found: {path}")
    wanted = tuple(CAPTURE_SIDE_KEYS.values()) + tuple(VALID_SIDE_KEYS.values()) + (
        "frame_ids", "timestamps", "unix_ms",
    )
    with h5py.File(path, "r") as handle:
        shapes = {
            name: (tuple(handle[name].shape) if name in handle else None) for name in wanted
        }
        attributes = {key: handle.attrs[key] for key in handle.attrs}
    left = shapes[CAPTURE_SIDE_KEYS["left"]]
    return {
        "path": str(path),
        "frames": int(left[0]) if left else 0,
        "datasets": shapes,
        "attributes": attributes,
    }
