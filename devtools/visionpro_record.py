"""Vision Pro 离线采集：把手部与手臂数据落成一个可回放的 H5（不需要 Mac）。

为什么有这个入口
----------------
团队契约里 Vision Pro 是一个**输入源**（``inputs/visionpro.py`` 里
``source="visionpro"`` + ``geometry_encoding="transforms25"``），但仓库里
**只有实时适配器，没有录制入口**：``hand record`` 只支持摄像头，
``cli/registry.py`` 里没有 visionpro 命令。本脚本补上"Vision Pro -> H5"这一环。

它记录什么，为什么不记录什么
----------------------------
上游串流（App Store 的 Tracking Streamer + ``avp_stream``）每帧给出：

    <side>_fingers        (25,4,4)  手部 25 个关节的世界变换矩阵（米）
    <side>_wrist          (4,4)     腕部 6D 位姿
    <side>_arm            (27,4,4)  含前臂的完整骨架（25=forearmWrist, 26=forearmArm）
    head                  (4,4)     头显位姿
    <side>_pinch_distance float     拇指-食指距离（米）

**Vision Pro 不提供肩、肘**（骨架到前臂为止）。所以本文件不写任何"肩肘"数字，
也不假装团队的 ``left_arm_keypoints (3,3)``（肩/肘/腕三点，MediaPipe Pose 产物）；
腕与前臂按原始矩阵如实保存。需要肩肘时只能另配摄像头或 IMU，并单独声明来源。

产出两个层
----------
原始层（不丢信息，可复现任何派生）::

    frame_ids                  (T,)        int64
    timestamps                 (T,)        float64  Unix 秒
    unix_ms                    (T,)        int64    Unix 毫秒（原样，不重采样）
    left_hand_transforms       (T,25,4,4)  float32  原始关节矩阵
    right_hand_transforms      (T,25,4,4)  float32
    left_wrist_transform       (T,4,4)     float32
    right_wrist_transform      (T,4,4)     float32
    left_forearm_transforms    (T,2,4,4)   float32  [forearmWrist, forearmArm]
    right_forearm_transforms   (T,2,4,4)   float32
    head_transform             (T,4,4)     float32
    left_pinch_distance        (T,)        float32
    right_pinch_distance       (T,)        float32
    left_valid / right_valid   (T,)        bool

派生层（复用队友自己的换算函数，保证与流水线一致）::

    left_hand_keypoints        (T,25,3)    float32  腕部相对、米制、项目 25 点拓扑
    right_hand_keypoints       (T,25,4,4)  见下

派生层调用 ``retargeting.hand.topology.visionpro_fingers_to_points`` 与
``ensure_hand25``，不做任何仓库之外的自创坐标变换，因此与
``CanonicalHandProcessor`` 对同一帧的处理逐位一致。派生层**尚未做掌面到 L21
的旋转对齐**，所以不声明 ``coordinate_alignment``；文件名与属性都写明
"unaligned"，避免被训练链路误读（契约规定未知来源必须报错，不许猜）。

用法::

    :: 先确认串流通了、字段名与形状对（不用头显也能跑 --self-test）
    E:\\python3.11.7\\python.exe devtools\\visionpro_record.py --list-keys --ip <头显IP>
    E:\\python3.11.7\\python.exe devtools\\visionpro_record.py --self-test

    :: 正式录制：默认写到 datasets/raw/，按本地时间命名
    E:\\python3.11.7\\python.exe devtools\\visionpro_record.py --ip <头显IP> --seconds 180
    E:\\python3.11.7\\python.exe devtools\\visionpro_record.py --ip <头显IP> --frames 3000

    :: 校验写出的文件（结构、时间戳、有效帧、尺度是否像"米"）
    E:\\python3.11.7\\python.exe devtools\\visionpro_check.py --input-h5 <文件.h5>

退出码
------
0 = 至少写入 1 帧（即使整段没检出手也写文件并明确告警）；
1 = 连接失败、参数非法、写盘失败或凭据/库缺失。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import h5py

from teleoperation.retargeting.hand.topology import ensure_hand25, visionpro_fingers_to_points

SIDES = ("left", "right")
HAND_JOINTS = 25
MATRIX_SHAPE = (4, 4)
FOREARM_JOINTS = ("forearmWrist", "forearmArm")
FORECAST_ORDER = (25, 26)
RECORDER = "devtools/visionpro_record.py"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "raw"
SELF_TEST_DIR = ROOT / "outputs" / "tmp_visionpro"
TIMESTAMP_UNIT = "unix_seconds"
SOURCE = "visionpro"
# 上游给的是世界坐标；这里是本项目尚未对齐的米制腕相对坐标，先起一个如实的新名字。
DERIVED_LANDMARK_SPACE = "visionpro_world_meters"
# 摘下头显后 App 仍会持续发送同一帧（实测末尾 3300 多帧逐位完全相同）。
# 某一侧连续这么多帧与前一帧逐位相同，就判定它已冻结，这些帧一律标 valid=False。
STALE_LIMIT = 30


def local_stamp() -> str:
    """Asia/Shanghai 本地时间戳，与 datasets/raw 里既有文件名风格一致。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_matrix(value) -> np.ndarray:
    """One (4,4) float32 matrix from a stream value; missing/odd input -> NaN."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != MATRIX_SHAPE:
        return np.full(MATRIX_SHAPE, np.nan, dtype=np.float32)
    return array


def _as_matrix_stack(value, count: int) -> np.ndarray:
    """(count,4,4) float32 from a stream value; mismatched input -> NaN stack."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (count, 4, 4):
        return np.full((count, 4, 4), np.nan, dtype=np.float32)
    return array


def _field(data, key: str):
    """Read one field from either the dict-style or attribute-style payload."""
    try:
        return data[key]
    except Exception:
        pass
    side_prefix = None
    for side in SIDES:
        if key.startswith(side + "_"):
            side_prefix = side
            break
    if side_prefix is None:
        try:
            return getattr(data, key)
        except Exception:
            return None
    hand = getattr(data, side_prefix, None)
    if hand is None:
        return None
    suffix = key[len(side_prefix) + 1 :]
    try:
        return getattr(hand, suffix)
    except Exception:
        return None


def extract_frame(data) -> dict:
    """One stream payload -> the plain arrays this script persists."""
    frame = {
        "hands": {},
        "wrists": {},
        "forearms": {},
        "pinch": {},
        "valid": {},
        "spread": {},
        "head": _as_matrix(_field(data, "head")),
        "derived": {},
    }
    for side in SIDES:
        fingers = _field(data, side + "_fingers")
        if fingers is None:
            arm = _field(data, side + "_arm")
            if arm is not None:
                arm_array = np.asarray(arm, dtype=np.float32)
                fingers = arm_array[:HAND_JOINTS] if arm_array.ndim == 3 else None
        hand = _as_matrix_stack(fingers, HAND_JOINTS) if fingers is not None else np.full(
            (HAND_JOINTS, 4, 4), np.nan, dtype=np.float32
        )
        wrist = _as_matrix(_field(data, side + "_wrist"))
        arm_full = _field(data, side + "_arm")
        if arm_full is not None:
            arm_array = np.asarray(arm_full, dtype=np.float32)
            if arm_array.ndim == 3 and arm_array.shape[0] >= max(FORECAST_ORDER) + 1:
                forearm = arm_array[list(FORECAST_ORDER)].astype(np.float32)
            else:
                forearm = np.full((len(FORECAST_ORDER), 4, 4), np.nan, dtype=np.float32)
        else:
            forearm = np.full((len(FORECAST_ORDER), 4, 4), np.nan, dtype=np.float32)
        pinch = _field(data, side + "_pinch_distance")
        frame["hands"][side] = hand
        frame["wrists"][side] = wrist
        frame["forearms"][side] = forearm
        frame["pinch"][side] = np.float32(pinch) if pinch is not None else np.float32(np.nan)
        # 有效 = 矩阵全有限 **且** 关节之间真的有间距。
        # 头显在"连着但没追踪到手"时会持续发单位矩阵：那是有限值，但不是数据。
        # 真手的关节间距至少几毫米，这里用 0.1 mm 当"退化"的判据。
        finite = bool(np.isfinite(hand).all())
        spread = 0.0
        if finite:
            translations = hand[:, :3, 3]
            spread = float(np.max(np.linalg.norm(translations - translations[0], axis=1)))
        frame["valid"][side] = bool(finite and spread > 1e-4)
        frame["spread"][side] = np.float32(spread)
        frame["derived"][side] = (
            ensure_hand25(visionpro_fingers_to_points(hand, side))
            if frame["valid"][side]
            else np.zeros((HAND_JOINTS, 3), dtype=np.float32)
        )
    return frame


class VisionProH5Writer:
    """Append frames incrementally so Ctrl+C keeps everything already written."""

    def __init__(self, path: Path, *, compression: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        kwargs = {} if not compression else {"compression": compression}
        self._raw = {}
        for side in SIDES:
            self._raw[side + "_hand_transforms"] = self._file.create_dataset(
                side + "_hand_transforms", shape=(0, HAND_JOINTS, 4, 4),
                maxshape=(None, HAND_JOINTS, 4, 4), dtype=np.float32,
                chunks=(1, HAND_JOINTS, 4, 4), **kwargs)
            self._raw[side + "_forearm_transforms"] = self._file.create_dataset(
                side + "_forearm_transforms", shape=(0, len(FORECAST_ORDER), 4, 4),
                maxshape=(None, len(FORECAST_ORDER), 4, 4), dtype=np.float32,
                chunks=(1, len(FORECAST_ORDER), 4, 4), **kwargs)
            self._raw[side + "_wrist_transform"] = self._file.create_dataset(
                side + "_wrist_transform", shape=(0, 4, 4), maxshape=(None, 4, 4),
                dtype=np.float32, chunks=(1, 4, 4), **kwargs)
            self._raw[side + "_hand_keypoints"] = self._file.create_dataset(
                side + "_hand_keypoints", shape=(0, HAND_JOINTS, 3),
                maxshape=(None, HAND_JOINTS, 3), dtype=np.float32,
                chunks=(1, HAND_JOINTS, 3), **kwargs)
            self._raw[side + "_pinch_distance"] = self._file.create_dataset(
                side + "_pinch_distance", shape=(0,), maxshape=(None,), dtype=np.float32,
                chunks=(1024,), **kwargs)
            self._raw[side + "_hand_spread"] = self._file.create_dataset(
                side + "_hand_spread", shape=(0,), maxshape=(None,), dtype=np.float32,
                chunks=(1024,), **kwargs)
            self._raw[side + "_valid"] = self._file.create_dataset(
                side + "_valid", shape=(0,), maxshape=(None,), dtype=bool, chunks=(1024,))
        self._raw["head_transform"] = self._file.create_dataset(
            "head_transform", shape=(0, 4, 4), maxshape=(None, 4, 4), dtype=np.float32,
            chunks=(1, 4, 4), **kwargs)
        self._frame_ids = self._file.create_dataset(
            "frame_ids", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        self._timestamps = self._file.create_dataset(
            "timestamps", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        self._unix_ms = self._file.create_dataset(
            "unix_ms", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        self._file.attrs.update({
            "recorder": RECORDER,
            "dataset_format_version": "visionpro_hand_arm_capture_v1",
            "source": SOURCE,
            "source_landmark_space": DERIVED_LANDMARK_SPACE,
            "derived_landmark_space": DERIVED_LANDMARK_SPACE,
            "derived_alignment_status": "unaligned_pending_team_confirmation",
            "derived_origin": "wrist",
            "hand_taxonomy": "project_hand25",
            "hand_joint_count": HAND_JOINTS,
            "left_forearm_transform_order": ",".join(FOREARM_JOINTS),
            "arm_source_note": "visionpro_forearm_only_no_shoulder_no_elbow",
            "forearm_joint_note": (
                "index25=forearmWrist, index26=forearmArm;"
                "实测 26 距腕约 0.22 m，接近肘的位置，可近似当肘方向用；没有肩关节"),
            "length_unit": "m",
            "units": "m",
            "timestamp_unit": TIMESTAMP_UNIT,
            "missing_hand_policy": "zeros",
            "transform_shape": "4x4_row_major_translation_in_column_3",
            "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        self._frames = 0
        self._valid_frames = {side: 0 for side in SIDES}
        self._first_unix_ms: int | None = None
        self._last_unix_ms: int | None = None
        self._started = ""
        self._closed = False

    def append(self, frame: dict, *, unix_ms: int) -> int:
        if self._closed:
            raise ValueError("writer is closed")
        index = self._frames
        # 同一毫秒内到两帧时向上取整，保证 unix_ms 与 timestamps 严格递增。
        if self._last_unix_ms is not None and unix_ms <= self._last_unix_ms:
            unix_ms = self._last_unix_ms + 1
        for side in SIDES:
            for name, value in (
                (side + "_hand_transforms", frame["hands"][side]),
                (side + "_forearm_transforms", frame["forearms"][side]),
                (side + "_wrist_transform", frame["wrists"][side]),
                (side + "_hand_keypoints", frame["derived"][side]),
            ):
                dataset = self._raw[name]
                dataset.resize(index + 1, axis=0)
                dataset[index] = value
            for name, value in (
                (side + "_pinch_distance", frame["pinch"][side]),
                (side + "_hand_spread", frame["spread"][side]),
                (side + "_valid", frame["valid"][side]),
            ):
                dataset = self._raw[name]
                dataset.resize(index + 1, axis=0)
                dataset[index] = value
            self._valid_frames[side] += int(bool(frame["valid"][side]))
        head = self._raw["head_transform"]
        head.resize(index + 1, axis=0)
        head[index] = frame["head"]
        self._frame_ids.resize(index + 1, axis=0)
        self._frame_ids[index] = index
        if self._first_unix_ms is None:
            self._first_unix_ms = unix_ms
        elapsed = (unix_ms - self._first_unix_ms) / 1000.0
        self._timestamps.resize(index + 1, axis=0)
        self._timestamps[index] = elapsed
        self._unix_ms.resize(index + 1, axis=0)
        self._unix_ms[index] = unix_ms
        self._last_unix_ms = unix_ms
        self._frames += 1
        return index

    def summary(self) -> dict:
        duration = 0.0
        if self._first_unix_ms is not None and self._last_unix_ms is not None:
            duration = (self._last_unix_ms - self._first_unix_ms) / 1000.0
        return {
            "path": str(self.path),
            "frames": int(self._frames),
            "left_valid_frames": int(self._valid_frames["left"]),
            "right_valid_frames": int(self._valid_frames["right"]),
            "duration_seconds": round(float(duration), 6),
            "measured_fps": round(self._frames / duration, 3) if duration > 0 else 0.0,
        }

    def close(self, status: str) -> dict:
        if self._closed:
            return self.summary()
        summary = self.summary()
        self._file.attrs.update({
            "recording_status": str(status),
            "recorded_frames": summary["frames"],
            "left_detected_frames": summary["left_valid_frames"],
            "right_detected_frames": summary["right_valid_frames"],
            "duration_seconds": summary["duration_seconds"],
            "measured_fps": summary["measured_fps"],
            "recording_started_at": str(self._started),
            "recording_timezone": "Asia/Shanghai",
            "finished_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        self._file.flush()
        self._file.close()
        self._closed = True
        return summary

    started_at = property(lambda self: self._started, lambda self, value: setattr(self, "_started", value))


def synthetic_streamer(frames: int, side_shift: float = 0.35):
    """A streamer stand-in for --self-test: no headset, exact known geometry."""

    def hand_points(side: str) -> np.ndarray:
        # 一个按项目 25 点拓扑排好的人手骨架，单位米（腕到中指指尖约 0.18 m）。
        # 拇指 1-4、食指 5-9、中指 10-14、无名指 15-19、小指 20-24；
        # 每根非拇指手指的第 0 个点是掌根，第 1 个点是 MCP（掌心侧）。
        table = (
            (0.000, 0.000, 0.000),   # 0 腕
            (0.030, 0.020, 0.005),   # 1 拇指 MCP
            (0.055, 0.038, 0.014),   # 2 拇指 PIP
            (0.072, 0.050, 0.020),   # 3 拇指 DIP
            (0.085, 0.056, 0.024),   # 4 拇指尖
            (0.030, 0.030, 0.000),   # 5 食指掌根
            (0.040, 0.070, 0.000),   # 6 食指 MCP
            (0.036, 0.105, 0.000),   # 7 食指 PIP
            (0.033, 0.128, 0.000),   # 8 食指 DIP
            (0.031, 0.145, 0.000),   # 9 食指尖
            (0.010, 0.032, 0.000),   # 10 中指掌根
            (0.012, 0.075, 0.000),   # 11 中指 MCP
            (0.012, 0.112, 0.000),   # 12 中指 PIP
            (0.012, 0.138, 0.000),   # 13 中指 DIP
            (0.012, 0.156, 0.000),   # 14 中指尖
            (-0.012, 0.030, 0.000),  # 15 无名指掌根
            (-0.013, 0.070, 0.000),  # 16 无名指 MCP
            (-0.014, 0.103, 0.000),  # 17 无名指 PIP
            (-0.015, 0.126, 0.000),  # 18 无名指 DIP
            (-0.016, 0.142, 0.000),  # 19 无名指尖
            (-0.032, 0.024, 0.000),  # 20 小指掌根
            (-0.034, 0.062, 0.000),  # 21 小指 MCP
            (-0.036, 0.090, 0.000),  # 22 小指 PIP
            (-0.038, 0.108, 0.000),  # 23 小指 DIP
            (-0.040, 0.121, 0.000),  # 24 小指尖
        )
        points = np.asarray(table, dtype=np.float64)
        if side == "left":
            points[:, 0] *= -1.0
        return points

    def matrices(points: np.ndarray, side: str) -> np.ndarray:
        out = np.zeros((HAND_JOINTS, 4, 4), dtype=np.float32)
        for index, (x, y, z) in enumerate(points):
            matrix = np.eye(4, dtype=np.float32)
            if side == "right":
                matrix[0, 3], matrix[1, 3], matrix[2, 3] = -z, -x, y
            else:
                matrix[0, 3], matrix[1, 3], matrix[2, 3] = z, x, -y
            out[index] = matrix
        return out

    class _Streamer:
        def __init__(self):
            self._index = 0
            self._total = int(frames)
            self.offset = 0.0

        def get_latest(self):
            if self._index >= self._total:
                raise StopIteration
            payload = {}
            self.offset = 0.05 * np.sin(self._index / 15.0)
            for side in SIDES:
                points = hand_points(side)
                points[:, 1] += self.offset
                matrices_ = matrices(points, side)
                payload[side + "_fingers"] = matrices_
                payload[side + "_wrist"] = matrices_[0].copy()
                payload[side + "_arm"] = np.concatenate(
                    [matrices_, np.stack([matrices_[0], matrices_[0]])], axis=0)
                payload[side + "_pinch_distance"] = float(
                    0.03 + 0.005 * np.sin(self._index / 7.0))
            payload["head"] = np.eye(4, dtype=np.float32)
            self._index += 1
            return payload

        def close(self):
            return None

    return _Streamer()


def connect(ip: str | None):
    """The upstream client; a clear message beats an ImportError traceback."""
    try:
        from avp_stream import VisionProStreamer
    except ImportError as error:
        raise SystemExit(
            "缺少 avp_stream：先执行 pip install --upgrade avp_stream，"
            "并在头显上安装 App Store 的 Tracking Streamer。"
            f"（原始错误：{error}）"
        )
    if not ip:
        raise SystemExit("需要 --ip <头显上显示的地址>（跨网时用 --room-code）")
    return VisionProStreamer(ip=ip)


def run(args) -> int:
    if args.self_test:
        streamer = synthetic_streamer(args.frames or 60)
        target_fps = args.target_fps
        output = Path(args.output) if args.output else (
            SELF_TEST_DIR / f"visionpro_selftest_{local_stamp()}.h5")
    else:
        streamer = connect(args.room_code or args.ip)
        target_fps = args.target_fps
        if args.output:
            output = Path(args.output)
        else:
            directory = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
            output = directory / f"visionpro_hand_arm_{local_stamp()}.h5"
    if output.exists():
        print(f"[FAIL] 输出文件已存在，不覆盖：{output}")
        return 1

    writer = VisionProH5Writer(output, compression=args.compression or None)
    writer.started_at = datetime.now().astimezone().isoformat()
    period = 1.0 / float(target_fps) if target_fps and target_fps > 0 else 0.0
    deadline = time.time() + float(args.seconds) if args.seconds else None
    status = "frames_reached"
    identity_frames = 0
    stale_frames = {"left": 0, "right": 0}
    stale_run = {"left": 0, "right": 0}
    previous_hand = {"left": None, "right": None}
    print(f"开始录制：{output}")
    print(f"停止条件：frames={args.frames} seconds={args.seconds}（按 Ctrl+C 提前结束）")
    try:
        while True:
            if args.frames and writer.summary()["frames"] >= int(args.frames):
                break
            if deadline is not None and time.time() >= deadline:
                status = "seconds_reached"
                break
            try:
                payload = streamer.get_latest()
            except StopIteration:
                status = "stream_ended"
                break
            # 停止开关：外部只要让这个文件出现，就按正常流程收尾（兜底手段，
            # 因为 Ctrl+C 有时传不到后台进程）。
            if args.stop_file and Path(args.stop_file).exists():
                status = "stop_file"
                print(f"  检测到停止开关 {args.stop_file}，正常收尾")
                break
            frame = extract_frame(payload)
            # 冻结检测：与前一帧逐位相同就累计，超过阈值即认定该侧已停止更新。
            for side in SIDES:
                current = frame["hands"][side]
                if previous_hand[side] is not None and np.array_equal(current, previous_hand[side]):
                    stale_run[side] += 1
                else:
                    stale_run[side] = 0
                previous_hand[side] = current
                if stale_run[side] >= int(args.stale_limit):
                    frame["valid"][side] = False
                    frame["derived"][side] = np.zeros((HAND_JOINTS, 3), dtype=np.float32)
                    stale_frames[side] += 1
            unix_ms = int(time.time() * 1000.0)
            writer.append(frame, unix_ms=unix_ms)
            if not (frame["valid"]["left"] or frame["valid"]["right"]):
                identity_frames += 1
            summary = writer.summary()
            if summary["frames"] % 30 == 0:
                print(
                    f"  已写 {summary['frames']} 帧，"
                    f"左手有效 {summary['left_valid_frames']}，"
                    f"右手有效 {summary['right_valid_frames']}，"
                    f"当前关节跨度 左手 {float(frame['spread']['left']) * 100:.1f}cm / "
                    f"右手 {float(frame['spread']['right']) * 100:.1f}cm")
            # 两侧都冻结达到阈值就自动收工：避免把"摘掉头显后的重复帧"录满整段。
            if args.auto_stop_stale and min(stale_run.values()) >= int(args.auto_stop_stale):
                status = "stale_autostop"
                print(f"  两侧均冻结超过 {args.auto_stop_stale} 帧，自动停止录制")
                break
            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        status = "keyboard_quit"
    finally:
        summary = writer.close(status)

    print(f"录制结束：status={status}")
    print(
        f"  文件={summary['path']}\n"
        f"  帧数={summary['frames']} 时长={summary['duration_seconds']}s "
        f"实测fps={summary['measured_fps']}\n"
        f"  左手有效帧={summary['left_valid_frames']} "
        f"右手有效帧={summary['right_valid_frames']}")
    if summary["frames"] == 0:
        print("[FAIL] 没有写入任何帧")
        return 1
    if summary["left_valid_frames"] == 0 and summary["right_valid_frames"] == 0:
        print(
            f"[WARN] 全程没有真正的手部数据（{identity_frames} 帧是单位矩阵）："
            "文件已写出但内容无效。请戴上头显、让双手进入视野、确认 App 在前台，再重录。")
    if stale_frames["left"] or stale_frames["right"]:
        print(
            f"[WARN] 检测到冻结帧：左手 {stale_frames['left']} 帧、右手 {stale_frames['right']} 帧"
            "（与前一帧逐位完全相同，通常是摘下头显后 App 仍在发同一帧）。"
            "这些帧已标 valid=False，交付前建议按有效区间裁剪。")
    return 0


def list_keys(args) -> int:
    streamer = synthetic_streamer(1) if args.self_test else connect(args.room_code or args.ip)
    try:
        payload = streamer.get_latest()
    except StopIteration:
        print("流里没有数据")
        return 1
    print("串流字段与形状：")
    keys = sorted(payload.keys()) if hasattr(payload, "keys") else [
        name for name in dir(payload) if not name.startswith("_")]
    for key in keys:
        try:
            value = payload[key] if hasattr(payload, "__getitem__") else getattr(payload, key)
        except Exception:
            continue
        shape = getattr(value, "shape", None)
        print(f"  {key}: shape={shape} type={type(value).__name__}")
    frame = extract_frame(payload)
    for side in SIDES:
        print(
            f"  [{side}] transforms={frame['hands'][side].shape} "
            f"wrist={frame['wrists'][side].shape} "
            f"forearm={frame['forearms'][side].shape} "
            f"derived={frame['derived'][side].shape} valid={frame['valid'][side]}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python devtools/visionpro_record.py",
        description="Vision Pro 手部 + 腕/前臂离线采集（写成 H5）",
    )
    parser.add_argument("--ip", help="头显地址（同一 WiFi 下由 Tracking Streamer 显示）")
    parser.add_argument("--room-code", help="跨网模式的房间码，与 --ip 二选一")
    parser.add_argument("--output", help="完整输出路径；已存在则拒绝覆盖")
    parser.add_argument("--output-dir", help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--frames", type=int, default=0, help="写够多少帧就停；0 表示不按帧数停")
    parser.add_argument("--seconds", type=float, default=0.0, help="录制多少秒后停；0 表示不按时间停")
    parser.add_argument("--target-fps", type=float, default=30.0,
                       help="采样节奏；0 表示不做节流（默认 30）")
    parser.add_argument("--compression", choices=("gzip", "lzf"), help="H5 压缩，默认不压缩")
    parser.add_argument("--stale-limit", type=int, default=STALE_LIMIT,
                       help=f"连续多少帧逐位相同就判定冻结（默认 {STALE_LIMIT}，约 1 秒）")
    parser.add_argument("--auto-stop-stale", type=int, default=90,
                       help="两侧都冻结这么多帧就自动停止（默认 90，约 3 秒；0 表示不自动停）")
    parser.add_argument("--stop-file", help="这个文件一旦出现就正常收尾（给外部停止用）")
    parser.add_argument("--list-keys", action="store_true", help="只打印串流字段与形状，不录制")
    parser.add_argument("--self-test", action="store_true",
                       help="不用头显：用合成数据跑通整条写入链路")
    args = parser.parse_args(argv)
    if args.ip and args.room_code:
        print("[FAIL] --ip 与 --room-code 只能给一个")
        return 1
    if args.self_test:
        if not args.frames and not args.seconds:
            args.frames = 60
    elif not args.ip and not args.room_code:
        print("[FAIL] 需要 --ip 或 --room-code（或用 --self-test）")
        return 1
    if args.list_keys:
        return list_keys(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
