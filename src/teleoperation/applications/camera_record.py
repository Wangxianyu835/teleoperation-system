"""摄像头 -> 原始 21 点 H5（可选再落一份 18 维角度 H5）。

为什么有这个入口
----------------
本仓库缺"摄像头到文件"这一步：``hand realtime`` 只把识别结果打印到屏幕，
``hand align`` 与 ``--input-h5`` 回放又都要求先有一个抓取 H5（格式见
``data/capture_h5.py`` 的说明）。本入口把 ``MediaPipeCameraInput`` 的每一帧交给
``CaptureH5Writer`` 落盘，于是"录一段 -> 对齐 -> 回放"这条离线闭环第一次有了起点。

它不注册进 python -m teleoperation
---------------------------------
``cli/registry.py`` 是队友已入库的文件（约定 3：不改），所以本入口自带 argparse，
用模块方式运行：

    cd F:\\simulation_platform_cs
    set PYTHONPATH=F:\\simulation_platform_cs\\src
    E:\\python3.11.7\\python.exe -m teleoperation.apps.camera_record --seconds 10

参数与产物
----------
--output-h5   原始抓取 H5；默认 outputs/tmp_probe/camera_record/capture_<时间戳>.h5
--angles-h5   可选：18 维 L21 角度 H5，用零权重几何后端算出
              （``retargeting/hand/geometric.py``）；schema 与队友
              ``datasets/raw/my_recording_angles.h5`` 一致，``hand inspect`` 可直接读
--seconds / --frames   停止条件（默认 --seconds 10.0）；--frames 优先
--model-asset-path     hand_landmarker.task；不给则依次在仓库根与 outputs/ 下找

关于 --angles-h5 必须如实说明：它是几何后端（读骨节夹角 + 一次张开手零位标定），
不是训练出来的 PoseTransformer，也不是论文的 dex-retargeting；只能当"没有检查点
时的可用角度"，标定完成前该侧 valid=False（角度全 0）。

退出码
------
0 = 至少写入 1 帧（即使整段没有检出手：仍写文件并明确告警）；
1 = 摄像头打不开、读帧失败、写盘失败或参数非法。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from teleoperation.contracts.constants import HAND_SIDES
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.data.capture_h5 import CaptureH5Writer, TIMESTAMP_UNIT
from teleoperation.data.hand_h5 import write_hand_angles
from teleoperation.paths import DEFAULT_MEDIAPIPE_ASSET, PROJECT_ROOT
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM
from teleoperation.retargeting.hand.geometric import GeometricHandRetargeter
from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
from teleoperation.retargeting.hand.temporal import TemporalBuffer

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tmp_probe" / "camera_record"
MODEL_ASSET_CANDIDATES = (
    DEFAULT_MEDIAPIPE_ASSET,
    PROJECT_ROOT / "outputs" / "hand_landmarker.task",
)
DEFAULT_SECONDS = 10.0
ANGLES_BACKEND = "geometric"


def default_model_asset() -> Path:
    """First existing hand_landmarker.task, else a clear FileNotFoundError."""
    for candidate in MODEL_ASSET_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "hand_landmarker.task not found; looked in "
        + ", ".join(str(item) for item in MODEL_ASSET_CANDIDATES)
        + " (pass --model-asset-path)"
    )


class AngleCollector:
    """``record_frames`` hook: raw landmarks -> palm-local window -> 18D angles.

    Identical chain to ``devtools/replay_capture_to_l21.py`` (processor -> three
    frame window -> geometric backend), so a recording and an offline replay of
    that recording give the same numbers for the same frames.
    """

    def __init__(self, calibration_frames: int = 15, smoothing: float = 0.0):
        self.calibration_frames = int(calibration_frames)
        self.smoothing = float(smoothing)
        self.processor = MediaPipePalmLocalProcessor()
        self.buffer = TemporalBuffer(3)
        self.retargeter = GeometricHandRetargeter(
            calibration_frames=self.calibration_frames, smoothing=self.smoothing)
        self.frame_ids: list[int] = []
        self.timestamps: list[float] = []
        self.angles = {side: [] for side in HAND_SIDES}
        self.valid = {side: [] for side in HAND_SIDES}

    @property
    def calibration_status(self):
        return self.retargeter.calibration_status

    def __call__(self, index: int, frame: dict, frame_id: int,
                 elapsed_seconds: float) -> None:
        """Per recorded frame; ``frame`` is the raw MediaPipe frame dict."""
        hands = {side: frame.get(side) for side in HAND_SIDES}
        window = self.buffer.append(self.processor.process_frame(
            RawHandFrame(hands, elapsed_seconds, "mediapipe_approx", {"frame_index": frame_id})))
        result = None if window is None else self.retargeter.retarget(window)
        for side in HAND_SIDES:
            values = None if result is None else getattr(result, side + "_angles")
            if values is None:
                # No window yet (fewer than 3 frames) or calibration unfinished:
                # write zeros and mark the frame invalid instead of guessing.
                self.angles[side].append(np.zeros(HAND_ANGLE_DIM, dtype=np.float32))
                self.valid[side].append(False)
                continue
            array = np.asarray(values.values, dtype=np.float32)
            if array.shape != (HAND_ANGLE_DIM,) or not np.isfinite(array).all():
                raise ValueError(f"invalid {side} angles: shape={array.shape}")
            self.angles[side].append(array)
            self.valid[side].append(True)
        self.frame_ids.append(int(frame_id))
        self.timestamps.append(float(elapsed_seconds))

    def write(self, path, attributes: dict | None = None) -> dict:
        """Persist the collected angles with the teammate's angle-H5 writer."""
        if not self.frame_ids:
            raise ValueError("no frame was collected; nothing to write")
        angles = {side: np.stack(self.angles[side]) for side in HAND_SIDES}
        valid = {side: np.asarray(self.valid[side], dtype=bool) for side in HAND_SIDES}
        metadata = {
            "recorder": "teleoperation.apps.camera_record",
            "backend": ANGLES_BACKEND,
            "backend_weights": "none",
            "invalid_angle_policy": "mark_invalid",
            "output_shape": np.asarray([HAND_ANGLE_DIM]),
            "calibration_frames": self.calibration_frames,
            "smoothing": self.smoothing,
            "timestamp_unit": TIMESTAMP_UNIT,
        }
        if attributes:
            metadata.update({str(key): value for key, value in attributes.items()})
        write_hand_angles(
            path, np.asarray(self.frame_ids, dtype=np.int64),
            np.asarray(self.timestamps, dtype=np.float64), angles, valid, metadata,
        )
        return {
            "path": str(path),
            "frames": int(len(self.frame_ids)),
            "left_valid_frames": int(valid["left"].sum()),
            "right_valid_frames": int(valid["right"].sum()),
            "calibration_status": {side: bool(self.calibration_status[side])
                                   for side in HAND_SIDES},
        }


def record_frames(source, writer, *, max_frames: int = 0, max_seconds: float = 0.0,
                  progress_every: int = 30, clock=time.perf_counter,
                  printer=print, on_frame=None) -> dict:
    """Pull ``source.next_frame()`` into ``writer`` until a limit is reached.

    Stops at ``max_frames`` (if > 0) or when ``max_seconds`` elapsed (if > 0),
    whichever comes first; at least one must be positive. ``on_frame`` is called
    as ``on_frame(row, frame, frame_id, elapsed_seconds)`` after each append, so
    a caller can collect side products (``AngleCollector``) or the observed
    frame size. Caller owns both objects' lifetimes.
    """
    if max_frames <= 0 and max_seconds <= 0:
        raise ValueError("pass max_frames > 0 or max_seconds > 0")
    started = clock()
    first_unix_ms = None
    index = 0
    while True:
        if max_frames > 0 and index >= max_frames:
            break
        if max_seconds > 0 and index > 0 and clock() - started >= max_seconds:
            break
        frame = source.next_frame()
        unix_ms = int(frame["timestamp"])
        if first_unix_ms is None:
            first_unix_ms = unix_ms
        elapsed = (unix_ms - first_unix_ms) / 1000.0
        row = writer.append(frame)
        if on_frame is not None:
            metadata = frame.get("metadata") or {}
            on_frame(row, frame, int(metadata.get("frame_index", row)), elapsed)
        index += 1
        if progress_every > 0 and index % progress_every == 0:
            seconds = clock() - started
            printer(f"frame={index} t={elapsed:.2f}s loop_fps={index / max(seconds, 1e-9):.1f}")
    summary = dict(writer.summary())
    loop_seconds = clock() - started
    summary["loop_seconds"] = round(float(loop_seconds), 6)
    summary["loop_fps"] = round(index / loop_seconds, 3) if loop_seconds > 0 else 0.0
    return summary


def run(args) -> int:
    """Open the camera, record into ``--output-h5``, close everything, report."""
    from teleoperation.inputs.mediapipe import MediaPipeCameraInput

    output_h5 = (Path(args.output_h5) if args.output_h5
                 else DEFAULT_OUTPUT_DIR / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.h5")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    model_asset = (Path(args.model_asset_path) if args.model_asset_path
                   else default_model_asset())
    collector = (AngleCollector(args.calibration_frames, args.smoothing)
                 if args.angles_h5 else None)
    observed: dict = {}

    def hook(row, frame, frame_id, elapsed):
        """Record what the detector actually produced, then feed the collector."""
        metadata = frame.get("metadata") or {}
        for key in ("camera_index", "image_width", "image_height"):
            if key in metadata:
                observed[key] = int(metadata[key])
        if collector is not None:
            collector(row, frame, frame_id, elapsed)

    source = MediaPipeCameraInput(
        model_asset_path=model_asset, camera_index=args.camera_index,
        width=args.width, height=args.height, fps=args.fps,
        invalid_hand_as_missing=True,
    )
    try:
        writer = CaptureH5Writer(
            output_h5, camera_index=args.camera_index, image_width=args.width,
            image_height=args.height, fps=args.fps, model_asset_path=str(model_asset),
            attributes={
                "requested_width": int(args.width),
                "requested_height": int(args.height),
                "requested_fps": int(args.fps),
                "angles_backend": ANGLES_BACKEND,
                "cli": "teleoperation.apps.camera_record",
            },
        )
    except BaseException:
        source.close()
        raise

    interrupted, failure = False, None
    loop_stats = {"loop_seconds": 0.0, "loop_fps": 0.0}
    try:
        summary = record_frames(
            source, writer, max_frames=args.frames, max_seconds=args.seconds,
            progress_every=args.progress_every, on_frame=hook,
        )
        loop_stats = {"loop_seconds": summary["loop_seconds"],
                      "loop_fps": summary["loop_fps"]}
    except KeyboardInterrupt:
        interrupted = True
        summary = dict(writer.summary())
        print("capture interrupted by Ctrl+C; frames written so far are kept", flush=True)
    except BaseException as error:
        failure = error
        summary = dict(writer.summary())
    finally:
        writer.update_attributes(
            {f"observed_{key}": int(value) for key, value in observed.items()})
        summary = writer.close()
        source.close()
    if failure is not None:
        raise failure

    report = {
        "capture_h5": str(output_h5),
        "frames": summary["frames"],
        "duration_seconds": summary["duration_seconds"],
        "measured_fps": summary["measured_fps"],
        "loop_seconds": loop_stats["loop_seconds"],
        "loop_fps": loop_stats["loop_fps"],
        "valid_frames": {"left": summary["left_valid_frames"],
                         "right": summary["right_valid_frames"]},
        "any_hand_frames": summary["any_hand_frames"],
        "interrupted": bool(interrupted),
        "camera_index": int(observed.get("camera_index", args.camera_index)),
        "image_width": int(observed.get("image_width", args.width)),
        "image_height": int(observed.get("image_height", args.height)),
        "model_asset_path": str(model_asset),
        "angles_h5": None,
        "angles": None,
        "exit_code": 0,
    }
    print(f"capture_h5={output_h5}")
    print(f"frames={report['frames']} duration_seconds={report['duration_seconds']} "
          f"measured_fps={report['measured_fps']} loop_fps={report['loop_fps']}")
    print(f"valid_frames left={summary['left_valid_frames']} "
          f"right={summary['right_valid_frames']} any_hand_frames={summary['any_hand_frames']}")
    if summary["frames"] < 1:
        report["exit_code"] = 1
        print("[FAIL] no frame was written", flush=True)
    if args.angles_h5 and summary["frames"] > 0:
        angles_path = Path(args.angles_h5)
        angles_path.parent.mkdir(parents=True, exist_ok=True)
        angles = collector.write(angles_path, {
            "capture_h5": str(output_h5),
            "capture_frames": summary["frames"],
            "camera_index": report["camera_index"],
        })
        report["angles_h5"] = str(angles_path)
        report["angles"] = angles
        print(f"angles_h5={angles_path}")
        print(f"angles_valid_frames left={angles['left_valid_frames']} "
              f"right={angles['right_valid_frames']} calibration={angles['calibration_status']}")
    if summary["any_hand_frames"] == 0:
        print("[WARN] no hand was detected in any frame: the file is written, but there is "
              "nothing to retarget (check that the hand is inside the frame and lit)",
              flush=True)
    elif summary["left_valid_frames"] == 0 or summary["right_valid_frames"] == 0:
        print("[WARN] only one hand was detected; the other side is stored as zeros", flush=True)
    if interrupted:
        print("[WARN] interrupted recording: the frames up to the interruption are on disk",
              flush=True)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report={report_path}")
    return report["exit_code"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m teleoperation.apps.camera_record",
        description="camera -> raw 21-landmark capture H5 (optionally an 18D angle H5)",
    )
    parser.add_argument("--output-h5", type=Path, default=None,
                        help="raw capture H5; default "
                             "outputs/tmp_probe/camera_record/capture_<timestamp>.h5")
    parser.add_argument("--angles-h5", type=Path, default=None,
                        help="optional 18D angle H5 from the zero-weight geometric backend")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help=f"stop after this many seconds (default {DEFAULT_SECONDS})")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = use --seconds; takes priority when > 0)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model-asset-path", type=Path, default=None,
                        help="hand_landmarker.task; default: repo root, else outputs/")
    parser.add_argument("--calibration-frames", type=int, default=15,
                        help="open-hand frames used as the geometric zero reference "
                             "(--angles-h5 only)")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="geometric backend smoothing (--angles-h5 only)")
    parser.add_argument("--progress-every", type=int, default=30,
                        help="print a progress line every N frames (0 = silent)")
    parser.add_argument("--report", type=Path, default=None,
                        help="write this JSON summary (same numbers as stdout)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames <= 0 and args.seconds <= 0:
        parser.error("pass --frames > 0 or --seconds > 0")
    try:
        return run(args)
    except Exception as error:  # noqa: BLE001 - CLI boundary reports instead of raising
        print(f"[FAIL] camera record stopped: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
