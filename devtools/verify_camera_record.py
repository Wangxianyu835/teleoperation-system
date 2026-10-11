"""摄像头 -> 抓取 H5 -> 对齐 -> 角度 -> 回放：Layer 1 的一键验收。

为什么要把"有没有手"和"链路通不通"分开
--------------------------------------
镜头前没人时 MediaPipe 会连续报 0 只手（本机实测：120 帧全部未检出），但这不代表
链路坏了。本脚本因此把两类结论分开打印：

    [OK  ] 录制/落盘/对齐/角度文件的结构与队友读取器兼容（与有没有手无关）
    [INFO] 本段检出手数（有手才可能接着验证重定向与回放）
    [SKIP] 整段没有手 -> 跳过回放（不是失败）

若确有手却重定向不出角度，那是真问题，判 [FAIL]。

用法::

    cd F:\\simulation_platform_cs
    set PYTHONPATH=F:\\simulation_platform_cs\\src
    E:\\python3.11.7\\python.exe devtools\\verify_camera_record.py --seconds 10

退出码：0 = 全部应过的判据都过；1 = 有判据失败（逐条打印原因）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SIDES = ("left", "right")


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/verify_camera_record.py",
        description="record from the camera, then prove the H5 chain is readable",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=0,
                        help="0 = use --seconds")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-asset-path", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "tmp_probe" / "camera_record")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip the PyBullet replay even when hands were detected")
    parser.add_argument("--render-every", type=int, default=25,
                        help="frames between replay PNGs (0 = none)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from teleoperation.applications.camera_record import build_parser as record_parser
    from teleoperation.applications.camera_record import run as record_run
    from teleoperation.applications.hand_align import align_h5
    from teleoperation.applications.hand_inspect import inspect_angle_h5
    from teleoperation.data.capture_h5 import read_capture_summary
    from teleoperation.data.hand_h5 import load_twohand_h5

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_h5 = out_dir / f"capture_{stamp}.h5"
    angles_h5 = out_dir / f"angles_{stamp}.h5"
    aligned_h5 = out_dir / f"aligned_{stamp}.h5"
    report_path = out_dir / f"verify_{stamp}.json"
    failures: list[str] = []
    any_hand = 0
    report: dict = {"out_dir": str(out_dir), "steps": {}}

    log("STEP", f"1/5 record camera {args.camera_index} -> {raw_h5.name} + {angles_h5.name}")
    argv = ["--output-h5", str(raw_h5), "--angles-h5", str(angles_h5),
            "--seconds", str(args.seconds), "--camera-index", str(args.camera_index),
            "--progress-every", str(args.progress_every)]
    if args.frames > 0:
        argv += ["--frames", str(args.frames)]
    if args.model_asset_path:
        argv += ["--model-asset-path", str(args.model_asset_path)]
    record_exit = int(record_run(record_parser().parse_args(argv)))
    report["steps"]["record_exit_code"] = record_exit
    if record_exit != 0 or not raw_h5.is_file():
        log("FAIL", f"camera record exit={record_exit} file_exists={raw_h5.is_file()}")
        failures.append("record")
    else:
        log("OK", f"record wrote {raw_h5.name}")

    if not failures:
        log("STEP", "2/5 read the capture back through the capture reader")
        capture = read_capture_summary(raw_h5)
        frames = int(capture["frames"])
        attributes = capture["attributes"]
        problems = []
        for side in SIDES:
            shape = capture["datasets"][f"{side}_hand_keypoints"]
            if shape != (frames, 21, 3):
                problems.append(f"{side}_hand_keypoints={shape}")
            if capture["datasets"][f"{side}_valid"] != (frames,):
                problems.append(f"{side}_valid={capture['datasets'][f'{side}_valid']}")
        for name, expected in (("timestamp_unit", "relative_seconds"),
                               ("missing_hand_encoding", "zeros"),
                               ("source_landmark_space", "mediapipe_normalized")):
            if attributes.get(name) != expected:
                problems.append(f"attrs.{name}={attributes.get(name)!r}")
        if int(attributes.get("frames", -1)) != frames:
            problems.append(f"attrs.frames={attributes.get('frames')!r} != {frames}")
        if frames < 1:
            problems.append("frames<1")
        any_hand = int(attributes.get("any_hand_frames", 0))
        report["steps"]["capture"] = {
            "frames": frames,
            "attributes": {key: str(value) for key, value in attributes.items()},
        }
        if problems:
            log("FAIL", "capture layout: " + "; ".join(problems))
            failures.append("capture_layout")
        else:
            log("OK", f"capture layout is (T,21,3)+valid+frame_ids+timestamps+unix_ms, "
                      f"frames={frames}")
        log("INFO", f"hand detection: left={attributes.get('left_valid_frames')} "
                    f"right={attributes.get('right_valid_frames')} any_hand={any_hand}")

    if not failures:
        log("STEP", "3/5 hand align (the teammate reader on our file)")
        try:
            align_report = align_h5(raw_h5, aligned_h5)
            frame_ids, timestamps, hands = load_twohand_h5(aligned_h5)
            ok = (int(align_report["frames"]) == frames
                  and hands["left"].shape == (frames, 25, 3)
                  and hands["right"].shape == (frames, 25, 3)
                  and frame_ids.shape == (frames,))
            report["steps"]["align"] = {
                "frames": int(align_report["frames"]),
                "left_zero_source_frames": int(align_report["sides"]["left"]["zero_source_frames"]),
                "right_zero_source_frames": int(align_report["sides"]["right"]["zero_source_frames"]),
            }
            if ok:
                log("OK", f"align -> {aligned_h5.name} (T,25,3), right "
                          f"zero_source_frames={align_report['sides']['right']['zero_source_frames']}")
            else:
                log("FAIL", f"align output has an unexpected shape: {hands['left'].shape}")
                failures.append("align")
        except Exception as error:  # noqa: BLE001 - report instead of raising
            log("FAIL", f"align: {type(error).__name__}: {error}")
            failures.append("align")

    if not failures:
        log("STEP", "4/5 hand inspect on the angle H5")
        try:
            inspected = inspect_angle_h5(angles_h5)
            report["steps"]["angles"] = {
                "frames": int(inspected["frames"]),
                "left_valid": int(inspected["left"]["valid"]),
                "right_valid": int(inspected["right"]["valid"]),
                "backend": str(inspected["attrs"].get("backend")),
            }
            if int(inspected["frames"]) != frames:
                log("FAIL", f"angle H5 has {inspected['frames']} frames, capture has {frames}")
                failures.append("angles")
            else:
                log("OK", f"angle H5 frames={inspected['frames']} "
                          f"valid left={inspected['left']['valid']} "
                          f"right={inspected['right']['valid']} "
                          f"backend={inspected['attrs'].get('backend')}")
        except Exception as error:  # noqa: BLE001 - report instead of raising
            log("FAIL", f"hand inspect: {type(error).__name__}: {error}")
            failures.append("angles")

    if failures:
        log("SKIP", "5/5 replay skipped: an earlier step failed")
    elif args.no_replay or any_hand == 0:
        log("SKIP", "5/5 replay: --no-replay or no hand was detected in this recording "
                    "(put a hand in front of the camera to exercise retargeting)")
    else:
        log("STEP", "5/5 replay the recording through the L21 geometric backend (PyBullet)")
        replay_out = out_dir / f"replay_{stamp}"
        command = [sys.executable, str(ROOT / "devtools" / "replay_capture_to_l21.py"),
                   "--input-h5", str(raw_h5), "--out-dir", str(replay_out),
                   "--render-every", str(args.render_every),
                   "--report", str(replay_out / "report.json")]
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True,
                                   env={**os.environ, "PYTHONUTF8": "1"})
        report["steps"]["replay"] = {
            "exit_code": int(completed.returncode), "out_dir": str(replay_out),
        }
        if completed.returncode == 0:
            log("OK", f"replay exit 0, artifacts in {replay_out}")
        else:
            tail = (completed.stdout or completed.stderr or "").strip().splitlines()
            log("FAIL", f"replay exit {completed.returncode}: "
                        f"{tail[-1] if tail else 'no output'}")
            failures.append("replay")

    report["failures"] = failures
    report["exit_code"] = 1 if failures else 0
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={report_path}")
    if failures:
        log("FAIL", f"{len(failures)} step(s) failed: {', '.join(failures)}")
        return 1
    log("OK", "camera -> H5 -> align -> angles chain verified")
    if any_hand == 0:
        log("WARN", "no hand was in frame, so this run proves the file chain only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
