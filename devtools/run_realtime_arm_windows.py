"""Windows launcher for camera hand and arm simulation.

Delegates to simulation.realtime_teleop, reusing environment and window helpers
from run_realtime_windows. Camera body input uses the SMPLer-X worker; see
docs/smplerx-arm.md for assets and runtime configuration.
Use --check for environment checks and --arm-trace for arm target diagnostics."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import run_realtime_windows as base      # 同目录：环境准备 / 体检 / 窗口位置检查都复用它

ROOT = base.ROOT
REPORT_DIR = ROOT / "outputs" / "tmp_realtime"
POSE_ASSET = ROOT / "datasets" / "pose_landmarker_lite.task"


def check_arm_entry() -> int:
    """检查机械臂入口导入及启动器资源状态。"""
    try:
        from teleoperation.simulation.realtime_teleop import build_parser  # noqa: F401
    except Exception as error:  # noqa: BLE001 - 体检要把原因原样报出来
        base.log("FAIL", f"import teleoperation.simulation.realtime_teleop: "
                         f"{type(error).__name__}: {error}")
        return 3
    base.log("OK", "import teleoperation.simulation.realtime_teleop")
    if POSE_ASSET.is_file():
        base.log("OK", f"pose model  : {POSE_ASSET} ({POSE_ASSET.stat().st_size} bytes)")
    else:
        base.log("WARN", f"pose model missing ({POSE_ASSET}): the arms stay held; download "
                         f"pose_landmarker_lite.task into datasets/ for hands + arms")
    base.log("OK", "check finished; ready to open the two windows (hands + arms)")
    return 0


def main(argv) -> int:
    check = "--check" in list(argv)
    argv = [item for item in argv if item != "--check"]
    base.prepare_environment()
    base.redirect_caches()
    handed_over = base.relaunch_with_canonical_python((["--check"] if check else []) + list(argv))
    if handed_over is not None:
        return handed_over
    if check:
        status = base.run_check(want_window=base.wants_window(list(argv)))
        return status if status != 0 else check_arm_entry()

    from teleoperation.simulation.arm_preview import ARM_PREVIEW_WINDOW as PREVIEW_WINDOW
    from teleoperation.simulation.realtime_teleop import build_parser
    from teleoperation.simulation.realtime_teleop import main as entry_main

    base.log("INFO", f"interpreter : {sys.executable}")
    base.log("INFO", f"repo root   : {ROOT}")
    base.log("INFO", f"entry       : teleoperation.simulation.realtime_teleop")
    gui_ok, gui_detail = base.gui_support()
    if gui_ok:
        base.log("OK", f"cv2 window support: {gui_detail}")
    else:
        base.log("WARN", f"cv2 window support: {gui_detail}")
        base.log("INFO", "PyBullet 窗口照常显示，但摄像头窗口不会出现：用 E: 的解释器重跑")
    if not POSE_ASSET.is_file():
        base.log("WARN", f"pose model missing ({POSE_ASSET}): only the hands will move")

    extra = list(argv)
    report: "Path | None" = None
    if not any(item == "--report" or item.startswith("--report=") for item in extra):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORT_DIR / f"arm_run_{time.strftime('%Y%m%d_%H%M%S')}.json"
        extra += ["--report", str(report)]
        base.log("INFO", f"report -> {report}")

    args = build_parser().parse_args(extra)
    base.log("INFO", "two windows: camera preview + PyBullet; press q/Esc in the camera "
                     "window (or close PyBullet) to stop")
    if getattr(args, "preview_window_pos", None) is not None:
        base.log("INFO", f"the camera window is moved to {args.preview_window_pos} and kept "
                         f"on top; look for the window titled {PREVIEW_WINDOW!r}")
    if args.arm_calibrate_mode == "hold":
        follow = ("the arms hold their working point until then "
                  "(--arm-calibrate-mode hold)")
    else:
        follow = (f"the arms already follow you from the first frame that sees your arm "
                  f"(--arm-calibrate-mode {args.arm_calibrate_mode}) while that neutral "
                  f"builds up")
    base.log("INFO", f"arms follow {', '.join(args.robot) if args.robot else 'all three robots'}"
                     f"; keep a neutral arm pose for about {args.arm_calibrate_frames} frames "
                     f"per side: {follow}")
    base.log("INFO", "add --arm-trace to print how far your wrist moved on each axis "
                     "(use it once to confirm the left/right/forward directions)")
    status = int(entry_main(args))
    if report is not None:
        base.summarise_camera_window(report, want_window=base.wants_window(list(argv)))
    return status


if __name__ == "__main__":
    # 与真实入口一致：先跑完 main()（统计/报告都写完），flush 之后直接 os._exit，
    # 避免退出时等待手部 landmarker 关闭。
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
