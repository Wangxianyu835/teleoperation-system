"""PyCharm 一键入口：摄像头窗口 + PyBullet 窗口，**手和手臂一起动**。

它和 ``devtools/run_realtime_windows.py`` 的关系
----------------------------------------------
那个脚本启动基线入口 ``applications/realtime_hand_sim``（只驱动手，手臂被 hold 住）。
本脚本启动 ``simulation/realtime_teleop``：同一帧同时驱动手（18 维）+ 手臂（相对重定向）。

两者需要的前置条件完全一样（E: 解释器、``PYTHONPATH=src``、工作目录=仓库根、
缓存不落 C 盘），所以这里直接 import 那个脚本的工具函数，只换被启动的入口模块，
另外补一项它不做的检查：pose 模型（手臂要用它）。

PyCharm 怎么配
--------------
    Script path        = F:\\simulation_platform_cs\\devtools\\run_realtime_arm_windows.py
    Working directory  = F:\\simulation_platform_cs
    Python interpreter = E:\\python3.11.7\\python.exe   <- 选错也能起来：本文件会自己在 E: 上重跑

命令行等价写法::

    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --check
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --arm-trace
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --robot h1_2 --side right

退出码：0 = 至少一侧有输出（手角度或手臂目标）；1 = 全程没有数据；3 = 解释器/依赖/模型不对。
"""
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
    """基线体检不做的两件事：新入口能不能 import、pose 模型在不在。"""
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
    # 免得卡在 MediaPipe landmarker.close()（本机实测约 42 s）上。
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
