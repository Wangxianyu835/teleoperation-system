"""Windows arm launcher with operator elbow-bend references enabled.

Adds --arm-elbow-follow operator unless explicitly overridden. --off selects
the off mode. All other options pass through run_realtime_arm_windows, which
owns environment setup, preview, simulation and reports.
Mapping and wrist configuration are documented in docs/multi-robot-arm.md."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import run_realtime_arm_windows as entry      # 同目录：环境 / 体检 / 窗口 / 报告全复用

FLAG = "--arm-elbow-follow"
THIS_BUILD = "operator"        # 默认启用操作者肘弯参考
OFF_SWITCH = "--off"           # 本文件的糖，等价 --arm-elbow-follow off


def apply_pin(argv: list[str]) -> list[str]:
    """把这一版的口径钉进参数表；``--off`` 与显式 ``--arm-elbow-follow`` 优先。"""
    argv = list(argv)
    wanted = "off" if OFF_SWITCH in argv else THIS_BUILD
    argv = [item for item in argv if item != OFF_SWITCH]
    if any(item == FLAG or item.startswith(FLAG + "=") for item in argv):
        return argv                     # 你已经自己写了这一项，不插手
    return argv + [FLAG, wanted]


def pinned_value(argv: list[str]) -> str:
    """参数表里最终生效的那一档（argparse 的规矩：后面的覆盖前面的）。"""
    value = THIS_BUILD
    for index, item in enumerate(argv):
        if item == FLAG and index + 1 < len(argv):
            value = argv[index + 1]
        elif item.startswith(FLAG + "="):
            value = item.split("=", 1)[1]
    return value


def hand_over(argv: list[str]) -> "int | None":
    """解释器不对时，用 E: 的解释器重跑**本文件**（返回子进程退出码；None = 继续本进程）。

    为什么不用上一层那个同名工具::

        run_realtime_windows.relaunch_with_canonical_python() 重跑的是它自己那个文件
        （``run_realtime_windows.py`` = 只驱动手部的入口），于是 ``--arm-*`` 会被喂给
        手部 parser 直接报 "unrecognized arguments"。本文件重跑自己，把这一版完整带过去。

    判据与上一层一致：缺 cv2/mediapipe/pybullet，或者 cv2 打不开窗口（headless wheel）。
    """
    base = entry.base
    if os.environ.get(base.REEXEC_FLAG) == "1":
        return None
    missing = base.missing_modules()
    problem = f"lacks: {', '.join(missing)}" if missing else ""
    if not problem and base.wants_window(argv):
        ok, detail = base.gui_support()
        if not ok:
            problem = f"cannot open a GUI window: {detail}"
    if not problem:
        return None
    base.log("WARN", f"current interpreter {sys.executable} {problem}")
    canonical = base.CANONICAL_PYTHON
    if not canonical.is_file():
        base.log("FAIL", f"canonical interpreter not found: {canonical}")
        return 3
    if Path(sys.executable).resolve() == canonical.resolve():
        base.log("FAIL", f"{canonical} itself cannot do it: {problem}")
        return 3
    env = dict(os.environ)
    env[base.REEXEC_FLAG] = "1"
    base.log("OK", f"re-running {Path(__file__).name} with {canonical}")
    completed = subprocess.run(
        [str(canonical), str(Path(__file__).resolve()), *argv],
        cwd=str(base.ROOT), env=env)
    return int(completed.returncode)


def main(argv: list[str]) -> int:
    argv = apply_pin(argv)
    entry.base.log("INFO", f"this build pins {FLAG} {pinned_value(argv)} "
                           f"(add {OFF_SWITCH} to compare against the historical behaviour)")
    handed_over = hand_over(argv)
    if handed_over is not None:
        return handed_over
    return entry.main(argv)


if __name__ == "__main__":
    # 与上一层入口一致：先跑完 main()（统计 / 报告都写完），flush 之后直接 os._exit，
    # 避免退出时等待手部 landmarker 关闭。
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
