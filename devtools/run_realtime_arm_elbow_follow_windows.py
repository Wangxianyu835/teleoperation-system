"""PyCharm 一键入口（2026-10-15 现场版）：摄像头窗口 + PyBullet 窗口，**肘跟着你**。

它和你上次存的那个入口的关系
----------------------------
``devtools/run_realtime_arm_windows.py`` 是通用入口（手臂默认口径，肘不跟手）。
本文件是**同一份东西的固化档**：在把参数交给它之前钉上
``--arm-elbow-follow operator``（就是 2026-10-15 你看过说"还不错"的那一版）。
环境准备 / 体检 / 两个窗口的位置与置顶 / 自动写报告 / 退出码，全部沿用上一层，
这里一行都不复制 —— 想改行为改上面那个文件，这个文件只负责"钉住这一档"。

PyCharm 怎么配（和你上次那个一样，只换 Script path）
--------------------------------------------------
    Script path        = F:\\simulation_platform_cs\\devtools\\run_realtime_arm_elbow_follow_windows.py
    Working directory  = F:\\simulation_platform_cs
    Python interpreter = E:\\python3.11.7\\python.exe   <- 必须是一个装齐 cv2/mediapipe/pybullet 的解释器
    Parameters         = （留空 = 这一版；想左右对照就填 --off）

命令行等价写法::

    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_elbow_follow_windows.py
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_elbow_follow_windows.py --check      # 只体检
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_elbow_follow_windows.py --off        # 对照档：肘不跟手
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_elbow_follow_windows.py --camera-index 1

- ``--off`` 是本文件自己的糖：等于 ``--arm-elbow-follow off``（历史行为），其它一律不变，
  方便你在两个档之间来回看。
- 你如果已经自己写了 ``--arm-elbow-follow xxx``，本文件不覆盖你的选择。
- 其余开关（``--robot`` / ``--side`` / ``--arm-trace`` / ``--print-every`` ...）原样透传给同一个 parser。
  现场想「感觉一下旧行为」：``--arm-slew-pair-floor 0``（关掉压扁护栏）；想让护栏少介入：
  ``--arm-slew-pair-floor 0.4``。默认 0.6，见 ``docs/multi-robot-arm.md`` 8.18。
- **方向口径搬谁的方向是逐台的**（2026-10-15，8.19）：``--arm-posture-frame`` 默认 ``auto`` =
  gr1_t2 用相对档（你的旋转套在它自己的中性方向上）、h1_2 / g1 用绝对档 —— 这就是现场
  「GR1-T2 的手臂怎么拐就怎么拐」的修复（同一份 200 帧回放：它左臂肘任务残差中位
  197.5 -> 25.6 mm，另两台一个数都不动）。想整体对照：``--arm-posture-frame mapped``（旧档）
  或 ``--arm-posture-frame delta``。启动行会打印 ``posture frame = auto (h1_2:mapped, ...)``。

两个窗口：摄像头预览（默认镜像，标题 ``camera: hands + arms (q/Esc quits)``，按 q/Esc 退出）
+ PyBullet（H1-2 / GR1-T2 / G1 三台并排、各自原装手）。

**报告**：自动写到 ``outputs\\tmp_realtime\\arm_run_<时间戳>.json``，里面有
``arm.robots.<robot>.elbow_posture``（本机量到的肘关节名 / 斜率 / 加载夹角）、
``bridge.elbow_bend_deg``（你这一帧的肘弯）。**按 q/Esc 退出才会落盘**：
直接关 PyBullet 窗口 = 进程被就地结束，报告不写（2026-10-15 实测，那次报告确实没生成）。

退出码沿用上一层入口：0 = 有输出；1 = 全程没数据；3 = 解释器 / 依赖 / 模型不对。
``--check`` 的退出码：0 = 环境都对（相机读不到只 WARN）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import run_realtime_arm_windows as entry      # 同目录：环境 / 体检 / 窗口 / 报告全复用

FLAG = "--arm-elbow-follow"
THIS_BUILD = "operator"        # 2026-10-15 现场版钉的这一档
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
    # 免得卡在 MediaPipe landmarker.close()（本机实测约 42 s）上。
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
