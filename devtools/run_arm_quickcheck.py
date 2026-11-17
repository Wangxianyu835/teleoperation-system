"""一键看「现在这份代码在窗口里是什么效果」：跑现场档 + 把关键读数打成一屏。

为什么另起一个文件（2026-11-17 现场要求）
-----------------------------------------
``devtools/run_realtime_arm_windows.py`` 是**现场入口，不改**。它工作区里曾经多过一段
``WRIST_PRESET``（``--arm-orientation on`` + ``--arm-orientation-scope arm`` +
``--arm-ik-mode pose``），按要求已 ``git checkout --`` 回退 —— 那个文件现在就是 HEAD 的样子，
窗口默认回到 ``--arm-orientation off``，也就是「手腕不动」。

本文件就是**不碰它**的替代品：

* 同样的两个窗口（摄像头 + PyBullet 三台）同一步序：环境准备、体检、窗口摆放都复用它里面的
  公开函数（只 import，不写）；
* 现场档参数在这里**显式钉死**（``FIELD_FLAGS``）；你自己在命令行写的同名 ``--arm-*``
  优先，只补你没写的那几项；
* 跑完读 ``--report`` 那份 JSON，把最近几节一直在追的读数打成一屏：**`operator_symmetry`**
  （左右到底是不是镜像）、**掌面基**（是不是迟到、偏多少）、**标定**（段长 / 可信度）、
  **三台残差 + 姿态误差**。

用法::

    E:\\python3.11.7\\python.exe devtools\\run_arm_quickcheck.py             # 900 帧，跑完自动收工
    E:\\python3.11.7\\python.exe devtools\\run_arm_quickcheck.py --frames 0  # 一直跑，窗口里按 q 停
    E:\\python3.11.7\\python.exe devtools\\run_arm_quickcheck.py --check     # 只体检，不开窗口
    E:\\python3.11.7\\python.exe devtools\\run_arm_quickcheck.py ^
        --arm-orientation-scope wrist                                      # 只转手腕，大臂不动

PyCharm 里也一样：Script path 指本文件，Working directory = 仓库根，Parameters 留空。

最后那一屏怎么读（配合 ``docs/multi-robot-arm.md`` §8.24）::

    operator symmetry: median  < 45 度 -> 你摆的是镜像，桥映出来的也是镜像；屏幕若还不对，
                                        问题在**下游**（残差 / 限位 / 掌面基 / 遮挡）
                       median >= 45 度 -> **输入本身同向**，问题在输入侧
                                        （看 counts.fallback、arm_quality、mediapipe 落点）
    palm basis       : ``late @frame N, X deg`` —— 那个 X 就是这条臂姿态目标带的固定偏置，
                       X 大（几十度以上）一定先排它，别再去怀疑镜像逻辑
    calibration      : ``segment scales OFF`` / ``arm reach below 0.30 m`` = 站得太近或整条手臂
                       没进画面，这一场按段的配方整场没生效

退出码沿用入口约定：0 = 至少一侧有输出；1 = 全程没有数据；3 = 解释器/依赖/模型不对。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_realtime_arm_windows as arm      # 同目录：只读复用窗口摆放 / pose 模型 / 报告目录
import run_realtime_windows as base        # 同目录：环境准备 / 体检 / 日志

ROOT = base.ROOT
POSE_ASSET = arm.POSE_ASSET
REPORT_DIR = arm.REPORT_DIR
#: 不给 ``--frames`` 时跑多少帧就自动收工（约 70 s，够看一轮左右摆动）。
DEFAULT_FRAMES = 900
#: 现场档：**显式**钉在这里，不依赖任何启动器预设（见模块说明）。
FIELD_FLAGS = (("--arm-elbow-follow", "operator"),
               ("--arm-orientation", "on"),
               ("--arm-orientation-scope", "wrist"),
               ("--arm-orientation-mode", "hand_base"),
               ("--arm-ik-mode", "pose"))


def with_field_flags(argv: list[str]) -> list[str]:
    """把现场档补进参数表：只管你**没写**的那几项，写了就以你的为准。"""
    cleaned = list(argv)
    for flag, value in FIELD_FLAGS:
        if any(item == flag or item.startswith(flag + "=") for item in cleaned):
            continue
        cleaned += [flag, value]
    return cleaned


def has_flag(argv: list[str], flag: str) -> bool:
    """``--frames`` / ``--frames=900`` 两种写法都算写了。"""
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def hand_over(argv: list[str]) -> "int | None":
    """解释器不对（缺 pybullet/cv2/mediapipe，或 cv2 开不了窗口）时用 E: 的解释器重跑**本文件**。

    为什么不直接用 ``base.relaunch_with_canonical_python()``：它重跑的是 ``run_realtime_windows.py``
    （只驱动手部那个入口），手臂这一整套 ``--arm-*`` 会被喂给手部 parser，直接报
    ``unrecognized arguments``。这里重跑本文件，参数原样带过去。
    """
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


def _num(value, digits: int = 1) -> str:
    """报告里的数字格式化：``None`` 打成 ``-``，别让摘要自己抛异常。"""
    if isinstance(value, bool) or value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _side_pair(values: dict, unit: str) -> str:
    """``{'h1_2': {'left': .., 'right': ..}}`` -> ``h1_2 10.8/150.6, ...``（左/右）。"""
    parts = []
    for robot, sides in (values or {}).items():
        sides = sides or {}
        parts.append(f"{robot} {_num(sides.get('left'))}/{_num(sides.get('right'))}")
    return f"{', '.join(parts)} {unit}"


def digest(report: "Path | None") -> None:
    """读 ``--report`` 那份 JSON，把关键读数打成一屏；读不到就安静跳过，不许把主链路带翻。"""
    if report is None or not Path(report).is_file():
        print(f"quickcheck: no report to read ({report})")
        return
    try:
        data = json.loads(Path(report).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - 摘要坏了也不能让跑完的这一场白跑
        print(f"quickcheck: cannot read {report}: {type(error).__name__}: {error}")
        return
    bridge = data.get("bridge") or {}
    arm_block = data.get("arm") or {}
    counts = bridge.get("counts") or {}
    neutral = bridge.get("neutral") or {}
    symmetry = bridge.get("operator_symmetry") or {}
    line = "-" * 78
    print()
    print(line)
    print(f"quickcheck readings: {report}")
    print(line)
    print(f"  frames        : {data.get('processed_frames')} at {_num(data.get('fps'))} fps "
          f"({_num(data.get('elapsed_seconds'))} s)")
    mode = bridge.get("orientation_mode", "?")
    scopes = sorted({str(robot.get("orientation_scope", "?"))
                     for robot in (arm_block.get("robots") or {}).values()
                     if isinstance(robot, dict)})
    print(f"  wrist mode    : orientation={mode}, scope={'/'.join(scopes) or '?'}"
          + ("" if mode == "hand_base" and scopes == ["wrist"]
             else "  [WARN] not the wrist-first field mode"))

    if symmetry:
        print(f"  operator symm : median {_num(symmetry.get('median_deg'), 2)} deg "
              f"(p90 {_num(symmetry.get('p90_deg'), 2)}, max {_num(symmetry.get('max_deg'), 2)}), "
              f"{symmetry.get('same_side_frames')}/{symmetry.get('frames')} frames same-side")
        if symmetry.get("note"):
            print(f"                  {symmetry['note']}")
    else:
        print("  operator symm : not in this report (older build)")

    for side in ("left", "right"):
        info = neutral.get(side) or {}
        source = info.get("palm_reference_source") or (
            "absent" if info.get("orientation_palm_recorded") is False else "?")
        detail = ""
        if source == "late":
            detail = (f" @frame {info.get('palm_reference_frame')}, "
                      f"{_num(info.get('palm_reference_pose_deg'))} deg from the calibration pose")
        print(f"  palm basis {side:<5}: {source}{detail} "
              f"(spread {_num(info.get('palm_spread_deg'))} deg, outliers {info.get('palm_outliers', '-')})")

    scale_notes = bridge.get("segment_scale_notes") or {}
    trust = bridge.get("arm_length_trust") or {}
    for side in ("left", "right"):
        block = trust.get(side) or {}
        note = scale_notes.get(side)
        if note:
            print(f"  calibration   : {side}: segment scales OFF - {note}")
        elif block.get("trusted") is False:
            print(f"  calibration   : {side}: {block.get('reason') or 'arm length untrusted'}")
    if bridge.get("segment_asymmetry_note"):
        print(f"  calibration   : {bridge['segment_asymmetry_note']}")

    if arm_block.get("residual_mm"):
        print(f"  wrist residual: {_side_pair(arm_block['residual_mm'], 'mm')} (left/right)")
    if arm_block.get("orientation_error_deg"):
        print(f"  orientation   : {_side_pair(arm_block['orientation_error_deg'], 'deg')} (left/right)")
    if arm_block.get("hand_orientation_error_deg"):
        print(f"  palm/hand axes: "
              f"{_side_pair(arm_block['hand_orientation_error_deg'], 'deg')} (left/right)")
    gap = (data.get("arm_forearm_gap_mm") or {}).get("min_mm") or {}
    if gap:
        print("  forearm gap   : "
              + ", ".join(f"{robot} {_num(value, 0)} mm" for robot, value in gap.items())
              + " (worst)")
    if counts:
        wanted = ("palm_late", "orientation_clamped", "fallback", "gain_guarded")
        print("  counts        : " + " | ".join(
            f"{key} left={((counts.get(key) or {}).get('left'))}"
            f" right={((counts.get(key) or {}).get('right'))}" for key in wanted))

    median = symmetry.get("median_deg")
    if isinstance(median, (int, float)) and not isinstance(median, bool):
        if median >= 45.0:
            print("  verdict       : input is NOT mirrored -> the problem is upstream "
                  "(operator / mediapipe), not the bridge")
        else:
            print("  verdict       : input is mirrored -> if the window still looks wrong, "
                  "look downstream (palm basis / residual / limits)")
    for side in ("left", "right"):
        pose = (neutral.get(side) or {}).get("palm_reference_pose_deg")
        if isinstance(pose, (int, float)) and not isinstance(pose, bool) and abs(pose) >= 60.0:
            print(f"  verdict       : {side} palm basis was learned {pose:.1f} deg away from the "
                  f"calibration pose -> that arm carries a fixed offset for the rest of the run")
    print(line)


def main(argv) -> int:
    argv = with_field_flags(list(argv))
    check = "--check" in list(argv)
    argv = [item for item in argv if item != "--check"]
    if any(item in ("-h", "--help") for item in argv):
        base.log("INFO", "this launcher always adds: "
                         + ", ".join(f"{flag} {value}" for flag, value in FIELD_FLAGS)
                         + f"; also --frames {DEFAULT_FRAMES} and its own --report when you "
                           f"do not give one (your own --arm-* / --frames / --report win)")
    base.prepare_environment()
    base.redirect_caches()
    handed_over = hand_over((["--check"] if check else []) + list(argv))
    if handed_over is not None:
        return handed_over
    if check:
        status = base.run_check(want_window=base.wants_window(list(argv)))
        return status if status != 0 else arm.check_arm_entry()

    from teleoperation.simulation.realtime_teleop import build_parser
    from teleoperation.simulation.realtime_teleop import main as entry_main

    extra = list(argv)
    if not has_flag(extra, "--frames"):
        extra += ["--frames", str(DEFAULT_FRAMES)]
        base.log("INFO", f"frames      : {DEFAULT_FRAMES} (add --frames 0 to run until q/Esc)")
    if not has_flag(extra, "--report"):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        extra += ["--report", str(REPORT_DIR / f"arm_quickcheck_{time.strftime('%Y%m%d_%H%M%S')}.json")]

    args = build_parser().parse_args(extra)
    base.log("INFO", f"interpreter : {sys.executable}")
    base.log("INFO", f"repo root   : {ROOT}")
    base.log("INFO", "fixed flags : " + ", ".join(f"{flag} {value}" for flag, value in FIELD_FLAGS))
    if not POSE_ASSET.is_file():
        base.log("WARN", f"pose model missing ({POSE_ASSET}): the arms stay held")
    base.log("INFO", "two windows: camera preview + PyBullet; q/Esc in the camera window stops it")
    arm.arrange_two_windows(args)
    status = int(entry_main(args))
    digest(getattr(args, "report", None))
    return status


if __name__ == "__main__":
    # 与现场入口一致：统计/报告都写完之后直接 os._exit，免得卡在 MediaPipe 的 close() 上。
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
