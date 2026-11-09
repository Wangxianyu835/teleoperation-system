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
    Script path        = <仓库根>\\devtools\\run_realtime_arm_windows.py
    Working directory  = <仓库根>
    Python interpreter = E:\\python3.11.7\\python.exe   <- 选错也能起来：本文件会自己在 E: 上重跑

<仓库根> = 含 ``src\\teleoperation`` 的那一层（本文件所在目录的上一级）。本机同时存在
``F:\\simulation_platform`` 与 ``F:\\simulation_platform_cs`` 两个 worktree：按 PyCharm 里
实际打开的那个填，不要照抄另一个，否则点 Run 跑的是那边那份代码。

命令行等价写法::

    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --check
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --elbow-follow operator
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --arm-trace
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py --robot h1_2 --side right

现场档（``--elbow-follow``）
----------------------------
2026-10-15 现场看过说"还不错"的那一版是**肘跟着你弯**（入口参数
``--arm-elbow-follow operator``）。它原先单独立了一个薄封装
``devtools/run_realtime_arm_elbow_follow_windows.py``；2026-11-09 按"一个功能只住一个文件"
并回本文件并删掉那个文件：

    PyCharm Parameters = --elbow-follow operator     <- 现场档
    PyCharm Parameters = （留空）                     <- 入口默认口径（off，历史行为）

``--elbow-follow`` 只是本文件的糖，交给入口前会换成 ``--arm-elbow-follow``；你自己写了
``--arm-elbow-follow`` 就不插手（以你的为准）。

两个窗口怎么摆（本文件负责，队友文件一行未改）
----------------------------------------------
运行起来就是**两个窗口并排**，方便一边看自己一边看机器人：

    左：摄像头预览  camera: hands + arms (q/Esc quits)   手 21 点 + 肩/肘/腕折线 + 每侧数字
    右：PyBullet    H1-2 / GR1-T2 / G1 三台并排          手由 18 维角度驱动、手臂由 IK 驱动

PyBullet 的窗口比摄像头窗口晚一步出现，所以这里开一个后台线程轮询（``DeviceWin32``
类 + 标题以 ``Bullet Physics`` 开头，2026-11-09 实测），找到后把它 ``MoveWindow``
到右半边；30 s 内没找到只打一行 WARN，不影响主链路。想自己拖窗口就把环境变量
``TELEOP_WINDOW_LAYOUT=off``（PyCharm：Run -> Edit Configurations -> Environment
variables），或加 ``--no-preview`` / ``--headless``。

退出码：0 = 至少一侧有输出（手角度或手臂目标）；1 = 全程没有数据；3 = 解释器/依赖/模型不对。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import run_realtime_windows as base      # 同目录：环境准备 / 体检 / 窗口位置检查都复用它

ROOT = base.ROOT
REPORT_DIR = ROOT / "outputs" / "tmp_realtime"
POSE_ASSET = ROOT / "datasets" / "pose_landmarker_lite.task"
#: 本文件的糖 ``--elbow-follow`` 与它换成入口参数后的名字（现场档 = operator）。
ELBOW_SWITCH = "--elbow-follow"
ELBOW_FLAG = "--arm-elbow-follow"
ELBOW_BUILD = "operator"
#: PyBullet GUI 窗口的身份（2026-11-09 实测：类名 DeviceWin32，标题以 'Bullet Physics' 开头）。
PYBULLET_WINDOW_CLASS = "DeviceWin32"
PYBULLET_WINDOW_TITLE = "Bullet Physics"
#: 关掉自动摆放：想自己拖窗口时设 ``TELEOP_WINDOW_LAYOUT=off``。
LAYOUT_ENV = "TELEOP_WINDOW_LAYOUT"
#: 等 PyBullet 建窗口的上限（秒）与轮询间隔；它通常入场后 1-3 s 出现。
LAYOUT_TIMEOUT_S = 30.0
LAYOUT_POLL_S = 0.4


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


def _win32():
    """``(ctypes, wintypes)``；非 Windows / 取不到时 ``None``（摆窗口只是显示糖）。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        return ctypes, wintypes
    except Exception:  # noqa: BLE001 - 摆不了就按系统默认位置跑，主链路不受影响
        return None


def screen_size():
    """主屏像素 ``(宽, 高)``；拿不到时 ``None``（远程桌面 / 无桌面会话都可能拿不到）。"""
    api = _win32()
    if api is None:
        return None
    try:
        width = int(api[0].windll.user32.GetSystemMetrics(0))
        height = int(api[0].windll.user32.GetSystemMetrics(1))
    except Exception:  # noqa: BLE001
        return None
    return (width, height) if width > 0 and height > 0 else None


def find_pybullet_window():
    """PyBullet GUI 窗口的 HWND；还没建出来 / 找不到时 ``None``。"""
    api = _win32()
    if api is None:
        return None
    ctypes, wintypes = api
    user32 = ctypes.windll.user32
    found = []

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, name, 256)
        if name.value != PYBULLET_WINDOW_CLASS:
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        if not title.value.startswith(PYBULLET_WINDOW_TITLE):
            return True
        found.append(int(hwnd))
        return False                     # 找到了就停，不再翻别的窗口

    user32.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND,
                                          wintypes.LPARAM)(callback), 0)
    return found[0] if found else None


def window_rect(hwnd):
    """``(x, y, 宽, 高)``；取不到时 ``None``。"""
    api = _win32()
    if api is None:
        return None
    ctypes, wintypes = api
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return (int(rect.left), int(rect.top),
            int(rect.right - rect.left), int(rect.bottom - rect.top))


def move_window(hwnd, target) -> bool:
    """把窗口挪/缩到 ``target=(x, y, 宽, 高)``；Windows 拒绝时 ``False``。"""
    api = _win32()
    if api is None:
        return False
    ctypes, _wintypes = api
    x, y, width, height = (int(value) for value in target)
    return bool(ctypes.windll.user32.MoveWindow(hwnd, x, y, width, height, True))


def planned_layout(args, screen):
    """``(摄像头位置, PyBullet 目标矩形)``：左摄像头、右 PyBullet，中间留 8 px。"""
    width, height = screen
    camera = getattr(args, "preview_window_pos", None)
    capture_width = int(getattr(args, "width", 640) or 640)
    # 分界线：至少让摄像头窗口（默认 640 宽）整块放进左半边；屏幕窄时保底给仿真留 320。
    split = max(capture_width + 60, width // 2)
    split = min(split, max(width - 320, 320))
    rect = (split + 8, 40, max(width - split - 16, 320), max(height - 120, 320))
    return camera, rect


def arrange_two_windows(args):
    """把 PyBullet 窗口摆到右半边（后台线程等它出现，不阻塞主链路）。"""
    if os.environ.get(LAYOUT_ENV, "").strip().lower() in ("off", "0", "no", "false"):
        base.log("INFO", f"window layout off ({LAYOUT_ENV}=off): drag the windows yourself")
        return None
    if getattr(args, "headless", False) or getattr(args, "no_preview", False):
        return None
    screen = screen_size()
    if screen is None or _win32() is None:
        base.log("INFO", "cannot query the screen size here: both windows stay where the "
                         "system puts them (pass --preview-window-pos X,Y to move the camera one)")
        return None
    camera, target = planned_layout(args, screen)
    where = ("where the system puts it" if camera is None
             else f"at {camera[0]},{camera[1]}")
    base.log("INFO", f"two windows side by side (screen {screen[0]}x{screen[1]}): camera preview "
                     f"on the left {where}, PyBullet (three robots) on the right at "
                     f"{target[0]},{target[1]} ({target[2]}x{target[3]})")
    base.log("INFO", f"set {LAYOUT_ENV}=off to place the windows yourself")

    def worker():
        deadline = time.monotonic() + LAYOUT_TIMEOUT_S
        while time.monotonic() < deadline:
            hwnd = find_pybullet_window()
            if hwnd:
                before = window_rect(hwnd)
                if before == target:
                    base.log("OK", f"PyBullet window already at {target}")
                elif move_window(hwnd, target):
                    base.log("OK", f"PyBullet window {before} -> {target}")
                else:
                    base.log("WARN", f"cannot move the PyBullet window (stays {before})")
                return
            time.sleep(LAYOUT_POLL_S)
        base.log("WARN", f"PyBullet window not found within {LAYOUT_TIMEOUT_S:.0f} s: "
                         f"drag it next to the camera window yourself")

    thread = threading.Thread(target=worker, name="pybullet-window-layout", daemon=True)
    thread.start()
    return thread


def apply_elbow_preset(argv: list[str]) -> list[str]:
    """``--elbow-follow operator|off`` -> ``--arm-elbow-follow operator|off``。

    2026-11-09 并回来的"现场档"：原先它在 ``run_realtime_arm_elbow_follow_windows.py`` 里是
    默认值（那个文件已经不在了），现在改成显式开关，不写就是入口自己的默认口径。
    你自己写了 ``--arm-elbow-follow`` 就不插手 —— 以你的为准。
    """
    cleaned: list[str] = []
    wanted: "str | None" = None
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == ELBOW_SWITCH and index + 1 < len(argv):
            wanted = argv[index + 1]
            index += 2
            continue
        if item.startswith(ELBOW_SWITCH + "="):
            wanted = item.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(item)
        index += 1
    if wanted is None:
        return cleaned
    if any(item == ELBOW_FLAG or item.startswith(ELBOW_FLAG + "=") for item in cleaned):
        return cleaned
    return cleaned + [ELBOW_FLAG, wanted]


def elbow_follow_value(argv: list[str]) -> str:
    """启动行要打印的那一档（argparse 的规矩：后面的覆盖前面的；没有就报入口默认 off）。"""
    value = "off"
    for index, item in enumerate(argv):
        if item == ELBOW_FLAG and index + 1 < len(argv):
            value = argv[index + 1]
        elif item.startswith(ELBOW_FLAG + "="):
            value = item.split("=", 1)[1]
    return value


def hand_over(argv: list[str]) -> "int | None":
    """解释器不对（缺 pybullet/cv2/mediapipe，或 cv2 开不了窗口）时用 E: 的解释器重跑**本文件**。

    为什么不直接用 ``base.relaunch_with_canonical_python()``：它重跑的是 ``base`` 那个文件
    （``run_realtime_windows.py`` = 只驱动手部的入口），手臂这一整套 ``--arm-*`` 会被喂给
    手部 parser，直接报 ``unrecognized arguments``。这里重跑本文件，参数完整带过去
    （2026-11-09 从并掉的那个固化档搬过来）。
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


def main(argv) -> int:
    argv = apply_elbow_preset(list(argv))
    check = "--check" in list(argv)
    argv = [item for item in argv if item != "--check"]
    if any(item in ("-h", "--help") for item in argv):
        base.log("INFO", "launcher switches: --elbow-follow {operator,off} (-> --arm-elbow-follow), "
                         "--check (environment check only); the rest goes to the entry below")
    base.prepare_environment()
    base.redirect_caches()
    handed_over = hand_over((["--check"] if check else []) + list(argv))
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
    base.log("INFO", f"elbow follow: {elbow_follow_value(list(argv))}"
                     f" (the field build is --elbow-follow {ELBOW_BUILD})")
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
    if getattr(args, "input_npz", None) is not None:
        base.log("INFO", f"replay mode (--input-npz): no camera window opens, only the "
                         f"PyBullet window shows the three robots")
    elif getattr(args, "preview_window_pos", None) is not None:
        base.log("INFO", f"the camera window is moved to {args.preview_window_pos} and kept "
                         f"on top; look for the window titled {PREVIEW_WINDOW!r}")
    arrange_two_windows(args)
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
