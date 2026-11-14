"""Shared Windows launch preparation, checks and reports; no device starts on import."""
from __future__ import annotations


import importlib.util


import json


import os


import subprocess


import sys


import time
import threading


from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SRC = ROOT / "src"


FORMAL_PYTHON = Path(r"D:\Anaconda\envs\teleoperation\python.exe")


CANONICAL_PYTHON = (FORMAL_PYTHON if FORMAL_PYTHON.is_file()
                    else Path(sys.executable))


CACHE_ROOT = ROOT / "outputs" / "tmp_realtime" / "cache"


CACHE_VARS = (
    ("PIP_CACHE_DIR", "pip"),
    ("MPLCONFIGDIR", "matplotlib"),
    ("HF_HOME", "huggingface"),
    ("TORCH_HOME", "torch"),
    ("XDG_CACHE_HOME", "xdg"),
    ("TEMP", "tmp"),
    ("TMP", "tmp"),
)


REQUIRED_MODULES = ("cv2", "mediapipe", "pybullet")


REEXEC_FLAG = "TP_REALTIME_LAUNCHER_REEXEC"


REPORT_DIR = ROOT / "outputs" / "tmp_realtime"


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def prepare_environment() -> None:
    """sys.path / cwd / 标准输出编码，三件 PyCharm 不会替我们做的事。"""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    os.environ.setdefault("PYTHONUTF8", "1")   # 只影响子进程；当前解释器启动时已定
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 老控制台不支持就算了
            pass
    if Path.cwd() != ROOT:
        log("INFO", f"working directory {Path.cwd()} -> {ROOT}")
        os.chdir(ROOT)


def redirect_caches() -> None:
    """把缓存/临时目录指到项目outputs目录；只改「没设」或「指向 C:」的变量，已有的好值不动。"""
    for name, sub in CACHE_VARS:
        current = os.environ.get(name, "")
        wanted = CACHE_ROOT / sub
        if current and not current.upper().startswith("C:"):
            log("KEEP", f"{name} = {current}")
            continue
        try:
            wanted.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            log("WARN", f"{name}: cannot create {wanted} ({error})")
            continue
        os.environ[name] = str(wanted)
        log("OK", f"{name} -> {wanted} ({'was on C:' if current else 'was unset'})")


def missing_modules(extra=()) -> list[str]:
    missing = []
    for name in (*REQUIRED_MODULES, *extra):
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def gui_support() -> tuple[bool, str]:
    """cv2 能不能开窗口。

    headless 版 wheel（``opencv-python-headless``）能读摄像头却**开不了窗口**，
    依赖检查会通过，然后入口里 ``imshow`` 报错、只打一行 note 就继续跑仿真
    （症状：只看得到 PyBullet 窗口）。这里读 build info 的 ``GUI:`` 行把它抓出来；
    只读字符串，不开窗口、不碰相机，所以不会挂住。
    """
    try:
        import cv2
    except Exception as error:  # noqa: BLE001
        return False, f"import cv2 failed: {type(error).__name__}: {error}"
    if not hasattr(cv2, "imshow"):
        return False, "this cv2 has no imshow"
    try:
        info = cv2.getBuildInformation()
    except Exception as error:  # noqa: BLE001 - 老版本没有这个方法：不拦
        return True, f"unknown (getBuildInformation failed: {type(error).__name__})"
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("GUI"):
            value = stripped.split(":", 1)[-1].strip()
            if value.upper().startswith("NONE"):
                return False, ("headless build (GUI: NONE): it can read the camera but "
                               "cannot open a window")
            return True, value or "yes"
    return True, "unknown (no GUI line in build info)"


def wants_window(argv: list[str]) -> bool:
    """没写 ``--no-preview`` / ``--headless`` 就打算开摄像头窗口（也就需要 GUI）。"""
    return not any(item in ("--no-preview", "--headless") for item in argv)


def _selected_backend(argv):
    backend = 'dex'
    for i, value in enumerate(argv):
        if value == '--backend' and i + 1 < len(argv):
            backend = argv[i + 1]
        if value.startswith('--backend='):
            backend = value.split('=', 1)[1]
    return backend


def relaunch_with_canonical_python(argv, *, script_path):
    """Re-execute the original launcher, keeping every argument and exit code."""
    if os.environ.get(REEXEC_FLAG) == '1':
        return None
    backend = _selected_backend(argv)
    missing = missing_modules(('dex_retargeting',) if backend == 'dex' else ('torch',))
    problem = f"lacks: {', '.join(missing)}" if missing else ''
    if not problem and wants_window(argv):
        ok, detail = gui_support()
        if not ok:
            problem = f'cannot open a GUI window: {detail}'
    if not problem:
        return None
    dex_python = ROOT / '.venv/dex-native/Scripts/python.exe'
    canonical = dex_python if backend == 'dex' and dex_python.is_file() else CANONICAL_PYTHON
    log('WARN', f'current interpreter {sys.executable} {problem}')
    if not canonical.is_file() or Path(sys.executable).resolve() == canonical.resolve():
        log('FAIL', f'no usable alternate interpreter: {canonical}; {problem}')
        return 3
    env = dict(os.environ)
    env[REEXEC_FLAG] = '1'
    completed = subprocess.run([str(canonical), str(Path(script_path).resolve()), *argv],
                               cwd=str(ROOT), env=env)
    return int(completed.returncode)


def probe_cameras(max_index: int = 3) -> list[int]:
    """返回能读出画面的相机下标（开-读一帧-立刻释放，不长时间占相机）。"""
    import cv2

    try:
        # 关掉 OpenCV 自己的 stderr 噪音（相机索引越界会打一堆 ERROR:0@...，
        # 在 PyCharm 控制台里显示成红色，容易误判成我们的失败）。
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:  # noqa: BLE001 - 老版本没有 logging 模块就算了
        pass

    usable: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    usable.append(index)
        finally:
            capture.release()
    return usable


def run_check(want_window: bool = True, *, backend=None, model_asset_path=None) -> int:
    """体检：解释器 / 依赖 / 窗口支持 / 仓库布局 / 模型文件 / 相机。不开任何窗口。"""
    log("INFO", f"interpreter : {sys.executable}")
    log("INFO", f"version     : {sys.version.split()[0]}")
    log("INFO", f"repo root   : {ROOT}")
    log("INFO", f"cwd         : {Path.cwd()}")
    if str(SRC) not in sys.path:
        log("FAIL", f"src not on sys.path: {SRC}")
        return 3
    missing = missing_modules(('dex_retargeting',) if backend == 'dex' else ('torch',) if backend == 'nn' else ())
    if missing:
        log("FAIL", f"missing modules: {', '.join(missing)}")
        return 3
    log("OK", f"modules present: {', '.join(REQUIRED_MODULES)}")
    gui_ok, gui_detail = gui_support()
    if gui_ok:
        log("OK", f"cv2 window support: {gui_detail}")
    elif want_window:
        log("FAIL", f"cv2 cannot open a window in this interpreter: {gui_detail}")
        log("INFO", f"use {CANONICAL_PYTHON}, or add --no-preview to skip the camera window")
        return 3
    else:
        log("WARN", f"cv2 window support: {gui_detail} (fine with --no-preview)")
    try:
        from teleoperation.applications.realtime_hand_sim import resolve_model_asset
    except Exception as error:  # noqa: BLE001 - 体检要把原因原样报出来
        log("FAIL", f"import teleoperation.applications.realtime_hand_sim: "
                    f"{type(error).__name__}: {error}")
        return 3
    log("OK", "import teleoperation.applications.realtime_hand_sim")
    try:
        asset = resolve_model_asset(model_asset_path)
        log("OK", f"model asset : {asset} ({asset.stat().st_size} bytes)")
    except Exception as error:  # noqa: BLE001
        log("FAIL", f"model asset not found: {type(error).__name__}: {error}")
        return 3
    try:
        cameras = probe_cameras()
    except Exception as error:  # noqa: BLE001
        log("FAIL", f"camera probe failed: {type(error).__name__}: {error}")
        return 3
    if cameras:
        log("OK", f"cameras readable: {cameras}  (pick one with --camera-index N)")
    else:
        log("WARN", "no camera gave a frame: check USB / another app holding it")
    log("OK", "check finished; ready to open the two windows")
    return 0


def summarise_camera_window(report: Path, want_window: bool) -> None:
    """跑完读一眼报告里的 ``preview`` 块：摄像头窗口到底开没开、开在哪。

    入口在预览失败时只打一行 note 就继续跑仿真（设计如此：不拖累重定向与仿真），
    所以在 IDE 里很容易"只看见机器人窗口"。这里在进程结束后明确回一句结论。

    **别把 ``preview["window"]`` 当成"窗口开没开"**：它读的是写报告那一刻的
    ``CameraPreview.show``，而入口收尾的 ``finally`` 里用后台线程调 ``preview.close()``
    （realtime_support的有界清理，最多等 2 s），``close()`` 又无条件把
    ``show`` 置 ``False``（camera_preview）—— 写报告与关窗是
    **竞态**，正常退出时这个字段可 ``True`` 可 ``False``（2026-10-14 实测：按 q 退出的长跑
    报 ``False``，``--frames 40`` 的短跑报 ``True``，两次窗口都是开着的）。

    真正能证明"窗口开过、且被摆到屏幕可见处"的是 ``window_rect``：它只在成功的
    ``_place()`` 里写（先 ``imshow`` 再 ``moveWindow`` + 置顶）。所以分三段判断：
    还在显示 -> OK；显示过（有矩形）-> OK（已随本次运行结束关闭）；从没开成 -> WARN + 重试命令。
    """
    if not want_window:
        return
    try:
        data = json.loads(Path(report).read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        log("WARN", f"cannot read report {report}: {type(error).__name__}: {error}")
        return
    preview = data.get("preview") or {}
    if not preview:
        return
    rect = preview.get("window_rect")
    if preview.get("window"):
        log("OK", f"camera window stayed open at {rect} "
                  f"(mirrored={preview.get('mirrored')})")
        return
    if rect:
        log("OK", f"camera window was open at {rect} "
                  f"(mirrored={preview.get('mirrored')}) while the run lasted; it is closed "
                  f"now because the run ended (q/Esc or the PyBullet window was closed)")
        return
    log("WARN", f"the camera window did NOT stay open: "
                f"{preview.get('note') or 'unknown reason'}")
    log("INFO", f"retry with: {CANONICAL_PYTHON} {Path(__file__).name} "
                f"--preview-window-pos 20,60")



def check_arm_assets(args):
    """Check the configured isolated SMPLer-X paths without running inference."""
    for name in ('smplerx_worker_python', 'smplerx_root', 'smplerx_checkpoint', 'smplx_model_dir'):
        path = Path(getattr(args, name))
        if not path.exists():
            log('FAIL', f'{name} not found: {path}')
            return 3
        log('OK', f'{name}: {path}')
    return 0


PYBULLET_WINDOW_CLASS = "DeviceWin32"

PYBULLET_WINDOW_TITLE = "Bullet Physics"

LAYOUT_ENV = "TELEOP_WINDOW_LAYOUT"

LAYOUT_TIMEOUT_S = 30.0

LAYOUT_POLL_S = 0.4

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
        log("INFO", f"window layout off ({LAYOUT_ENV}=off): drag the windows yourself")
        return None
    if getattr(args, "headless", False) or getattr(args, "no_preview", False):
        return None
    screen = screen_size()
    if screen is None or _win32() is None:
        log("INFO", "cannot query the screen size here: both windows stay where the "
                         "system puts them (pass --preview-window-pos X,Y to move the camera one)")
        return None
    camera, target = planned_layout(args, screen)
    where = ("where the system puts it" if camera is None
             else f"at {camera[0]},{camera[1]}")
    log("INFO", f"two windows side by side (screen {screen[0]}x{screen[1]}): camera preview "
                     f"on the left {where}, PyBullet (three robots) on the right at "
                     f"{target[0]},{target[1]} ({target[2]}x{target[3]})")
    log("INFO", f"set {LAYOUT_ENV}=off to place the windows yourself")

    def worker():
        deadline = time.monotonic() + LAYOUT_TIMEOUT_S
        while time.monotonic() < deadline:
            hwnd = find_pybullet_window()
            if hwnd:
                before = window_rect(hwnd)
                if before == target:
                    log("OK", f"PyBullet window already at {target}")
                elif move_window(hwnd, target):
                    log("OK", f"PyBullet window {before} -> {target}")
                else:
                    log("WARN", f"cannot move the PyBullet window (stays {before})")
                return
            time.sleep(LAYOUT_POLL_S)
        log("WARN", f"PyBullet window not found within {LAYOUT_TIMEOUT_S:.0f} s: "
                         f"drag it next to the camera window yourself")

    thread = threading.Thread(target=worker, name="pybullet-window-layout", daemon=True)
    thread.start()
    return thread


def apply_elbow_preset(argv):
    """Translate old launcher switches; explicit public options take priority."""
    cleaned, wanted = [], 'operator'
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == '--off':
            wanted = 'off'
        elif item == '--elbow-follow':
            if i + 1 == len(argv) or argv[i + 1] not in ('operator', 'off'):
                raise ValueError('--elbow-follow requires operator or off')
            i += 1
            wanted = argv[i]
        elif item.startswith('--elbow-follow='):
            wanted = item.split('=', 1)[1]
            if wanted not in ('operator', 'off'):
                raise ValueError('--elbow-follow requires operator or off')
        else:
            cleaned.append(item)
        i += 1
    if not any(v == '--arm-elbow-follow' or v.startswith('--arm-elbow-follow=') for v in cleaned):
        cleaned.extend(('--arm-elbow-follow', wanted))
    return cleaned


def run_camera_launcher(argv, *, with_arms, script_path):
    """Keep diagnostic camera options while sharing launch orchestration."""
    argv = list(argv)
    if with_arms:
        argv = apply_elbow_preset(argv)
        if any(v in ('--help', '-h') for v in argv):
            log('INFO', 'launcher: --elbow-follow {operator,off}, --off, --check; explicit --arm-elbow-follow wins')
    check = '--check' in argv
    argv = [value for value in argv if value != '--check']
    prepare_environment()
    redirect_caches()
    handed_over = relaunch_with_canonical_python((['--check'] if check else []) + argv,
                                                 script_path=script_path)
    if handed_over is not None:
        return handed_over
    from teleoperation.simulation.realtime_teleop import build_parser, main as entry_main
    args = build_parser().parse_args(argv if with_arms else ['--no-arms', *argv])
    if check:
        status = run_check(want_window=wants_window(argv), backend=args.backend,
                           model_asset_path=args.model_asset_path)
        return status if status or args.no_arms else check_arm_assets(args)
    if not args.robot or len(set(args.robot)) != len(args.robot):
        log('FAIL', 'Choose distinct targets explicitly with --robot NAME; repeat for comparison')
        return 2
    report = None
    if args.report is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        prefix = 'arm_run' if not args.no_arms else 'run'
        report = REPORT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        args.report = report
    log('INFO', f'interpreter: {sys.executable}')
    log('INFO', f"targets: {', '.join(args.robot)}; backend={args.backend}; arms={not args.no_arms}")
    if wants_window(argv):
        log('INFO', 'camera preview and PyBullet enabled; q/Esc stops the preview')
    arrange_two_windows(args)
    status = int(entry_main(args))
    if report is not None:
        summarise_camera_window(report, want_window=wants_window(argv))
    return status
