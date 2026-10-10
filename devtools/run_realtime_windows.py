"""PyCharm 一键入口：摄像头窗口 + PyBullet 仿真窗口（不用敲命令）

为什么要有这个文件
------------------
真正的入口是 ``python -m teleoperation.apps.realtime_hand_sim``，它要求四件事同时成立：

1. 解释器是 ``E:\\python3.11.7\\python.exe``（仓库里的 ``.venv`` 是**空壳**，连
   ``python.exe`` 都没有）；
2. ``PYTHONPATH`` 里有 ``src``（包没 pip install 进解释器）；
3. 当前目录是仓库根目录（PyCharm 默认把工作目录设成脚本所在目录）；
4. 缓存/临时目录不在 C 盘（本仓库的硬约束）。

在 PyCharm 里点 Run 时这四条通常一条都不满足，本文件把它们补齐，所以"点 Run 就出两个窗口"。

PyCharm 怎么配
--------------
Run -> Edit Configurations -> Python -> 新建：

    Script path        = F:\\simulation_platform_cs\\devtools\\run_realtime_windows.py
    Working directory  = F:\\simulation_platform_cs
    Python interpreter = E:\\python3.11.7\\python.exe   <- 选错也能起来：本文件会自己在 E: 上重跑
    Parameters         = （留空；要换场景再填 --scene hands / --camera-index 1 等）

另外：**不要勾** "Run with Python Console"（那会把进程跑在交互式控制台里，退出方式不同）；
勾了也能跑，只是关窗口后控制台会被一起关掉。

命令行等价写法（随便一条）::

    E:\\python3.11.7\\python.exe devtools\\run_realtime_windows.py
    E:\\python3.11.7\\python.exe devtools\\run_realtime_windows.py --scene hands
    E:\\python3.11.7\\python.exe devtools\\run_realtime_windows.py --check      # 只体检

两个窗口
--------
1. 摄像头窗口（OpenCV 预览，默认左右镜像；按 q / Esc 退出）；
2. PyBullet 窗口：默认 ``--scene robots`` = H1-2 / GR1-T2 / G1 三台并排（原装手）。

退出码：0 = 有手输出过角度；1 = 全程没手（真实入口的设计如此）；3 = 解释器/依赖/模型文件不对。
``--check`` 的退出码：0 = 解释器/依赖/模型都对（相机读不到只 WARN）；3 = 有硬问题。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CANONICAL_PYTHON = Path(r"E:\python3.11.7\python.exe")
CACHE_ROOT = Path(r"E:\cache")
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
    """把缓存/临时目录指到 E:；只改「没设」或「指向 C:」的变量，已有的好值不动。"""
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


def missing_modules() -> list[str]:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def relaunch_with_canonical_python(argv: list[str]) -> "int | None":
    """解释器不对（缺 pybullet/cv2/mediapipe）时，用 E: 的解释器重跑本文件。

    返回子进程退出码；``None`` = 当前解释器就够用，继续在本进程里跑。
    """
    if os.environ.get(REEXEC_FLAG) == "1":
        return None
    missing = missing_modules()
    if not missing:
        return None
    log("WARN", f"current interpreter {sys.executable} lacks: {', '.join(missing)}")
    if not CANONICAL_PYTHON.is_file():
        log("FAIL", f"canonical interpreter not found: {CANONICAL_PYTHON}")
        log("INFO", "run this file with any interpreter that has pybullet+cv2+mediapipe")
        return 3
    if Path(sys.executable).resolve() == CANONICAL_PYTHON.resolve():
        log("FAIL", f"{CANONICAL_PYTHON} itself lacks: {', '.join(missing)}")
        return 3
    env = dict(os.environ)
    env[REEXEC_FLAG] = "1"
    log("OK", f"re-running with {CANONICAL_PYTHON} (PyCharm SDK may stay as it is)")
    completed = subprocess.run(
        [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *argv],
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


def run_check() -> int:
    """体检：解释器 / 依赖 / 仓库布局 / 模型文件 / 相机。不开任何窗口。"""
    log("INFO", f"interpreter : {sys.executable}")
    log("INFO", f"version     : {sys.version.split()[0]}")
    log("INFO", f"repo root   : {ROOT}")
    log("INFO", f"cwd         : {Path.cwd()}")
    if str(SRC) not in sys.path:
        log("FAIL", f"src not on sys.path: {SRC}")
        return 3
    missing = missing_modules()
    if missing:
        log("FAIL", f"missing modules: {', '.join(missing)}")
        return 3
    log("OK", f"modules present: {', '.join(REQUIRED_MODULES)}")
    try:
        from teleoperation.apps.realtime_hand_sim import resolve_model_asset
    except Exception as error:  # noqa: BLE001 - 体检要把原因原样报出来
        log("FAIL", f"import teleoperation.apps.realtime_hand_sim: "
                    f"{type(error).__name__}: {error}")
        return 3
    log("OK", "import teleoperation.apps.realtime_hand_sim")
    try:
        asset = resolve_model_asset(None)
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


def main(argv: list[str]) -> int:
    check = "--check" in argv
    argv = [item for item in argv if item != "--check"]

    prepare_environment()
    redirect_caches()
    handed_over = relaunch_with_canonical_python((["--check"] if check else []) + argv)
    if handed_over is not None:
        return handed_over
    if os.environ.get(REEXEC_FLAG) == "1":
        log("INFO", "running inside the re-executed process (E: interpreter)")

    if check:
        return run_check()

    from teleoperation.apps.realtime_hand_sim import build_parser
    from teleoperation.apps.realtime_hand_sim import main as sim_main

    extra = list(argv)
    if not any(item == "--report" or item.startswith("--report=") for item in extra):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORT_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
        extra += ["--report", str(report)]
        log("INFO", f"report -> {report}")

    args = build_parser().parse_args(extra)
    log("INFO", "two windows: camera preview + PyBullet; press q/Esc in the camera "
                "window (or close PyBullet) to stop")
    if getattr(args, "backend", "auto") != "checkpoint":
        log("INFO", f"backend={getattr(args, 'backend', 'auto')} is weight-free: hold your "
                    f"hand open for about {getattr(args, 'calibration_frames', 15)} frames "
                    f"so it can calibrate")
    return int(sim_main(args))


if __name__ == "__main__":
    # 与真实入口一致：先跑完 main()（统计/报告都写完），flush 之后直接 os._exit，
    # 免得卡在 MediaPipe landmarker.close()（本机实测约 42 s）上。
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
