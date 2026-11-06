"""Native L21 simulation and shared camera preview utilities."""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

from teleoperation.data.urdf import parse_joint_limits
from teleoperation.paths import (
    DEFAULT_MEDIAPIPE_ASSET,
    L21_ROOT,
    PROJECT_ROOT,
)
from teleoperation.robots.l21 import L21_JOINT_ORDER, L21HandAdapter

SIDES = ("left", "right")
URDF_NAME = "linkerhand_l21_{side}.urdf"
ASSET_NAME = "hand_landmarker.task"
# 双手并排摆位；单只手时放在原点前方。
BASE_POSITIONS = {"left": (-0.15, 0.0, 0.3), "right": (0.15, 0.0, 0.3)}
# 摄像头窗口：标题、指尖下标（复用 tools/plotting 的 COLORS 与 SOURCE_EDGES 连线）。
PREVIEW_WINDOW = "camera: MediaPipe 21-landmark input (q/Esc quits)"
PREVIEW_TIPS = (4, 8, 12, 16, 20)
PREVIEW_HINT_ROBOTS = (
    "two windows: PyBullet shows H1-2 / GR1-T2 / G1 side by side with their native "
    "hands, the camera window shows your hand; press q/Esc in the camera window to stop.")
PREVIEW_HINT_HANDS = (
    "two windows: PyBullet shows the L21 hand, the camera window shows your hand; "
    "press q/Esc in the camera window to stop.")


def resolve_model_asset(path: Path | None) -> Path:
    """Return the MediaPipe asset path, or raise with the exact places searched."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([DEFAULT_MEDIAPIPE_ASSET, PROJECT_ROOT / ASSET_NAME, PROJECT_ROOT / "outputs" / ASSET_NAME])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"MediaPipe model asset not found; searched: {searched}. Download the "
        f"official hand_landmarker.task (float16, about 7.5 MB) and put it in one "
        f"of those paths, or pass --model-asset-path."
    )






class RealtimeL21Scene:
    """PyBullet L21 scene; native named commands plus explicit historical layout reader.

    Only the native joints in ``L21_JOINT_ORDER`` are written; any
    extra URDF joints (if a model ever has them) stay at their loaded state.
    """

    def __init__(self, sides, gui: bool = True):
        self.cid = -1
        try:
            self._initialize(sides,gui)
        except BaseException:
            if self.cid >= 0 and hasattr(self,'p'):
                self.close()
            raise

    def _initialize(self, sides, gui):
        import pybullet as p

        if not sides:
            raise ValueError("at least one hand side is required")
        self.p = p
        self.sides = tuple(sides)
        self.gui = bool(gui)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0 or not p.isConnected(self.cid):
            raise RuntimeError(
                "PyBullet GUI connection could not be created (no display or a "
                "non-interactive session); rerun with --headless."
            )
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # PyBullet 的 GUI 服务端是异步起来的：本机实测第一次 showGUI 后立刻调用
        # setGravity 会报 "Not connected to physics server"。这里重试到服务端应答为止。
        # 这是可视化场景：不模拟重力，否则 resetJointState 写入的角度会在
        # stepSimulation 后被重力拖走（探针实测拇指自由关节漂移约 1e-3 rad）。
        self._prepare_physics(p.setGravity, (0, 0, 0))
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
        self.bodies = {}
        self.base_positions = {}
        self.view = "full"
        self.joint_index = {}
        self.adapters = {}
        for side in self.sides:
            directory = L21_ROOT / side
            urdf = directory / URDF_NAME.format(side=side)
            if not urdf.is_file():
                raise FileNotFoundError(f"L21 model not found: {urdf}")
            p.setAdditionalSearchPath(str(directory), physicsClientId=self.cid)
            base = list(BASE_POSITIONS[side]) if len(self.sides) > 1 else [0.0, 0.0, 0.3]
            self.base_positions[side] = base
            body = p.loadURDF(str(urdf), base, useFixedBase=True, physicsClientId=self.cid)
            names = {}
            for index in range(p.getNumJoints(body, physicsClientId=self.cid)):
                info = p.getJointInfo(body, index, physicsClientId=self.cid)
                name = info[1].decode() if isinstance(info[1], bytes) else info[1]
                names[name] = index
            missing = [name for name in L21_JOINT_ORDER if name not in names]
            if missing:
                raise RuntimeError(f"{urdf.name} lacks L21 joints: {missing}")
            self.bodies[side] = body
            self.joint_index[side] = names
            self.adapters[side] = L21HandAdapter(list(L21_JOINT_ORDER), parse_joint_limits(urdf))
        self.frame_scene("full")
        self.applied = {side: 0 for side in self.sides}
        self.loss_reported = False

    def frame_scene(self, view="full") -> None:
        """Front: both palms toward the camera; full: original URDF layout.

        Both native palms face +X. A -90 degree world-Z rotation points them
        toward a yaw=0 camera on -Y, with the fingers still pointing upward.
        Only rigid display poses change; native joint angles remain untouched.
        """
        if view not in ("full", "front"):
            raise ValueError(f"Unknown L21 view: {view}")
        if not self.alive():
            return
        if view != self.view:
            orientation = self.p.getQuaternionFromEuler([0, 0, -np.pi / 2 if view == "front" else 0])
            for side, body in self.bodies.items():
                dynamics = self.p.getDynamicsInfo(body, -1, physicsClientId=self.cid)
                # resetBasePositionAndOrientation expects the inertial/COM pose,
                # whereas loadURDF takes the root-link pose (different on left).
                position, rotation = self.p.multiplyTransforms(
                    self.base_positions[side], orientation, dynamics[3], dynamics[4])
                self.p.resetBasePositionAndOrientation(body, position, rotation, physicsClientId=self.cid)
            self.view = view
        if self.gui:
            self.p.resetDebugVisualizerCamera(
                cameraDistance=0.7 if len(self.sides) > 1 else 0.55,
                cameraYaw=0 if view == "front" else 40,
                cameraPitch=0 if view == "front" else -20,
                cameraTargetPosition=[0.0, 0.0, 0.43 if view == "front" else 0.3],
                physicsClientId=self.cid,
            )

    def _prepare_physics(self, call, positional, timeout: float = 15.0) -> None:
        """Call a PyBullet setup function, retrying until the server answers.

        ``p.connect(p.GUI)`` returns before the GUI server is ready, so the first
        call can raise ``Not connected to physics server`` (observed here on the
        first GUI start). Retrying keeps the entry usable when the window needs a
        moment to appear; the DIRECT server answers immediately.
        """
        deadline = time.perf_counter() + timeout
        while True:
            try:
                call(*positional, physicsClientId=self.cid)
                return
            except Exception as error:  # noqa: BLE001 - retried until the deadline
                if time.perf_counter() >= deadline:
                    raise RuntimeError(
                        f"PyBullet did not answer within {timeout:.0f} s "
                        f"({type(error).__name__}: {error}); rerun with --headless."
                    ) from error
                time.sleep(0.1)


    def apply_native(self, command):
        """Apply expanded named L21 commands without a historical angle layout."""
        from teleoperation.simulation.hand_executor import HandExecutor
        from teleoperation.robots.hand_spec import load_hand_spec
        if not hasattr(self, '_native_executor'):
            specs = {('l21',side):load_hand_spec('l21',side) for side in self.sides}
            def write(value):
                if not self.alive():
                    raise RuntimeError('L21 simulator connection closed')
                for name, angle in value.named_positions().items():
                    self.p.resetJointState(self.bodies[value.side],self.joint_index[value.side][name],angle,physicsClientId=self.cid)
                self.applied[value.side] += 1
            self._native_executor = HandExecutor(specs,{key:write for key in specs})
        self._native_executor.apply(command)

    def step(self) -> None:
        """Step physics; a lost connection is reported once and then ignored.

        The PyBullet GUI connection can drop mid-run (window closed, or a session
        that cannot create an OpenGL window). Stepping a dead server raises
        ``Not connected to physics server``, which would also skip the summary and
        the ``--report`` file; instead we note it once and let the loop exit.
        """
        if not self.alive():
            if not self.loss_reported:
                self.loss_reported = True
                print("PyBullet connection lost (GUI window closed or no display in this "
                      "session); stopping and still writing the summary. Use --headless "
                      "when the GUI cannot open here.", flush=True)
            return
        self.p.stepSimulation(physicsClientId=self.cid)

    def alive(self) -> bool:
        """False once the user closes the GUI window."""
        return bool(self.p.isConnected(self.cid))

    def close(self) -> None:
        if self.p.isConnected(self.cid):
            self.p.disconnect(self.cid)

    def joint_values(self, side: str) -> dict:
        """Current L21 joint values, for reporting what the sim actually shows."""
        if side not in self.bodies or not self.alive():
            return {}
        body = self.bodies[side]
        index = self.joint_index[side]
        return {
            name: float(self.p.getJointState(body, index[name], physicsClientId=self.cid)[0])
            for name in L21_JOINT_ORDER
        }


class _PreviewCapture:
    """透过式包装 ``cv2.VideoCapture``：记住最近一帧 BGR，其余行为原样转发。

    输入层（``inputs/mediapipe.py``，队友文件）只返回关键点、不暴露图像，而约定 3
    不允许改队友文件；于是给**已有的**采集对象套一层代理来取图。``read()`` 的返回值
    一字不改地透传给输入层，检测结果/时间戳/元数据都不受影响。
    """

    def __init__(self, capture, sink):
        self._capture = capture
        self.sink = sink

    def read(self):
        ok, frame = self._capture.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            self.sink["frame"] = frame
        return ok, frame

    def __getattr__(self, name):
        # release()/isOpened()/set() 等仍然落到真正的 VideoCapture 上。
        return getattr(object.__getattribute__(self, "_capture"), name)


def attach_preview_capture(workflow):
    """把工作流内部的 ``cv2.VideoCapture`` 换成 ``_PreviewCapture``，返回它的 sink。

    ``sink["frame"]`` 就是最近一帧 BGR 画面；输入层没暴露采集对象时返回 ``None``，
    调用方静默跳过预览。
    """
    device = getattr(workflow, "_input", None)
    capture = getattr(device, "_cap", None)
    if capture is None:
        return None
    if isinstance(capture, _PreviewCapture):
        return capture.sink
    proxy = _PreviewCapture(capture, {})
    try:
        device._cap = proxy
    except Exception:  # noqa: BLE001 - 预览是可选功能：装不上就当作没有
        return None
    return proxy.sink


def _desktop_size():
    """Windows 虚拟桌面像素尺寸（多屏时是并集）；非 Windows 或取不到时返回 ``None``。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(78)), int(user32.GetSystemMetrics(79))
    except Exception:  # noqa: BLE001 - 取不到尺寸就不做越界检查
        return None


def _window_pos(text: str):
    """``"20,60"`` -> ``(20, 60)``；``"none"`` -> ``None``（保留系统默认位置）。

    给 ``--preview-window-pos`` 用：把摄像头窗口钉在确定的屏幕坐标上，避免它落在
    屏幕外或被 PyBullet 窗口整个盖住（IDE 里跑时的典型症状：只看见机器人窗口）。
    """
    value = text.strip().lower()
    if value in ("none", "default", "off"):
        return None
    parts = value.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y such as 20,60 (or 'none')")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "X and Y must be integers, for example 20,60") from None


class CameraPreview:
    """摄像头窗口：原始画面 + 21 点骨架 + 一行状态（纯显示，像素不回灌算法）。

    画的是**原始**归一化关键点乘上图像尺寸，因此这个窗口同时是"MediaPipe 到底看没
    看到手"的现场证据。镜像显示时关键点 x 同步翻转，骨架仍贴合画面。
    """

    def __init__(self, sink, sides, *, show_window=True, mirror=True,
                 dump_dir=None, dump_frames=0, window_name=PREVIEW_WINDOW,
                 window_pos=(20, 60), topmost=True):
        import cv2

        from teleoperation.tools.plotting import COLORS, SOURCE_EDGES, draw_skeleton, text

        self._cv2 = cv2
        self._colors, self._edges = COLORS, SOURCE_EDGES
        self._draw_skeleton, self._text = draw_skeleton, text
        self.sink = sink
        self.sides = tuple(sides)
        self.show = bool(show_window)
        self.mirror = bool(mirror)
        self.window_name = window_name
        self.window_pos = None if window_pos is None else tuple(window_pos)
        self.topmost = bool(topmost)
        self.screen = _desktop_size()
        self.rect = None
        self._placed = False
        self.dump_dir = None if dump_dir is None else Path(dump_dir)
        self.dump_frames = int(dump_frames)
        self.dumped = 0
        self.quit_requested = False
        self.note = ""
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    def pixels(self, points, width, height):
        """归一化 (21, 3) -> 整数像素 (21, 2)；镜像时同步翻转 x。"""
        xy = np.asarray(points, dtype=np.float64)[:, :2]
        values = np.stack((xy[:, 0] * (width - 1), xy[:, 1] * (height - 1)), axis=-1)
        if self.mirror:
            values[:, 0] = (width - 1) - values[:, 0]
        return values.astype(np.int32)

    def annotate(self, raw_frame, angles, hud):
        """把一帧摄像头画面画成预览画布；还没有画面时返回 ``None``。"""
        frame = self.sink.get("frame")
        if frame is None:
            return None
        canvas = self._cv2.flip(frame, 1) if self.mirror else frame.copy()
        height, width = canvas.shape[:2]
        for index, side in enumerate(self.sides):
            points = None if raw_frame is None else raw_frame.get(side)
            if points is None:
                continue
            pixels = self.pixels(points, width, height)
            self._draw_skeleton(canvas, pixels, self._edges, self._colors[side],
                                PREVIEW_TIPS, False)
            self._text(canvas, side, (int(pixels[0][0]) + 8, int(pixels[0][1]) - 8),
                       self._colors[side], 0.6)
            if angles.get(side) is not None:
                self._text(canvas, f"{side} angles: {len(angles[side])} dims", (10, height - 10 - 20 * index),
                           self._colors[side], 0.5)
        if all((raw_frame or {}).get(side) is None for side in self.sides):
            self._text(canvas, "no hand detected (keep the hand in frame, well lit)",
                       (10, height - 12), (60, 60, 235), 0.55)
        self._text(canvas, hud, (10, 22), (235, 235, 235), 0.5)
        return canvas

    def update(self, raw_frame, angles, hud):
        """画一帧、按需存 PNG、刷新窗口；用户按 q/Esc 或关窗时置 ``quit_requested``。"""
        canvas = self.annotate(raw_frame, angles, hud)
        if canvas is None:
            return
        if self.dump_dir is not None and self.dumped < self.dump_frames:
            self._cv2.imwrite(str(self.dump_dir / f"preview_{self.dumped:04d}.png"), canvas)
            self.dumped += 1
        if not self.show:
            return
        try:
            self._cv2.imshow(self.window_name, canvas)
            if not self._placed:
                # 窗口此时才真正存在，这里挪位置/置顶最可靠（先 imshow 再 move）。
                self._place(canvas)
            key = self._cv2.waitKey(1) & 0xFF
        except self._cv2.error as error:      # 没有桌面 / 没有 HighGUI 支持
            self.show = False
            self.note = f"window unavailable here ({error}); preview kept off"
            print(f"note: {self.note}", flush=True)
            return
        if key in (27, ord("q")):
            self.quit_requested = True
            return
        try:
            if self._cv2.getWindowProperty(self.window_name, self._cv2.WND_PROP_VISIBLE) < 1:
                self.quit_requested = True
        except self._cv2.error:
            self.quit_requested = True

    def _place(self, canvas):
        """把窗口挪到确定可见的位置并置顶，再打印它的实际屏幕矩形。

        PyBullet 的窗口先创建且尺寸更大，OpenCV 窗口按系统默认位置（多屏时可能落在
        副屏或屏幕外）出现，很容易"看不见窗口但程序在跑"。这里把它钉在
        ``--preview-window-pos`` 并置顶，同时把矩形打进日志，方便事后核对。
        """
        self._placed = True
        height, width = canvas.shape[:2]
        try:
            if self.window_pos is not None:
                self._cv2.moveWindow(self.window_name, int(self.window_pos[0]),
                                     int(self.window_pos[1]))
            self._cv2.resizeWindow(self.window_name, width, height)
            self._cv2.imshow(self.window_name, canvas)
            if self.topmost:
                self._cv2.setWindowProperty(self.window_name, self._cv2.WND_PROP_TOPMOST, 1)
                self._cv2.imshow(self.window_name, canvas)
        except AttributeError:
            # 老 OpenCV 没有 WND_PROP_TOPMOST：位置已经挪过，继续跑
            pass
        except self._cv2.error as error:
            print(f"note: cannot position the camera window ({error}); leaving it as is",
                  flush=True)
        try:
            self.rect = tuple(int(value) for value in
                              self._cv2.getWindowImageRect(self.window_name))
        except Exception:  # noqa: BLE001 - 有的后端不支持查询矩形
            self.rect = None
        where = "unknown" if self.rect is None else \
            f"{self.rect[0]},{self.rect[1]} ({self.rect[2]}x{self.rect[3]})"
        screen = "" if self.screen is None else f", screen {self.screen[0]}x{self.screen[1]}"
        print(f"camera window placed at {where}"
              f"{' [always on top]' if self.topmost else ''}{screen} "
              f"title={self.window_name!r}", flush=True)
        if self.rect is not None and self.screen is not None:
            x, y, w, h = self.rect
            off = (w <= 0 or h <= 0 or x + w < 40 or y + h < 40
                   or x > self.screen[0] - 40 or y > self.screen[1] - 40)
            if off:
                print(f"WARNING: the camera window is outside the visible desktop "
                      f"(rect={self.rect}, screen={self.screen}); pass "
                      f"--preview-window-pos 20,60 to move it.", flush=True)

    def close(self):
        """关窗口；显示失败也不影响主链路与统计。"""
        if self.show:
            try:
                self._cv2.destroyWindow(self.window_name)
            except self._cv2.error:
                pass
        self.show = False


def build_preview(args, workflow, sides):
    """按参数建预览并打印它到底开没开窗；不需要 / 装不上时返回 ``None``。"""
    want_window = not args.headless and not args.no_preview
    if not want_window and not args.preview_dump_dir:
        return None
    sink = attach_preview_capture(workflow)
    if sink is None:
        print("note: the input layer does not expose a camera frame; preview skipped",
              flush=True)
        return None
    preview = CameraPreview(sink, sides, show_window=want_window,
                            mirror=not args.preview_no_mirror,
                            dump_dir=args.preview_dump_dir,
                            dump_frames=args.preview_dump_frames,
                            window_pos=args.preview_window_pos,
                            topmost=not args.preview_no_topmost)
    dump = (""
            if preview.dump_dir is None
            else f", dumping first {preview.dump_frames} frames to {preview.dump_dir}")
    print(f"camera window: {'on' if preview.show else 'off'}"
          f"{' (mirrored)' if preview.mirror else ''}{dump}"
          f", title={preview.window_name!r}"
          f"{'' if preview.window_pos is None else f', position={preview.window_pos}'}",
          flush=True)
    return preview




def _describe(vectors, mapping=None) -> dict:
    """Frame count plus per-dimension observed range for one side."""
    mapping = mapping if mapping is not None else list(L21_JOINT_ORDER)
    array = np.asarray(vectors, dtype=np.float64).reshape(-1, len(mapping))
    return {
        "frames": int(array.shape[0]),
        "min": [float(value) for value in array.min(axis=0)],
        "max": [float(value) for value in array.max(axis=0)],
        "mapping": list(mapping),
    }


def build_scene(args, sides):
    """Construct a native hand or whole-robot simulation scene."""
    if args.scene == "hands":
        return RealtimeL21Scene(sides, gui=not args.headless)
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene

    return ThreeRobotHandScene(sides, gui=not args.headless,
                               steps_per_frame=args.sim_steps)




def _loop_fps(marks) -> float:
    """Loop rate over the last second of frame marks (0.0 when undefined)."""
    if len(marks) < 2:
        return 0.0
    cutoff = marks[-1] - 1.0
    recent = [mark for mark in marks if mark >= cutoff]
    if len(recent) < 2:
        recent = marks[-2:]
    return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-9)


def _median_interval(marks) -> float:
    if len(marks) < 2:
        return 0.0
    return statistics.median([later - earlier for earlier, later in zip(marks, marks[1:])])


# MediaPipe 的 TFLite 释放在本机实测 42.2 s（cv2 只要 0.3 s），PyBullet 在 GUI 客户端
# 已经死掉时 disconnect 也会挂住，所以所有收尾动作都只等这么久。
RELEASE_TIMEOUT_SECONDS = 2.0


def _drain_in_background(action, label: str, timeout: float = RELEASE_TIMEOUT_SECONDS) -> None:
    """Run a slow teardown step in a daemon thread; never block the exit.

    ``MediaPipeCameraInput.release()`` spends about 42 s inside
    ``landmarker.close()`` (TFLite/XNNPACK teardown) on this machine, and
    ``p.disconnect()`` hangs when the PyBullet GUI client already died. Both look
    like a freeze. We wait at most ``timeout`` seconds and then continue;
    a teardown still running at that point remains in the daemon thread.
    """

    def run_action() -> None:
        try:
            action()
        except Exception as error:  # noqa: BLE001 - teardown must not hide the summary
            print(f"note: {label} failed: {type(error).__name__}: {error}", flush=True)

    thread = threading.Thread(target=run_action, name="teardown", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"note: {label} is still running after {timeout:.0f} s; exiting without "
              f"waiting.", flush=True)
