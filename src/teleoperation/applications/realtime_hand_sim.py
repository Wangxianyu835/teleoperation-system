"""实时闭环入口：摄像头 -> 手掌局部 25 点 -> 重定向 -> PyBullet 里的机器人灵巧手。

显示目标由 ``--scene`` 选择（**默认 robots**）::

    robots  三台论文机器人 H1-2 / GR1-T2 / G1 并排，各用出厂自带的【原装手】
            （``apps/three_robot_scene.py``；画面里是完整机器人，不是单独一只手）
    hands   只摆 1-2 只 LinkerHand L21 手（旧行为，方便单独看手部关节）

这条链路原本缺一个"入口"：``hand realtime`` 只加载训练出的检查点模型并打印
角度，``replay l21-hand`` 只回放离线 H5，两者都不能把摄像头画面实时显示到
仿真里。本模块补上这一环，并且**不改动任何既有模块**：

    MediaPipeCameraInput -> MediaPipeCameraWorkflow（掌面局部对齐 + 三帧窗口）
    -> GeometricHandRetargeter（零权重，本仓库自带的几何后端）
    -> PyBullet GUI 里的显示目标（``--scene``：默认三台机器人的原装手）

后端选择（``--backend``）：

    auto       默认。检查点存在就用训练模型，否则自动回落到几何后端并打印原因。
    geometric  强制几何后端（零权重，不需要任何 .pth）。
    checkpoint 强制训练后端（需要 palm_local 检查点）。

模型文件（``hand_landmarker.task``，约 7.5 MB）默认按以下顺序查找：
``--model-asset-path`` > 仓库根 ``hand_landmarker.task`` > ``outputs/hand_landmarker.task``。

用法::

    # 两个窗口：PyBullet 里三台机器人并排 + 摄像头画面（叠加 21 点骨架）
    python -m teleoperation.apps.realtime_hand_sim --backend auto

    # 只摆一只手（旧行为）
    python -m teleoperation.apps.realtime_hand_sim --scene hands

    # 正面视角看三台机器人
    python -m teleoperation.apps.realtime_hand_sim --view front

    # 只要 PyBullet 窗口，不显示摄像头画面
    python -m teleoperation.apps.realtime_hand_sim --no-preview

    # 无头自检（不弹窗，跑 300 帧后打印统计并写报告）
    python -m teleoperation.apps.realtime_hand_sim --headless --frames 300 \
        --report outputs/tmp_probe/realtime_report.json

摄像头窗口（``cv2.imshow``）：原始画面 + MediaPipe 21 点骨架（``COLORS``：浅色
left、深色 right），左上角一行是帧数/帧率/有效角度帧数/开手标定状态，没有手时
在左下角写 "no hand detected"。默认左右**镜像**（像照镜子，方便对着屏幕调手势），
``--preview-no-mirror`` 关掉。默认开窗口；``--headless`` 下自动不开（只跑自检）。
``--preview-dump-dir DIR --preview-dump-frames N`` 会把前 N 帧带骨架的画面存成 PNG，
用来在没有显示器的环境里核对（这两个开关在 ``--headless`` 下也能用）。

输入层（``inputs/mediapipe.py``，队友文件）只返回关键点、不暴露 BGR 图像，
CONVENTIONS 约定 3 又不允许改队友文件，因此这里给它内部的 ``cv2.VideoCapture``
套了一层**只读代理**（``_PreviewCapture``）取图：``read()`` 原样透传，管线的像素
路径不变，预览失败也只打印一行提示、不影响重定向与仿真。

显示约定（两套场景都只做可视化，不代表物理接触效果）：

``--scene hands``：用 ``resetJointState`` 直接把角度写进 L21 关节（与 ``replay
l21-hand`` 的力矩控制不同），画面跟手不抖、不滞后；角度本身与离线管线完全一致
（同一套 ``L21HandAngles`` 与 ``L21HandAdapter``）。

``--scene robots``（默认）：把同一份 18 维角度经 ``robots/native_hand.py`` 降维
映射到三台机器人**原装**的灵巧手，用 ``POSITION_CONTROL`` 驱动，每帧推进
``--sim-steps`` 步物理（默认 8 步 x 1/240 s = 33 ms，约等于一帧实时数据）。
机器人手臂不参与（重定向不输出手臂关节角），锁定在中性姿态静止。覆盖率、被限位
截断的次数、仿真帧数都写进 ``--report`` 的 ``scene_summary``。

退出行为：``MediaPipeCameraInput.release()`` 里的 ``landmarker.close()`` 在本机
实测要 **42.2 s**（TFLite/XNNPACK 释放，``cv2`` 只要 0.3 s），PyBullet 在 GUI 客户端
已经死掉时 ``disconnect()`` 也会挂住，两者看起来都像卡死。因此本入口把所有收尾动作
丢到后台守护线程（每个最多等 ``RELEASE_TIMEOUT_SECONDS``），打印完统计并写完
``--report`` 后由 ``__main__`` 直接 ``os._exit`` 结束进程：Ctrl+C 或关窗口后立刻回到
命令行，不会再等那 42 s。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

from teleoperation.contracts.hand import L21HandAngles
from teleoperation.data.urdf import parse_joint_limits
from teleoperation.paths import (
    DEFAULT_REALTIME_CHECKPOINT,
    L21_ROOT,
    PROJECT_ROOT,
)
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM, RUNTIME
from teleoperation.robots.l21 import L21_JOINT_ORDER, MAPPING_18, L21HandAdapter

SIDES = ("left", "right")
URDF_NAME = "linkerhand_l21_{side}.urdf"
ASSET_NAME = "hand_landmarker.task"
# 双手并排摆位；单只手时放在原点前方（与 replay l21-hand 保持一致的观感）。
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
    candidates.extend([PROJECT_ROOT / ASSET_NAME, PROJECT_ROOT / "outputs" / ASSET_NAME])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"MediaPipe model asset not found; searched: {searched}. Download the "
        f"official hand_landmarker.task (float16, about 7.5 MB) and put it in one "
        f"of those paths, or pass --model-asset-path."
    )


def build_retargeter(backend: str, checkpoint: Path, device: str, calibration_frames: int,
                     smoothing: float):
    """Return ``(retargeter, backend_name, note)`` for the requested backend.

    ``auto`` prefers the trained checkpoint and falls back to the geometric
    backend when the file is missing, so the realtime loop always starts with a
    printed reason instead of a stack trace.
    """
    from teleoperation.retargeting.hand.geometric import GeometricHandRetargeter

    checkpoint = Path(checkpoint)
    if backend == "checkpoint":
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        return _checkpoint_retargeter(checkpoint, device)
    if backend == "geometric":
        return (
            GeometricHandRetargeter(calibration_frames=calibration_frames, smoothing=smoothing),
            "geometric",
            "forced by --backend geometric",
        )
    if checkpoint.is_file():                      # auto + checkpoint available
        return _checkpoint_retargeter(checkpoint, device)
    return (
        GeometricHandRetargeter(calibration_frames=calibration_frames, smoothing=smoothing),
        "geometric",
        f"auto fallback: checkpoint missing ({checkpoint}), using the zero-weight backend",
    )


def _checkpoint_retargeter(checkpoint: Path, device: str):
    """Wrap the teammate's trained palm-local model behind ``HandRetargeter``."""
    from teleoperation.apps.hand_realtime import load_palm_local_retargeter
    from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter

    model = load_palm_local_retargeter(checkpoint, device)
    return (
        PoseTransformerRetargeter(model, device),
        "checkpoint",
        f"palm_local checkpoint {checkpoint}",
    )


class RealtimeL21Scene:
    """Minimal PyBullet scene holding 1-2 L21 hands, driven by 18D angles.

    Only the joints listed in ``MAPPING_18``/``L21_JOINT_ORDER`` are written; any
    extra URDF joints (if a model ever has them) stay at their loaded state.
    """

    def __init__(self, sides, gui: bool = True):
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
        self.joint_index = {}
        self.adapters = {}
        for side in self.sides:
            directory = L21_ROOT / side
            urdf = directory / URDF_NAME.format(side=side)
            if not urdf.is_file():
                raise FileNotFoundError(f"L21 model not found: {urdf}")
            p.setAdditionalSearchPath(str(directory), physicsClientId=self.cid)
            base = list(BASE_POSITIONS[side]) if len(self.sides) > 1 else [0.0, 0.0, 0.3]
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
            # L21HandAdapter 的约定：joints_order 必须是 18 项、第 0 项为占位的
            # MAPPING_18（它内部用 values[index-1] 取值）。传 17 项的
            # L21_JOINT_ORDER 会让所有维度错位一格（探针实测到过）。
            self.adapters[side] = L21HandAdapter(list(MAPPING_18), parse_joint_limits(urdf))
        if self.gui:
            # 20 cm 的手在默认视角下几乎看不见，直接把镜头推到手上。
            p.resetDebugVisualizerCamera(
                cameraDistance=0.7 if len(self.sides) > 1 else 0.55,
                cameraYaw=40, cameraPitch=-20,
                cameraTargetPosition=[0.0, 0.0, 0.3], physicsClientId=self.cid,
            )
        self.applied = {side: 0 for side in self.sides}
        self.loss_reported = False

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

    def apply(self, side: str, angles: np.ndarray) -> None:
        """Write one 18D angle vector into this side's joints (instant, visual)."""
        if side not in self.bodies or not self.alive():
            return
        dofs = L21HandAngles(np.asarray(angles, dtype=np.float64)).native_mapping_dofs()
        targets = self.adapters[side].map(dofs)
        body = self.bodies[side]
        index = self.joint_index[side]
        for name, value in targets.items():
            self.p.resetJointState(body, index[name], float(value), physicsClientId=self.cid)
        self.applied[side] += 1

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


class CameraPreview:
    """摄像头窗口：原始画面 + 21 点骨架 + 一行状态（纯显示，像素不回灌算法）。

    画的是**原始**归一化关键点乘上图像尺寸，因此这个窗口同时是"MediaPipe 到底看没
    看到手"的现场证据。镜像显示时关键点 x 同步翻转，骨架仍贴合画面。
    """

    def __init__(self, sink, sides, *, show_window=True, mirror=True,
                 dump_dir=None, dump_frames=0, window_name=PREVIEW_WINDOW):
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
                self._text(canvas, f"{side} angles: 18 dims", (10, height - 10 - 20 * index),
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
                            dump_frames=args.preview_dump_frames)
    dump = (""
            if preview.dump_dir is None
            else f", dumping first {preview.dump_frames} frames to {preview.dump_dir}")
    print(f"camera window: {'on' if preview.show else 'off'}"
          f"{' (mirrored)' if preview.mirror else ''}{dump}", flush=True)
    return preview


def preview_hud(processed, marks, sides, samples, invalid, retargeter, raw_seen) -> str:
    """窗口左上角那行：帧数 / 帧率 / 原始检出 / 有效角度帧数 / 无效帧数 / 开手标定。"""
    raw = " ".join(f"{side}={raw_seen[side]}" for side in sides)
    counts = " ".join(f"{side}={len(samples[side])}" for side in sides)
    return (f"frame={processed} fps={_loop_fps(marks):.1f} raw={raw} valid={counts} "
            f"invalid={sum(invalid.values())} "
            f"calibration={getattr(retargeter, 'calibration_status', None)}")


def _describe(vectors) -> dict:
    """Frame count plus per-dimension observed range for one side."""
    array = np.asarray(vectors, dtype=np.float64).reshape(-1, HAND_ANGLE_DIM)
    return {
        "frames": int(array.shape[0]),
        "min": [float(value) for value in array.min(axis=0)],
        "max": [float(value) for value in array.max(axis=0)],
        "mapping": [MAPPING_18[dim] or "placeholder" for dim in range(HAND_ANGLE_DIM)],
    }


def build_scene(args, sides):
    """按 ``--scene`` 建显示目标；两套场景接口一致（apply / step / alive / close）。

    ``robots``（默认）把三台论文机器人并排摆开、驱动各自的**原装手**；
    ``hands`` 是旧行为，只摆 1-2 只 LinkerHand L21 手。两者都由同一份 18 维角度
    驱动，因此主循环不需要分叉。
    """
    if args.scene == "hands":
        return RealtimeL21Scene(sides, gui=not args.headless)
    from teleoperation.apps.three_robot_scene import ThreeRobotHandScene

    return ThreeRobotHandScene(sides, gui=not args.headless,
                               steps_per_frame=args.sim_steps)


def run(args) -> int:
    """Capture the camera until ``--frames`` or quit, retarget, and show the sim."""
    from teleoperation.apps.hand_realtime import MediaPipeCameraWorkflow

    asset = resolve_model_asset(args.model_asset_path)
    sides = SIDES if args.hand == "both" else (args.hand,)
    retargeter, backend, note = build_retargeter(
        args.backend, Path(args.checkpoint), args.device,
        args.calibration_frames, args.smoothing,
    )
    print(f"backend={backend} ({note})", flush=True)
    print(f"model_asset={asset}", flush=True)
    print(f"camera_index={args.camera_index} hands={','.join(sides)} "
          f"display={'headless(DIRECT)' if args.headless else 'PyBullet GUI'}", flush=True)

    samples = {side: [] for side in sides}
    invalid = {side: 0 for side in sides}
    raw_seen = {side: 0 for side in sides}    # MediaPipe 原始检出帧数（含窗口未攒够的帧）
    latest = {side: None for side in sides}   # 本帧 18D 角度，只给预览窗口显示用
    scene = None
    camera = None
    preview = None
    processed = 0
    marks = []
    elapsed, final = 0.0, {}
    try:
        scene = build_scene(args, sides)
        if args.scene == "robots":
            print(f"scene=three_robots: H1-2 / GR1-T2 / G1 side by side, native hands, "
                  f"{args.sim_steps} physics steps per frame.", flush=True)
        else:
            print(f"scene=hands: L21 models {', '.join(sides)}.", flush=True)
        if hasattr(scene, "frame_scene"):
            # 并排三台的镜头默认已框住整机；这里让 --view front / full 生效。
            scene.frame_scene(args.view)
        print(f"keep your hands in front of the camera and hold them open for about "
              f"{args.calibration_frames} frames to finish the open-hand calibration.",
              flush=True)
        # 刻意不用 MediaPipeCameraWorkflow 的 with 语句：它的 __exit__ 会阻塞在
        # landmarker.close() 上（实测 42 s，见模块 docstring 的"退出行为"）。
        camera = MediaPipeCameraWorkflow(asset, args.camera_index,
                                        args.width, args.height, args.fps)
        preview = build_preview(args, camera, sides)
        if not args.headless:
            print(PREVIEW_HINT_ROBOTS if args.scene == "robots" else PREVIEW_HINT_HANDS,
                  flush=True)
        started = time.perf_counter()
        while True:
            if args.frames and processed >= args.frames:
                break
            if not scene.alive():
                print("PyBullet window closed; stopping.", flush=True)
                break
            window = camera.next_window()
            processed += 1
            raw_frame = camera.last_raw_frame
            for side in sides:
                if raw_frame is not None and raw_frame.get(side) is not None:
                    raw_seen[side] += 1
            if window is None:
                # 三帧窗口还没攒够（或刚丢手）：预览里不留上一帧的角度，避免误会。
                latest = {side: None for side in sides}
            else:
                result = retargeter.retarget(window)
                for side in sides:
                    angles = getattr(result, side + "_angles")
                    if angles is None:
                        invalid[side] += 1
                        latest[side] = None
                        continue
                    values = np.asarray(angles.values, dtype=np.float64)
                    if values.shape != (HAND_ANGLE_DIM,) or not np.isfinite(values).all():
                        raise ValueError(f"invalid {side} angles: shape={values.shape}")
                    scene.apply(side, values)
                    samples[side].append(values)
                    latest[side] = values
            if preview is not None:
                preview.update(camera.last_raw_frame, latest,
                               preview_hud(processed, marks, sides, samples, invalid,
                                           retargeter, raw_seen))
                if preview.quit_requested:
                    print("camera window closed; stopping.", flush=True)
                    break
            scene.step()
            marks.append(time.perf_counter())
            if args.print_every and processed % args.print_every == 0:
                counts = " ".join(f"{side}={len(samples[side])}" for side in sides)
                print(f"frame={processed} fps={_loop_fps(marks):.1f} valid={counts} "
                      f"invalid={invalid} calibration="
                      f"{getattr(retargeter, 'calibration_status', None)}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        if preview is not None:
            _drain_in_background(preview.close, "camera preview close")
        if camera is not None:
            _drain_in_background(camera.release, "camera release")
        if scene is not None:
            final = {side: scene.joint_values(side) for side in sides}
            _drain_in_background(scene.close, "PyBullet disconnect")

    fps = _loop_fps(marks)

    print(f"\nprocessed={processed} frames in {elapsed:.2f} s  fps={fps:.1f}  "
          f"median frame interval {_median_interval(marks) * 1000:.1f} ms")
    if hasattr(scene, "summary"):
        info = scene.summary()
        print(f"scene={info['scene']} sim_frames={info['sim_frames']} "
              f"applied={info['applied_frames']} "
              f"clipped={info['clipped_dimensions']}", flush=True)
    if preview is not None:
        print(f"camera preview: window={'on' if preview.show else 'off'} "
              f"mirror={'on' if preview.mirror else 'off'} dumped_png={preview.dumped}"
              f"{'' if not preview.note else ' note=' + preview.note}", flush=True)
    report = {
        "backend": backend,
        "backend_note": note,
        "model_asset": str(asset),
        "camera_index": args.camera_index,
        "hand": args.hand,
        "headless": bool(args.headless),
        "scene": args.scene,
        "scene_summary": (scene.summary() if hasattr(scene, "summary") else None),
        "preview": None if preview is None else {
            "window": bool(preview.show),
            "mirrored": bool(preview.mirror),
            "dumped_png": int(preview.dumped),
            "dump_dir": None if preview.dump_dir is None else str(preview.dump_dir),
            "note": preview.note,
        },
        "processed_frames": processed,
        "elapsed_seconds": round(float(elapsed), 3),
        "fps": round(float(fps), 2),
        "median_frame_interval_ms": round(float(_median_interval(marks) * 1000), 2),
        "sides": {
            side: {
                "raw_detected_frames": raw_seen[side],
                "valid_frames": len(samples[side]),
                "invalid_frames": invalid[side],
                "angles": _describe(samples[side]) if samples[side] else None,
                "final_joint_values": final.get(side),
            }
            for side in sides
        },
    }
    for side in sides:
        entry = report["sides"][side]
        print(f"  {side}: raw={entry['raw_detected_frames']}/{processed} "
              f"valid={entry['valid_frames']} invalid={entry['invalid_frames']}")
        if entry["angles"]:
            print(f"    angle min (18D): {[round(value, 3) for value in entry['angles']['min']]}")
            print(f"    angle max (18D): {[round(value, 3) for value in entry['angles']['max']]}")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {path}")
    if not any(samples[side] for side in sides):
        if any(raw_seen[side] for side in sides):
            print("[FAIL] MediaPipe detected a hand but produced no angles: put the whole hand "
                  "in the middle of the frame and hold it open for about "
                  f"{args.calibration_frames} frames to finish calibration (the camera window "
                  "shows raw detection and calibration side by side).")
        else:
            print("[FAIL] no raw hand detection at all: check the camera window - the hand must "
                  "really be visible in the image (lighting, distance, not only at the frame "
                  "edge).")
        return 1
    return 0


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
    ``__main__`` ends the process with ``os._exit`` right after the summary, so a
    leftover teardown cannot delay the command prompt (the OS reclaims the camera,
    the window and the memory anyway).
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


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser; ``main`` also accepts an already parsed namespace."""
    parser = argparse.ArgumentParser(
        prog="python -m teleoperation.apps.realtime_hand_sim",
        description="camera -> palm-local alignment -> hand retargeting -> L21 PyBullet",
    )
    parser.add_argument("--model-asset-path", type=Path, default=None,
                        help="hand_landmarker.task path (default: repo root, else outputs/)")
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=("auto", "geometric", "checkpoint"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_REALTIME_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=RUNTIME.camera_device)
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--scene", choices=("robots", "hands"), default="robots",
                        help="robots: H1-2 / GR1-T2 / G1 side by side with their native "
                             "hands (default); hands: the LinkerHand L21 hands only")
    parser.add_argument("--sim-steps", type=int, default=8,
                        help="physics steps per frame for --scene robots (default 8, "
                             "about one 30 Hz frame at a 1/240 s step)")
    parser.add_argument("--view", choices=("full", "front"), default="full",
                        help="camera for --scene robots: full = angled, front = head on")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames; 0 runs until Ctrl+C or window close")
    parser.add_argument("--headless", action="store_true",
                        help="PyBullet DIRECT mode: no window, useful for self checks")
    parser.add_argument("--calibration-frames", type=int, default=15)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--no-preview", action="store_true",
                        help="do not open the camera window (PyBullet window only)")
    parser.add_argument("--preview-no-mirror", action="store_true",
                        help="show the camera image unmirrored (default is mirror-like)")
    parser.add_argument("--preview-dump-dir", type=Path, default=None,
                        help="also write the first --preview-dump-frames annotated frames here")
    parser.add_argument("--preview-dump-frames", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--report", type=Path, default=None,
                        help="write a JSON summary (frame counts, ranges, fps)")
    return parser


def main(args=None) -> int:
    if args is None:
        args = build_parser().parse_args()
    else:
        # 若将来把这个入口注册进统一 CLI（cli/** 是队友文件，当前不改），
        # 这里会补齐它没声明的可选参数，保证两条调用方式行为一致。
        for name, value in vars(build_parser().parse_args([])).items():
            if not hasattr(args, name):
                setattr(args, name, value)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nCapture stopped by user.", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary reports instead of raising
        print(f"[FAIL] realtime hand sim stopped: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    # 跳过 MediaPipe 约 42 s 的 TFLite 释放，Ctrl+C / 关窗口后立刻回到命令行。
    # 统计/报告都在 main() 里打印并写入，所以先跑完 main() 再 flush，最后 os._exit。
    # main() 本身仍返回退出码，方便被 import 调用（测试、探针）。
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
