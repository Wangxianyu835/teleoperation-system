"""Camera window and overlay display; pixels never feed retargeting."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np


PREVIEW_WINDOW = "camera: MediaPipe 21-landmark input (q/Esc quits)"


PREVIEW_TIPS = (4, 8, 12, 16, 20)


PREVIEW_HINT_ROBOTS = (
    "two windows: PyBullet shows H1-2 / GR1-T2 / G1 side by side with their native "
    "hands, the camera window shows your hand; press q/Esc in the camera window to stop.")


PREVIEW_HINT_HANDS = (
    "two windows: PyBullet shows the L21 hand, the camera window shows your hand; "
    "press q/Esc in the camera window to stop.")


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
