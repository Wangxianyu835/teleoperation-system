"""实时入口的采集窗口（手 + 手臂）：在基线预览上再叠一层手臂骨架与每侧数据行。

它和 ``applications/realtime_hand_sim.py`` 的关系
-------------------------------------------------
基线的 ``CameraPreview`` 只画 **21 点手部骨架**（那个入口不驱动手臂），而
``realtime_teleop`` 同一帧里还拿到了 pose 的肩/肘/腕。要让"一个窗口看到手和手臂"
成立，这里**不改基线文件**，而是子类化它的预览、把额外那一层画上去：

    CameraPreview（基线：取画面 / 镜像 / 挪窗口 / 置顶 / 存 PNG / q|Esc 退出）
      `- ArmCameraPreview.annotate()   <- 本文件：手臂折线 + 每侧数据行 + 图例

数据由调用方（``realtime_teleop``）准备：

``arms``  每侧 (3, 3) 归一化肩/肘/腕（``UpperBodyObservation.arms``，画图时乘图像尺寸）
``lines`` 每侧已格式化好的一行文字（来源字母、相对中性姿势的三轴位移厘米、肘角）

两者都可以缺：缺哪侧就不画哪侧，窗口其余部分照常，因此丢帧时画面不会前后不一致。

用法（参数都在 ``realtime_teleop``，本文件不新增 CLI）::

    E:\\python3.11.7\\python.exe -m teleoperation.simulation.realtime_teleop
    ... --no-arm-overlay      # 回到基线那种"只有手"的窗口

约定
----
- 本层**只画**：像素不回灌重定向；开不出窗口（没桌面 / 没 HighGUI）时只打印一行 note，
  主链路与 ``--report`` 不受影响（与基线预览同一套失败策略）。
- 非有限（NaN / Inf）或形状不对的手臂点一律当"没有这一侧"，不会拿它去 ``cv2.line``；
  这条与 ``RealtimeArmBridge`` 的"丢帧保持"口径一致 —— 看不见就不画，也不猜。
- 画的是 pose **归一化坐标**在画面上的投影，不是三维重建：本层没有相机内参、没有手眼标定，
  所以折线的绝对位置只能当"MediaPipe 到底看没看到这条手臂"的现场证据。
- ``.py`` 里不出现非 GBK 字符（``->`` 不写箭头，``<=`` 不写数学符号）。
"""

from __future__ import annotations

import numpy as np

from teleoperation.applications.realtime_hand_sim import (
    CameraPreview,
    attach_preview_capture,
)

#: 窗口标题：与基线区分开，用户一眼能看出"这条预览额外画了手臂"。
ARM_PREVIEW_WINDOW = "camera: hands + arms (q/Esc quits)"
#: 手臂折线：肩 -> 肘 -> 腕（``UpperBodyObservation`` 的点序，见 inputs/mediapipe.py）。
ARM_EDGES = ((0, 1), (1, 2))
#: 一条手臂有几个点（肩/肘/腕）—— 提示"有几个点在画面外"时用。
ARM_POINT_COUNT = 3
#: 把腕画大一点、颜色与手指尖一致，方便看出"这条线到哪结束"。
ARM_TIPS = (2,)
#: 手臂用比手部骨架浅一档的同色系，两条骨架叠在一起时仍分得清。
ARM_COLORS = {"left": (255, 214, 170), "right": (170, 245, 185)}
#: 底部图例：解释顶部 HUD 里的来源字母（与 realtime_teleop.SOURCE_SHORT 同一套口径）。
ARM_LEGEND = "arm: P=pose arm H=hand fallback -=held (no arm data this frame)"


def finite_arm_points(arms, sides) -> dict:
    """``{side: (3, 3)}`` -> ``{side: (3, 3) 或 None}``：形状不对 / 含 NaN 的一侧给 ``None``。

    只做"能不能画"的判断，不做任何几何处理：点的含义（哪个是肩、单位是什么）由输入层决定。
    """
    result = {}
    for side in sides:
        points = None if arms is None else arms.get(side)
        try:
            usable = (np.shape(points) == (3, 3)
                      and bool(np.isfinite(np.asarray(points, dtype=np.float64)).all()))
        except (TypeError, ValueError):
            usable = False
        result[side] = np.asarray(points, dtype=np.float64) if usable else None
    return result


def outside_frame(pixels, width, height) -> int:
    """``(N, 2)`` 像素里落在画布外的个数 —— 用来提示"手臂没进画面"（实测常见）。"""
    array = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    inside = ((array[:, 0] >= 0) & (array[:, 0] < width)
              & (array[:, 1] >= 0) & (array[:, 1] < height))
    return int((~inside).sum())


class ArmCameraPreview(CameraPreview):
    """基线预览 + 手臂层：同一帧里 21 点手骨架与肩/肘/腕折线一起画。

    ``overlay=False`` 时行为与基线预览逐像素一致（``--no-arm-overlay`` 用它做逃生门），
    因此上层只需要一条代码路径。
    """

    def __init__(self, *args, overlay: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlay = bool(overlay)
        self.arm_points = {}
        self.arm_lines = {}

    def update(self, raw_frame, angles, hud, arms=None, lines=None):
        """先记下这一帧的手臂数据，再交给基类的 ``update``（它内部会调 ``self.annotate``）。

        ``arms`` 在这里统一过一遍 :func:`finite_arm_points`：形状不对 / 含 NaN 的一侧
        当作"没有"，所以调用方（``realtime_teleop``）不用自己挑，也不会有人拿 NaN 去画线。
        """
        self.arm_points = finite_arm_points(arms, self.sides)
        self.arm_lines = {} if lines is None else dict(lines)
        return super().update(raw_frame, angles, hud)

    def annotate(self, raw_frame, angles, hud):
        """基类画完手与 HUD 之后，把手/臂两侧的数据补上（纯显示，返回值只用于窗口/PNG）。"""
        canvas = super().annotate(raw_frame, angles, hud)
        if canvas is None or not self.overlay:
            return canvas
        height, width = canvas.shape[:2]
        offscreen = {}
        for index, side in enumerate(self.sides):
            color = ARM_COLORS.get(side, (225, 225, 225))
            points = self.arm_points.get(side)
            if points is not None:
                pixels = self.pixels(points, width, height)
                self._draw_skeleton(canvas, pixels, ARM_EDGES, color, ARM_TIPS, False)
                self._text(canvas, f"{side} arm",
                           (int(pixels[2][0]) + 8, int(pixels[2][1]) + 18), color, 0.5)
                offscreen[side] = outside_frame(pixels, width, height)
            line = self.arm_lines.get(side)
            if line:
                self._text(canvas, line, (10, 44 + 20 * index), color, 0.5)
        # 实测最常见的"看不见手臂"就是肩/肘/腕跑到画面外（人离相机太近）：说清几个点在外，
        # 用户就知道该往后退，而不是怀疑代码。
        outside = " ".join(f"{side} {count}/{ARM_POINT_COUNT}"
                           for side, count in sorted(offscreen.items()) if count)
        if outside:
            self._text(canvas, f"{outside} arm point(s) outside the frame "
                               f"(step back or tilt the camera)",
                       (10, height - 30 - 20 * len(self.sides)), (60, 220, 255), 0.5)
        self._text(canvas, ARM_LEGEND, (10, height - 10 - 20 * len(self.sides)),
                   (205, 205, 205), 0.45)
        return canvas


def build_arm_preview(args, camera, sides):
    """按 ``args`` 建"手 + 手臂"预览；不需要窗口且不存 PNG 时返回 ``None``。

    与基线 ``build_preview`` 同一套开关、同一套失败策略，只换窗口标题、类，并多透传
    ``--no-arm-overlay``；输入层没暴露采集对象时打印一行 note 并返回 ``None``。
    """
    want_window = not args.headless and not args.no_preview
    if not want_window and not args.preview_dump_dir:
        return None
    sink = attach_preview_capture(camera)
    if sink is None:
        print("note: the input layer does not expose a camera frame; preview skipped",
              flush=True)
        return None
    preview = ArmCameraPreview(
        sink, sides, show_window=want_window,
        overlay=not getattr(args, "no_arm_overlay", False),
        mirror=not args.preview_no_mirror,
        dump_dir=args.preview_dump_dir,
        dump_frames=args.preview_dump_frames,
        window_name=ARM_PREVIEW_WINDOW,
        window_pos=args.preview_window_pos,
        topmost=not args.preview_no_topmost)
    dump = ("" if preview.dump_dir is None
            else f", dumping first {preview.dump_frames} frames to {preview.dump_dir}")
    print(f"camera window: {'on' if preview.show else 'off'}"
          f"{' (mirrored)' if preview.mirror else ''}{dump}"
          f"{'' if preview.overlay else ' (arm overlay off)'}"
          f", title={preview.window_name!r}"
          f"{'' if preview.window_pos is None else f', position={preview.window_pos}'}",
          flush=True)
    return preview
