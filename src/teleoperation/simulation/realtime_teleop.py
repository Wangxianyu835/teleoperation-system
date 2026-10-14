"""实时遥操作入口：摄像头 -> 手（18 维）+ 手臂（相对重定向）-> H1-2 / GR1-T2 / G1。

它和 ``applications/realtime_hand_sim.py`` 的关系
------------------------------------------------
那个入口只驱动**原装手**：三台机器人的手臂被 ``three_robot_scene.py`` 每帧按住
（``hold_ids`` + HOLD_FORCE=500），所以画面里手会动、手臂不动；而且它取的
``MediaPipeCameraWorkflow.next_window()`` 只返回手部窗口，同一帧的 pose 肩/肘/腕在那一层
就被丢掉了。本入口在那条链路旁边补上手臂：

    MediaPipeCameraInput.next_teleop_observation()   # 同一帧：手 21 点 + pose 肩/肘/腕
      -> 手：MediaPipePalmLocalPipeline（三帧窗口 + 掌面局部对齐）
             -> GeometricHandRetargeter / 训练检查点 -> 18 维 -> scene.apply(side)
      -> 臂：RealtimeArmBridge（相对重定向：以中性姿势为原点）
             -> ArmSceneDriver.apply_each(source="base") -> PyBulletArmIK 两遍热启动 -> 7 关节

**零改动**：``applications/**``、``inputs/**``、``retargeting/**`` 一行都没改，这里只是
import 它们的公开接口并复用基线入口里那几个纯工具函数；手臂侧三个模块
（``simulation/arm_profiles.py`` / ``arm_ik.py`` / ``arm_scene_driver.py``）是我们自己的新文件。

用法::

    # PyCharm 一键（推荐）：解释器/PYTHONPATH/工作目录/缓存四项都替你补齐
    E:\\python3.11.7\\python.exe devtools\\run_realtime_arm_windows.py

    # 命令行等价
    cd F:\\simulation_platform_cs
    $env:PYTHONPATH = "$pwd\\src"
    E:\\python3.11.7\\python.exe -m teleoperation.simulation.realtime_teleop

    # 无头自检 + 报告 + 把这一场的原始帧录下来（可以离线复现）
    E:\\python3.11.7\\python.exe -m teleoperation.simulation.realtime_teleop \\
        --headless --frames 300 --report outputs\\tmp_realtime\\arm_report.json \\
        --dump-npz outputs\\tmp_realtime\\frames.npz

    # 回到基线行为：只驱动手
    E:\\python3.11.7\\python.exe -m teleoperation.simulation.realtime_teleop --no-arms

两个窗口
--------
1. **采集窗口**（``cv2.imshow``，标题见 ``arm_preview.ARM_PREVIEW_WINDOW``）：原始画面 +
   手 21 点骨架 + **手臂肩/肘/腕折线**（同色系浅一档）+ 每侧一行数字（来源字母
   P/H/-、相对中性姿势的三轴位移厘米、臂长、肘角）+ 底部图例。这一层由
   ``simulation/arm_preview.py`` 在基线预览上**子类化**画出来，基线文件一行未改；
   ``--no-arm-overlay`` 回到基线那种"只有手"的窗口。
2. **PyBullet 窗口**：H1-2 / GR1-T2 / G1 并排 —— 原装手由 18 维角度驱动、手臂由
   ``ArmSceneDriver`` 的 IK 结果驱动。它只看仿真结果，不看摄像头。

采集窗口里那层手臂是**显示投影**（pose 归一化坐标乘图像尺寸），不是三维重建：
本层没有相机内参、没有手眼标定，折线只能当"MediaPipe 到底看没看到这条手臂"的现场证据。

中性标定（第一次跑必看）
------------------------
手臂走的是**相对**重定向：启动后前 ``--arm-calibrate-frames`` 帧要**保持一个中性姿势不动**
（建议双手抬到胸前、肘略弯，和你平时开始操作前的姿势一致），程序取这些帧的中位数作为
原点；之后你手腕相对肩的位移（米）按 ``profile.metric_scale``（再按你标定时的臂长归一化）
搬进机器人基座系。标定期间手臂停在各自的中性工作点，不会乱跑。``--save-neutral FILE``
可以存下这次的中性姿势，下次 ``--neutral-json FILE`` 直接复用，不用再站一次。

轴向约定（要现场核对一次）
--------------------------
操作者系（MediaPipe pose world）：x 图像向右、y 向下、z 远离相机为正；机器人基座系：
X 前、Y 左、Z 上。默认映射见 ``arm_realtime.AxisMap``（``(Z, X, Y)/(-, +, -)``）。

**第一次跑请加 ``--arm-trace``**：每 ``--print-every`` 帧会打印 ``delta_op_cm``（你手腕
相对肩、相对中性姿势的位移，厘米）。你只往一个方向动手时，只有一个分量明显变化，符号
也应与画面方向一致；对不上就用 ``--arm-axis-order`` / ``--arm-axis-signs`` 覆盖。
**这一步没做过之前，"跟手"只是相对自洽，不能称作标定。**

退出码与报告
------------
0 = 跑完且至少一侧给出过输出（手角度或手臂目标）；1 = 全程既没有手也没有手臂数据。
``--report`` 的 ``arm`` 段给出每台机器人的稳态残差、行程、被 hold 的关节漂移；
``bridge`` 段给出轴向约定、gain、行程上限、中性记录与各来源帧数。

如实声明
--------
这里**没有**相机内参、没有手眼标定，也**不是**论文的 SMPLer-X + PINK 链路：位置来自
相对重定向的调参，姿态约定只在 ``ik_mode="pose"`` 的机器人（h1_2）上使用，
``gr1_t2`` / ``g1`` 只做位置 IK。每台机器人每侧的行程上限 ``profile.realtime_travel_m``
是实测出来的（``devtools\\verify_realtime_arm_bridge.py --probe-travel``），超出即被夹住。

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.applications.realtime_hand_sim import (
    PREVIEW_HINT_ROBOTS,
    _describe,
    _drain_in_background,
    _loop_fps,
    _median_interval,
    _window_pos,
    build_retargeter,
    build_scene,
    resolve_model_asset,
)
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.paths import DEFAULT_POSE_ASSET, realtime_checkpoint
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM, RUNTIME
from teleoperation.simulation.arm_preview import (
    ARM_PREVIEW_WINDOW,
    build_arm_preview,
)
from teleoperation.simulation.arm_profiles import loadable_robots, profile_for
from teleoperation.simulation.arm_ik import (ELBOW_PASS_ORDERS, ELBOW_RECOVERY_COOLDOWN_FRAMES,
                                             ELBOW_RECOVERY_DEG, ELBOW_RECOVERY_MARGIN_DEG)
from teleoperation.simulation.arm_realtime import (
    DEFAULT_LATERAL_GUARD_M,
    MAP_MODES,
    MAX_DIRECTION_RATE_DEG_S,
    TRAVEL_MODES,
    AxisMap,
    RealtimeArmBridge,
    load_neutral,
    save_neutral,
)
from teleoperation.simulation.arm_scene_driver import SLEW_MAX_STEP_M, ArmSceneDriver

SIDES = ("left", "right")
ARM_ENTRY = "realtime_teleop"
#: 摄像头循环名义帧长（秒）：第一帧还没有"上一帧"可比，斜坡按它换算，
#: 这样第一帧也是 20 mm/帧（旧口径），不会退化成"一步跳到工作点"。
NOMINAL_FRAME_DT = 1.0 / 30.0
#: 斜坡默认速率（mm/s）：600 mm/s 在 30 fps 下正好是 20 mm/帧（与 2026-10-13 实测口径同速），
#: 现场 10-12 fps 下按帧换算成 50-60 mm/帧、再被 SLEW_MAX_STEP_M（40 mm）压到 40 mm/帧 ——
#: 40 mm 是实测拐点（横摆传递 0.80，再快没有改善而残差变大）。见 docs/multi-robot-arm.md §8.9。
DEFAULT_ARM_SLEW_MM_PER_S = 600.0
#: HUD 里把来源缩成一个字母：P = pose 手臂（主），H = 手部兜底，- = 无数据（保持上一帧）。
SOURCE_SHORT = {"pose": "P", "hand": "H", "none": "-"}


class CameraFrameSource:
    """一帧 = 手部三帧窗口 + pose 肩/肘/腕：把基线工作流丢掉的那一半补回来。

    ``MediaPipeCameraWorkflow.next_window()`` 只把手交给下游（它调的是不含 pose 的
    ``next_observation``），而 ``inputs/mediapipe.py`` 的 ``next_teleop_observation()``
    一次采集就同时给出两套数据。基线文件不能改，所以这里复刻那 20 行胶水（同样的相对秒
    时间戳、同样的 ``MediaPipePalmLocalPipeline`` 与 ``world_*`` 字段），额外把
    ``upper_body``（pose 手臂）原样带出来给 ``RealtimeArmBridge``。

    没有配 pose 模型时自动退化为基线行为（``next_observation``），手臂链路由上层关掉。
    """

    def __init__(self, model_asset_path, camera_index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 30, landmark_space: str = "world",
                 pose_model_asset_path=None, pose_min_visibility=None):
        self.model_asset_path = Path(model_asset_path)
        self.landmark_space = landmark_space
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space)
        self.processor = self.pipeline.processor
        self.last_raw_frame = None
        self.last_observation = None
        self.pose_enabled = pose_model_asset_path is not None
        extra = ({} if pose_min_visibility is None
                 else {"pose_min_visibility": float(pose_min_visibility)})
        self._input = MediaPipeCameraInput(
            model_asset_path=model_asset_path, camera_index=camera_index,
            width=width, height=height, fps=fps, invalid_hand_as_missing=True,
            pose_model_asset_path=pose_model_asset_path, **extra)
        self._start_time = time.time()

    def next_frame(self):
        """采集一帧；返回手部三帧窗口（还没攒够就是 ``None``）。

        ``last_raw_frame`` 是这一帧的手部（归一化 + ``world_*``），``last_observation``
        是同一帧的完整观测（含 pose 手臂），两者同源且同一时间戳。
        """
        if self.pose_enabled:
            observation = self._input.next_teleop_observation()
            self.last_observation = observation
            raw = observation.hands
        else:
            raw = self._input.next_observation(include_world=True)
            self.last_observation = None
        raw_frame = {**raw.hands, "timestamp": raw.timestamp,
                     "metadata": dict(raw.metadata)}
        self.last_raw_frame = raw_frame
        frame = {
            **raw_frame,
            "timestamp": round(time.time() - self._start_time, 3),
            "metadata": {
                **raw_frame.get("metadata", {}),
                "raw_timestamp_ms": raw_frame["timestamp"],
                "timestamp_unit": "relative_seconds",
            },
        }
        if raw.world_hands is not None:
            frame.update({"world_" + side: raw.world_hands[side] for side in SIDES})
        return self.pipeline.process_window(frame)

    def arms(self):
        """同一帧的 pose 肩/肘/腕（``UpperBodyObservation``）；没有就是 ``None``。"""
        observation = self.last_observation
        return None if observation is None else observation.upper_body

    def release(self, **kwargs):
        """转发给输入层；``close_landmarker=False`` 可让 42 s 的 TFLite 释放不挡退出。"""
        return self._input.release(**kwargs)



class ReplayFrameSource:
    """把 ``--dump-npz`` 录下来的原始帧当成"摄像头"再放一遍：不需要相机，可复现。

    键名与 ``--dump-npz`` 写出的一致：``left_hand``/``right_hand``（归一化 21x3）、
    ``world_left_hand``/``world_right_hand``（米制 21x3）、``left_arm``/``right_arm``
    （pose 米制肩/肘/腕 3x3）、``timestamps``；NaN 表示那一帧没检到（会触发"保持上一帧"）。
    手部仍然走同一条 ``MediaPipePalmLocalPipeline``，所以三帧窗口语义和实时一致。
    """

    def __init__(self, path, landmark_space: str = "world"):
        # 一次性读进内存：np.load 的句柄会锁住文件（Windows 上连删临时目录都不行），
        # 而这里的帧本来就只有几十 MB 级，读完之后不被文件句柄牵着走更安全。
        with np.load(str(path)) as handle:
            self.data = {key: np.asarray(handle[key]) for key in handle.files}
        self.frames = int(np.asarray(self.data["timestamps"]).shape[0])
        self.landmark_space = landmark_space
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space)
        self.processor = self.pipeline.processor
        self.last_raw_frame = None
        self.last_observation = None
        self.pose_enabled = True
        self.exhausted = False
        self.index = 0
        self._arms = None

    def _points(self, key, index, shape):
        if key not in self.data:
            return None
        values = np.asarray(self.data[key][index], dtype=np.float64)
        if values.shape != tuple(shape) or not np.isfinite(values).all():
            return None
        return values

    def next_frame(self):
        """下一帧的手部窗口；录完了就把 ``exhausted`` 置 True 并返回 ``None``。"""
        if self.index >= self.frames:
            self.exhausted = True
            return None
        index, self.index = self.index, self.index + 1
        timestamp = float(np.asarray(self.data["timestamps"])[index])
        # 元数据不是装饰：队友的处理器在 world 口径下会校验
        # ``world_landmark_space`` / ``world_length_unit`` / ``source_scale``
        # （retargeting/hand/processing.py 的 process_frame），而 --dump-npz 只存数组，
        # 所以回放时按 inputs/mediapipe.py 的同一口径把这些声明补回来；
        # 少一项整场在第一帧就 ValueError（这是实测踩到的，见 docs 第 8 节）。
        raw_frame = {"timestamp": timestamp,
                     "metadata": {"source": "npz_replay",
                                  "timestamp_unit": "relative_seconds",
                                  "source_landmark_space": "mediapipe_normalized",
                                  "world_landmark_space": "mediapipe_world_meters",
                                  "world_length_unit": "m",
                                  "world_origin": "hand_geometric_center",
                                  "source_scale": 1.0}}
        arms = {}
        for side in SIDES:
            raw_frame[side] = self._points(f"{side}_hand", index, (21, 3))
            raw_frame[f"world_{side}"] = self._points(f"world_{side}_hand", index, (21, 3))
            arms[side] = self._points(f"{side}_arm", index, (3, 3))
        # NaN（= 那帧没检到）照样报个原因，和实时链路 inputs/mediapipe.py 同一套口径。
        missing = {side: "npz replay: hand not detected in this frame"
                   for side in SIDES if raw_frame[side] is None}
        if missing:
            raw_frame["metadata"]["invalid_reasons"] = missing
        world_missing = {side: "npz replay: no world landmarks in this frame"
                         for side in SIDES if raw_frame[f"world_{side}"] is None}
        if world_missing:
            raw_frame["metadata"]["world_invalid_reasons"] = world_missing
        self.last_raw_frame = raw_frame
        self._arms = arms
        return self.pipeline.process_window(raw_frame)

    def arms(self):
        """这一帧的 pose 手臂，包装成 ``UpperBodyObservation``（米制放在 ``world_arms``）。"""
        if self._arms is None:
            return None
        from teleoperation.contracts.observations import UpperBodyObservation

        timestamp = float(self.last_raw_frame["timestamp"]) if self.last_raw_frame else None
        return UpperBodyObservation(
            {side: None for side in SIDES},
            {side: self._arms[side] is not None for side in SIDES},
            timestamp, "mediapipe_pose_normalized", "normalized",
            world_arms={side: self._arms[side] for side in SIDES})

    def release(self, **kwargs):
        """回放没有设备要关；保留接口以便和 ``CameraFrameSource`` 互换。"""
        return None


def build_parser() -> argparse.ArgumentParser:
    """所有开关；默认值都往"能直接跑起来"调（world 米制、两侧、三台全上）。"""
    parser = argparse.ArgumentParser(
        prog="python -m teleoperation.simulation.realtime_teleop",
        description="Realtime camera -> hand angles + arm targets -> H1-2 / GR1-T2 / G1")
    parser.add_argument("--backend", choices=("auto", "geometric", "checkpoint"),
                        default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=RUNTIME.camera_device)
    parser.add_argument("--model-asset-path", type=Path, default=None)
    parser.add_argument("--pose-model-asset-path", type=Path, default=None,
                        help="pose landmarker .task (default datasets/pose_landmarker_lite.task); "
                             "when it is missing the arms stay off and only the hands move")
    parser.add_argument("--pose-min-visibility", type=float, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--landmark-space", choices=("world", "normalized"), default="world")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--robot", action="append", choices=loadable_robots(), default=None,
                        help="drive only these robots (repeatable; default: all three)")
    parser.add_argument("--no-arms", action="store_true",
                        help="baseline behaviour: drive the hands only, arms stay held")
    parser.add_argument("--arm-orientation", choices=("off", "auto", "on"), default="off",
                        help="auto = keep the orientation constraint where the profile has one "
                             "(h1_2 only); off = position-only for every robot (default since "
                             "2026-10-13); on = force the constraint. Measured on a 1560 frame "
                             "live recording, same input, only this flag changed: h1_2 wrist "
                             "residual 95.8 / 133.3 mm -> 17.1 / 3.9 mm, orientation error "
                             "46.0 / 46.4 deg -> 1.5 / 0.4 deg, shoulder yaw travel 322.4 / "
                             "235.5 deg -> 209.6 / 68.7 deg (the 3-point wrist frame flips its "
                             "elbow-plane normal: 4-7%% of frames command a >60 deg step, max "
                             "~178 deg). g1 / gr1_t2 are position-only either way")
    parser.add_argument("--arm-axis-order", default=None, metavar="ZXY",
                        help="robot axes taken from operator axes (default from arm_realtime)")
    parser.add_argument("--arm-axis-signs", default=None, metavar="-1,1,-1")
    parser.add_argument("--arm-gain", type=float, nargs=2, action="append", default=None,
                        metavar=("ROBOT", "GAIN"), help="override metric_scale for one robot")
    parser.add_argument("--arm-map", choices=MAP_MODES, default="chain",
                        help="how operator motion becomes a target. 'chain' (default, "
                             "unchanged) = wrist minus shoulder, one gain. 'segments' = the "
                             "paper's scale_joints: one term per bone, scaled by the robot's "
                             "own measured link lengths, which is also what the elbow task "
                             "needs. Measured 2026-10-13 on a synthetic 'fists at the chest, "
                             "only the upper arm swings' stream: the target moved 0.0 mm on "
                             "all three robots with 'chain' (that is why that motion gets no "
                             "reaction at all) vs 68-74 / 8-10 / 2 mm with 'segments'. It is "
                             "not a cure on its own: when a robot's two bone lengths have "
                             "almost the same ratio as yours (g1 0.193/0.184) the two terms "
                             "cancel again, and only the elbow task can move the shoulder. "
                             "See docs/multi-robot-arm.md 8.10. 'posture' (added 2026-10-13 evening, after the field verdict "
                              "that both displacement recipes do not look like the operator) "
                              "= move the DIRECTION instead of a displacement, with the "
                              "robot's own measured bone lengths: elbow = shoulder_anchor + "
                              "L_u*M(unit(你的上臂)), wrist = elbow + L_f*M(unit(你的前臂)). "
                              "Your 40 deg shoulder swing is then 40 deg on every robot, the "
                              "per-axis travel clamp does not apply, and both targets are "
                              "always reachable. See 8.11")
    parser.add_argument("--arm-elbow-task", action="store_true",
                        help="also hand the IK an elbow position target (the paper's elbow "
                             "task; needs --arm-map segments; --arm-map posture builds the "
                             "elbow target by itself, so this flag is not needed there). "
                             "Off by default. Measured "
                             "0.01-0.06 deg of upper-arm motion without it (the redundant "
                             "7 DoF chain keeps its configuration, which is why the upper "
                             "arm 'does not move') vs 7-33 deg with it. Cost: the wrist "
                             "residual grows when the two targets disagree, so try it and "
                             "watch --arm-trace / the report")
    parser.add_argument("--arm-elbow-pass", choices=ELBOW_PASS_ORDERS, default="last",
                        help="which IK pass carries the elbow target (default 'last': with "
                             "'first' the wrist-pose finishing pass pulls the elbow back "
                             "again, measured elbow error 141-256 mm)")
    parser.add_argument("--arm-elbow-gain", type=float, default=1.0, metavar="0-1",
                        help="how much of the upper-arm displacement to hand the elbow "
                             "target (default 1.0 = the whole term). PyBullet gives the two "
                             "position targets equal weight, so this is the only knob on the "
                             "trade: measured 2026-10-13 on a real recording (120 frames, "
                             "h1_2 right) 1.0 gave 83.98 deg of shoulder travel at 65.75 mm "
                             "wrist residual (ceiling 30) where the wrist-only run gave 49.93 "
                             "deg at 6.38 mm. Lower it to buy wrist accuracy back")
    parser.add_argument("--arm-posture-gain", type=float, default=1.0, metavar="0-1",
                        help="direction recipe only (--arm-map posture): 1.0 (default) = the "
                             "robot's upper arm / forearm point exactly where yours point; "
                             "0.5 = stop halfway between the robot's own neutral direction "
                             "and yours; 0 = stay at the robot's neutral. It is an angle "
                             "blend (slerp), not a displacement scale: your 40 deg shoulder "
                             "swing stays 40 deg on every robot either way")
    parser.add_argument("--arm-ik-scope", choices=("arm", "body"), default="arm",
                        help="which joints the IK may move (default 'arm'): 'arm' pins every "
                             "non-arm joint to its current angle (arm_ik._locked_arguments), "
                             "'body' lets PyBullet solve over *all* movable joints with "
                             "restPoses pulling them (the historical behaviour, kept for A/B). "
                             "Measured 2026-10-14 on outputs/tmp_realtime/live_long_frames.npz, "
                             "200 frames, --arm-map posture: 'arm' makes the forearm-reversed "
                             "fraction 0.0 on all six sides (body: 0.099 gr1_t2 left / 0.041 g1 "
                             "left / 0.006 gr1_t2 right), wrist residual 134.6 -> 21.2 mm "
                             "(gr1_t2 left) and 74.9 -> 34.8 (h1_2 left), forearm direction "
                             "error 66.0 -> 44.7 deg (gr1_t2 left); elbow jerk maxima rise "
                             "(13.1 -> 42.7 deg on h1_2 left) because the arm now actually "
                             "follows the operator, whose recorded direction jumps reach 153 deg "
                             "in one frame. 3 repeats of 'arm' gave identical numbers (opposite "
                             "fraction 0.0 on all six sides); the solver has 7 variables instead "
                             "of ~55 so it is not slower, but no speedup is claimed -- fps over "
                             "these runs varied 2.5-12.7 with machine load under both scopes")
    parser.add_argument("--arm-anchor", choices=("upper", "shoulder"), default="upper",
                        help="which FK point counts as the upper arm's shoulder end, used both "
                             "by the direction recipe's elbow/wrist targets and by the "
                             "shoulder-follow shift (default 'upper' = the elbow's parent link "
                             "origin, rigid to the elbow; 'shoulder' = the name-matched shoulder "
                             "joint link, the historical behaviour kept for A/B). Measured "
                             "2026-10-14 with devtools/probe_arm_segment_rigidity.py: the "
                             "name-matched point drifts 48.5-101.7 mm when shoulder_roll turns "
                             "(gr1_t2 left +101.6 mm) while the parent link origin is 0.00 mm, "
                             "so the old recipe asked for an elbow target off the reachable "
                             "shell and the solver had to compromise (the field 'GR1-T2 forearm "
                             "still reversed', 222 mm elbow residual)")
    parser.add_argument("--arm-no-shoulder-follow", action="store_true",
                        help="turn off shoulder following for the elbow/wrist targets (default: "
                             "on for the recipes that measure anchors). The commanded elbow is "
                             "'load-time shoulder anchor + upper-arm length * your direction', a "
                             "point fixed to the robot BASE; when the torso is pushed by the arm "
                             "reaction forces the distance from the CURRENT shoulder stops being "
                             "one upper arm, so the 7 DoF chain compromises and the elbow lands "
                             "somewhere else (the field 'bends the opposite way'). With it on, "
                             "both targets are shifted by 'current FK shoulder - anchor' every "
                             "frame. Use this flag to reproduce the old numbers")
    parser.add_argument("--arm-direction-rate-deg-s", type=float,
                        default=MAX_DIRECTION_RATE_DEG_S, metavar="DEG/S",
                        help="operator direction glitch guard (default %.0f deg/s, 0 = off). "
                             "A single-frame direction change faster than a human joint can "
                             "turn is a landmark jump, not your arm: measured on the recording "
                             "(outputs/tmp_realtime/live_long_frames.npz) the median step is "
                             "4.7-6.9 deg at ~10 fps but the MAX is 153 deg. The direction "
                             "recipe copies directions 1:1, so such a jump asks the IK to swing "
                             "the elbow 153 deg in one frame -- it either flips branch or jerks "
                             "(the field 'H1-2 twitches at the wrist'). Over the limit the "
                             "previous direction is held and counted in counts.direction_glitch"
                             % MAX_DIRECTION_RATE_DEG_S)
    parser.add_argument("--arm-elbow-recovery-deg", type=float, default=ELBOW_RECOVERY_DEG,
                        metavar="MM",
                        help="elbow branch recovery threshold in degrees (default %.0f, "
                             "0 = off). The IK starts from the current pose and pays for "
                             "distance to restPoses, so once a frame drops the elbow into the "
                             "flipped branch every later frame is solved from there and never "
                             "comes back -- the field complaint 'GR1-T2 bends the opposite way' "
                             "(200-frame replay, 2026-10-13: gr1_t2 left 91%% of frames with "
                             ">30 deg elbow error, 29%% with the forearm more than 90 deg away "
                             "from the asked direction; h1_2 right was 5 deg). Above this "
                             "threshold the same pass sequence is re-solved from the load "
                             "pose and the branch with the smaller forearm+0.5*upper direction "
                             "error (degrees) wins"
                             % ELBOW_RECOVERY_DEG)
    parser.add_argument("--arm-elbow-recovery-margin-deg", type=float,
                        default=ELBOW_RECOVERY_MARGIN_DEG, metavar="DEG",
                        help="hysteresis of the branch recovery in degrees (default %.0f, "
                             "0 = accept any improvement). Measured 2026-10-14: with a 1 deg "
                             "margin the recovery fired every ~6 frames (h1_2 left: 19 "
                             "recoveries in 123 commanded frames) and the field saw the whole "
                             "arm twitch; 8 deg keeps one branch" % ELBOW_RECOVERY_MARGIN_DEG)
    parser.add_argument("--arm-elbow-recovery-frames", type=int,
                        default=ELBOW_RECOVERY_COOLDOWN_FRAMES, metavar="N",
                        help="after any recovery attempt (accepted or rejected), wait N "
                             "frames before trying again (default %d, 0 = try every frame). "
                             "The accepted branch warm-starts from itself, so the cooldown "
                             "does not fall back -- it only stops the per-frame flipping and "
                             "the duplicate IK solves" % ELBOW_RECOVERY_COOLDOWN_FRAMES)

    parser.add_argument("--arm-travel-mm", type=float, default=None,
                        help="override the per-axis travel clamp for every robot (mm)")
    parser.add_argument("--arm-travel-mode", choices=TRAVEL_MODES, default="clip",
                        help="what to do when the requested wrist target exceeds the travel "
                             "limit. 'clip' = cut each axis on its own (default since "
                             "2026-10-13: measured on a live composite step of "
                             "(-9.3, +6.4, +24.5) cm, where only op_z is over the limit, it "
                             "keeps g1's lateral wrist travel at 3.44 cm / shoulder roll 5.5 "
                             "deg / elbow 7.1 deg). 'scale' = shrink the whole step so its "
                             "longest axis lands exactly on the limit, direction kept: one "
                             "over-limit axis (the jittery MediaPipe depth) then shrinks every "
                             "axis, which is what made the robot look like it only ever swung "
                             "up and down (same step: g1 lateral 1.31 cm, roll 2.0 deg, elbow "
                             "2.8 deg). See docs/multi-robot-arm.md 8.8")
    parser.add_argument("--arm-lateral-guard-mm", type=float,
                        default=1000.0 * DEFAULT_LATERAL_GUARD_M, metavar="MM",
                        help="keep each wrist target on its own side of the body: left wrist "
                             "y >= +MM, right wrist y <= -MM (**default 0 = off** since "
                             "2026-10-13 evening; 20 was the daytime default). It was added "
                             "because h1_2's neutral working point is only y=+-0.05 m while the "
                             "per-axis travel limit is 0.10 / 0.15 m, so the targets can legally "
                             "cross the midline and the two forearms touch (measured on a 1560 "
                             "frame session: gap down to 2.21 mm). Cost measured 2026-10-13: "
                             "swing the left hand +-9 cm and h1_2's lateral swing drops from "
                             "0.49 to 0.43 of the operator's, and a 9.0 cm inward request lands "
                             "at 2.9 cm instead of 6.3 cm -- that half of the swing is what the "
                             "operator complained was missing, and the crossing itself is the "
                             "operator's own hand crossing the midline")
    parser.add_argument("--no-arm-gain-quality-guard", action="store_true",
                        help="trust the calibrated arm length even when it is not humanly "
                             "possible. In the measured live_long session the pose "
                             "calibration gave 0.2742 m (left) / 0.3976 m (right), 31%% apart "
                             "-> gain h1_2 left 1.4 (the 2x cap), and then every axis of "
                             "every frame slammed into the travel limit")
    parser.add_argument("--arm-slew-mm-s", type=float, default=DEFAULT_ARM_SLEW_MM_PER_S,
                        metavar="MM/S",
                        help="how fast the commanded wrist target may follow the hand, as a "
                             "**speed** (mm/s, default 600 = 20 mm per 1/30 s frame). The app "
                             "converts it every frame with the loop's measured interval, so a "
                             "10 fps camera gets 40 mm/frame (the SLEW_MAX_STEP_M ceiling) "
                             "instead of 20: the recorded operator's hand moves 37 mm/frame "
                             "(median) / 147 mm/frame (p90) at 10.05 fps, so a fixed "
                             "20 mm/frame cap put the robot on 13%% of the hand speed and the "
                             "measured lateral swing transfer was only 0.43 (0.80 with this "
                             "default). 0 = off (jump straight there; the two-pass IK can then "
                             "lock into a bad branch)")
    parser.add_argument("--arm-slew-mm", type=float, default=None, metavar="MM",
                        help="fixed per-frame cap (mm), overriding --arm-slew-mm-s. Give this "
                             "to reproduce a 2026-10-13 number exactly (that session ran "
                             "20 mm/frame at 10 fps = 200 mm/s); leave it None to let the app "
                             "scale the speed with the measured frame time")
    parser.add_argument("--arm-ik-iterations", type=int, default=60,
                        help="PyBullet IK maxNumIterations per pass (default 60: measured "
                             "2026-10-12, 60/1e-4 beats 200/1e-5 in residual AND speed here)")
    parser.add_argument("--arm-ik-residual-threshold", type=float, default=1e-4,
                        help="PyBullet IK residualThreshold per pass (default 1e-4)")
    parser.add_argument("--no-arm-fallback", action="store_true",
                        help="do not fall back to the 21-point wrist mapping when pose is missing")
    parser.add_argument("--calibration-frames", type=int, default=15,
                        help="open-hand calibration frames for the geometric backend")
    parser.add_argument("--arm-calibrate-frames", type=int, default=30,
                        help="neutral frames per side before the arms start following (default 30)")
    parser.add_argument("--neutral-json", type=Path, default=None,
                        help="reuse a neutral pose saved by --save-neutral")
    parser.add_argument("--save-neutral", type=Path, default=None,
                        help="write this run's neutral pose so the next run can reuse it")
    parser.add_argument("--arm-trace", action="store_true",
                        help="print the operator-side delta per axis (use this to check the axes)")
    parser.add_argument("--dump-npz", type=Path, default=None,
                        help="record the raw frames (hands + pose arms) for offline replay")
    parser.add_argument("--input-npz", type=Path, default=None,
                        help="replay a recorded session instead of the camera (no camera needed)")
    parser.add_argument("--sim-steps", type=int, default=8,
                        help="physics steps per frame (default 8 = about one 30 Hz frame)")
    parser.add_argument("--view", choices=("full", "front"), default="full")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames; 0 runs until Ctrl+C or window close")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-no-mirror", action="store_true")
    parser.add_argument("--preview-window-pos", type=_window_pos, default=(20, 60),
                        metavar="X,Y", help="screen position for the camera window (or 'none')")
    parser.add_argument("--preview-no-topmost", action="store_true")
    parser.add_argument("--preview-dump-dir", type=Path, default=None)
    parser.add_argument("--preview-dump-frames", type=int, default=5)
    parser.add_argument("--no-arm-overlay", action="store_true",
                        help="do not draw the arm skeleton/numbers in the camera window "
                             "(the window then matches the baseline hand-only preview)")
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--report", type=Path, default=None)
    return parser



def arm_hud(bridge, driver, targets) -> str:
    """HUD 里的手臂段：中性标定进度 + 每侧来源（P/H/-）+ 每台机器人的末端残差。"""
    if bridge is None:
        return "arm=off (hands only)"
    per_side = next(iter(targets.values())) if targets else {}
    sources = " ".join(
        f"{side[0].upper()}={SOURCE_SHORT.get(getattr(target, 'source', 'none'), '?')}"
        for side, target in sorted(per_side.items()))
    residuals = ""
    if driver is not None:
        values = driver.residuals_mm()
        text = " ".join(f"{name}={max(entry.values()):.0f}"
                        for name, entry in sorted(values.items()) if entry)
        residuals = f" res(mm) {text}"
    # sat / guard 是累计帧数：sat 一直涨说明手已经跑出机器人行程（目标被钉住，
    # 再动也不跟）；guard 是横向护栏把手腕挡回本侧的帧数。现场用这两个数一眼分辨
    # "手在动但机器人不跟"到底是行程不够还是护栏在挡。
    pinned = sum((bridge.counts.get("clamped") or {}).values())
    guarded = sum((bridge.counts.get("guarded") or {}).values())
    tail = f" sat={pinned}" + ("" if not guarded else f" guard={guarded}")
    return f"{bridge.status()} arm {sources}{residuals}{tail}"


def arm_lines(bridge, sides) -> dict:
    """采集窗口里每侧一行手臂数字：来源字母 + 相对中性姿势的三轴位移（厘米）+ 臂长 + 肘角。

    数据就是 ``bridge.trace()``（``--arm-trace`` 打的同一份）；窗口只显示，不参与计算。
    ``bridge`` 为 ``None``（``--no-arms``）时给空字典，调用方不用分叉。
    """
    if bridge is None:
        return {}
    lines = {}
    for side, entry in sorted(bridge.trace().items()):
        if side not in sides:
            continue
        letter = SOURCE_SHORT.get(entry.get("source", "none"), "?")
        delta = entry.get("delta_op_cm")
        moved = ("--" if delta is None
                 else "[" + ",".join(f"{float(value):+5.1f}" for value in delta) + "]")
        reach = entry.get("operator_reach_m")
        elbow = entry.get("elbow_bend_deg")
        reason = str(entry.get("reason") or "")
        lines[side] = (f"{side} {letter} d(cm){moved} "
                       f"reach={'--' if reach is None else f'{float(reach):.3f}'}m "
                       f"elbow={'--' if elbow is None else f'{float(elbow):.0f}'}deg"
                       f"{'' if not reason else ' ' + reason[:44]}")
    return lines


def print_arm_state(processed, bridge, driver, targets, profiles) -> None:
    """``--arm-trace`` 的现场证据：你动了多少（厘米）以及机器人被要求动到哪（毫米）。"""
    trace = bridge.trace()
    for side, entry in sorted(trace.items()):
        delta = entry["delta_op_cm"]
        text = ("n/a" if delta is None
                else "[" + ", ".join(f"{value:+6.1f}" for value in delta) + "]")
        print(f"  arm trace frame={processed} {side:<5} src={entry['source']:<4} "
              f"delta_op_cm={text} reach={entry['operator_reach_m']} "
              f"elbow={entry['elbow_bend_deg']} reason={entry['reason'] or '-'}",
              flush=True)
    for robot, per_side in sorted(targets.items()):
        fragments = []
        for side, target in sorted(per_side.items()):
            offset = profiles[robot].realtime_neutral(side)
            if target.position is None:
                fragments.append(f"{side}=--")
                continue
            delta = [round((target.position[axis] - offset[axis]) * 1000.0, 1)
                     for axis in range(3)]
            flag = " CLAMP" if target.info.get("clamped") else ""
            fragments.append(f"{side}=[{delta[0]:+5.0f},{delta[1]:+5.0f},{delta[2]:+5.0f}]mm"
                             f" {target.source}{flag}")
        print(f"  arm {robot:>7}: " + "  ".join(fragments), flush=True)



def _record_frame(dumped, raw_frame, arms) -> None:
    """记下这一帧的原始输入（没检到的位置填 NaN），供 ``--dump-npz`` 与离线复验。"""
    for side in SIDES:
        hand = None if raw_frame is None else raw_frame.get(side)
        world = None if raw_frame is None else raw_frame.get(f"world_{side}")
        arm = None
        if arms is not None and arms.world_arms:
            arm = arms.world_arms.get(side)
        dumped["hand"][side].append(np.full((21, 3), np.nan, dtype=np.float32)
                                    if hand is None
                                    else np.asarray(hand, dtype=np.float32))
        dumped["world"][side].append(np.full((21, 3), np.nan, dtype=np.float32)
                                     if world is None
                                     else np.asarray(world, dtype=np.float32))
        dumped["arm"][side].append(np.full((3, 3), np.nan, dtype=np.float32)
                                   if arm is None
                                   else np.asarray(arm, dtype=np.float32))
    dumped["timestamps"].append(float(0.0 if raw_frame is None
                                     else raw_frame.get("timestamp", 0.0)))


def _save_dump(path, dumped) -> Path:
    """把记录写成 npz；键名与 ``devtools/verify_realtime_arm_bridge.py`` 的读取口径一致。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        timestamps=np.asarray(dumped["timestamps"], dtype=np.float64),
        left_hand=np.stack(dumped["hand"]["left"]),
        right_hand=np.stack(dumped["hand"]["right"]),
        world_left_hand=np.stack(dumped["world"]["left"]),
        world_right_hand=np.stack(dumped["world"]["right"]),
        left_arm=np.stack(dumped["arm"]["left"]),
        right_arm=np.stack(dumped["arm"]["right"]))
    return target


def run(args) -> int:
    """采集 -> 手 + 臂 -> 仿真；0 = 至少一侧给出过输出，1 = 全程没有手也没有臂。"""
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    asset = resolve_model_asset(args.model_asset_path) if args.input_npz is None else None
    pose_asset = (Path(args.pose_model_asset_path) if args.pose_model_asset_path
                  else DEFAULT_POSE_ASSET)
    checkpoint = (Path(args.checkpoint) if args.checkpoint
                  else realtime_checkpoint(args.landmark_space))
    robots = tuple(args.robot) if args.robot else loadable_robots()
    profiles = {name: profile_for(name) for name in robots}
    arms_wanted = not args.no_arms
    if args.input_npz is None and arms_wanted and not pose_asset.is_file():
        arms_wanted = False
        print(f"[WARN] pose model not found at {pose_asset}: the arms stay held and only the "
              f"hands move. Put pose_landmarker_lite.task into datasets/ or pass "
              f"--pose-model-asset-path (--no-arms silences this).", flush=True)
    retargeter, backend, note = build_retargeter(
        args.backend, checkpoint, args.device, args.calibration_frames,
        args.smoothing, args.landmark_space)
    scene = camera = preview = driver = bridge = None
    fixed_slew_m = None
    arm_failures = {}
    processed = 0
    started = None
    marks = []
    samples = {side: [] for side in sides}
    invalid = {side: 0 for side in sides}
    raw_seen = {side: 0 for side in sides}
    dumped = {"hand": {side: [] for side in SIDES},
              "world": {side: [] for side in SIDES},
              "arm": {side: [] for side in SIDES},
              "timestamps": []}
    targets = {}
    try:
        args.scene = "robots"                      # 手臂只存在于机器人场景
        scene = build_scene(args, sides)
        if hasattr(scene, "frame_scene"):
            scene.frame_scene(args.view)
        if arms_wanted:
            # 斜坡两种口径（见 §8.9）：给了 --arm-slew-mm 就用固定 mm/帧（复现旧数字），
            # 否则按 --arm-slew-mm-s 的速率每帧换算 —— 现场 8-12 fps 下固定 20 mm/帧
            # 只有 0.2 m/s，而录制里操作者手速中位就是 0.37 m/s，摆动会被整段削掉。
            fixed_slew_m = (None if args.arm_slew_mm is None
                            else max(0.0, float(args.arm_slew_mm)) / 1000.0)
            slew_speed_m_per_s = (0.0 if fixed_slew_m is not None
                                  else max(0.0, float(args.arm_slew_mm_s)) / 1000.0)
            driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                    ik_iterations=args.arm_ik_iterations,
                                    residual_threshold=args.arm_ik_residual_threshold,
                                    slew_m=(0.0 if fixed_slew_m is None else fixed_slew_m),
                                    slew_m_per_s=slew_speed_m_per_s,
                                    elbow_pass=args.arm_elbow_pass,
                                    elbow_recovery_deg=args.arm_elbow_recovery_deg,
                                    elbow_recovery_margin_deg=args.arm_elbow_recovery_margin_deg,
                                    elbow_recovery_cooldown=args.arm_elbow_recovery_frames,
                                    shoulder_anchor_key=args.arm_anchor,
                                    lock_others=(args.arm_ik_scope == "arm"))
            # segments 口径与肘任务要的「机器人自己的量」：实测连杆长 + 中性腕/肘点（基座系）。
            # 加载后量一次即可，之后每帧只是加减法（见 arm_realtime.MAP_MODES）。
            robot_links, robot_anchors = {}, {}
            if args.arm_map in ("segments", "posture"):
                for name in robots:
                    adapter = driver.ik[name]
                    robot_links[name], robot_anchors[name] = {}, {}
                    for side in sides:
                        points = adapter.arm_points(side)
                        # 上臂的「肩端」取哪个点由 --arm-anchor 决定（默认 "upper"）：
                        # "upper" = 肘关节父连杆原点，与肘刚性相连，|肘 - 它| 与姿势无关
                        # （实测最大偏差 0.00 mm）；"shoulder" = 名字里第一个含 shoulder 的
                        # 连杆（= 上臂 pitch 那节），会随 shoulder_roll 摆动 48.5-101.7 mm。
                        # 方向配方的两个目标都建在这个锚点上：用后者会让「肩 + 上臂长*方向」
                        # 在非中性姿势下偏离可达壳面，IK 只能折中 —— 现场
                        # 「GR1-T2 前臂还是反」（肘残差中位 222 mm）的根因。
                        # 复现/对比命令：E:\python3.11.7\python.exe devtools\probe_arm_segment_rigidity.py
                        shoulder = np.asarray(points[args.arm_anchor], dtype=np.float64)
                        elbow = np.asarray(points["elbow"], dtype=np.float64)
                        wrist = np.asarray(points["wrist"], dtype=np.float64)
                        robot_links[name][side] = {
                            "upper": float(np.linalg.norm(elbow - shoulder)),
                            "fore": float(np.linalg.norm(wrist - elbow))}
                        # arm_points 给的是世界坐标 -> 基座系（肘目标按基座系给）
                        robot_anchors[name][side] = {
                            # 肩锚点只有方向口径（--arm-map posture）用：腕/肘目标 =
                            # 肩锚点 + 机器人自己的连杆长 * 你的单位方向。
                            "shoulder": list(adapter.to_base(shoulder)),
                            "elbow": list(adapter.to_base(elbow)),
                            "wrist": list(adapter.to_base(wrist))}
                if not args.arm_no_shoulder_follow:
                    # 肘/腕目标按「当前肩 - 加载锚点」平移（见 ArmSceneDriver._shoulder_shift）：
                    # 躯干被手臂反作用力推着走时，目标仍然落在「离当前肩正好一根上臂」的球面上。
                    driver.set_shoulder_anchors(robot_anchors)
                elbow_note = ("elbow task: built by the direction recipe (pass "
                              f"{args.arm_elbow_pass})"
                              if args.arm_map == "posture" else
                              f"elbow task {'on' if args.arm_elbow_task else 'off'}")
                pass_note = ("" if args.arm_map == "posture" else
                             (f" (pass {args.arm_elbow_pass})"
                              if args.arm_elbow_task else ""))

                print(f"arm map: {args.arm_map} | shoulder end = {args.arm_anchor} "
                      f"({'rigid: the elbow parent link' if args.arm_anchor == 'upper' else 'historical name-matched link'})"
                      f" | ik scope = {args.arm_ik_scope}"
                      f" | measured links (m) "
                      + " ".join(f"{name}:{side[0]}" +
                                 f" {robot_links[name][side]['upper']:.3f}"
                                 f"+{robot_links[name][side]['fore']:.3f}"
                                 for name in robots for side in sides)
                      + f" | {elbow_note}"
                      + pass_note
                      + (f" | elbow branch recovery: >{args.arm_elbow_recovery_deg:.0f} deg"
                         f" (margin {args.arm_elbow_recovery_margin_deg:.0f} deg,"
                         f" retry every {args.arm_elbow_recovery_frames} frames)"

                         if args.arm_elbow_recovery_deg > 0.0 else
                         " | elbow branch recovery: off"),
                      flush=True)
            gains = {str(pair[0]): float(pair[1]) for pair in (args.arm_gain or [])}
            axis_map = AxisMap()
            if args.arm_axis_order and args.arm_axis_signs:
                axis_map = AxisMap.parse(args.arm_axis_order, args.arm_axis_signs)
            elif args.arm_axis_order:
                axis_map = AxisMap.parse(args.arm_axis_order,
                                         ",".join(f"{sign:+.0f}" for sign in axis_map.signs))
            elif args.arm_axis_signs:
                axis_map = AxisMap.parse("".join(axis_map.order), args.arm_axis_signs)
            neutral = load_neutral(args.neutral_json) if args.neutral_json else None
            bridge = RealtimeArmBridge(
                [profiles[name] for name in robots], sides=sides, axis_map=axis_map,
                calibration_frames=args.arm_calibrate_frames,
                orientation=args.arm_orientation, gains=gains,
                max_travel_m=(None if args.arm_travel_mm is None
                              else float(args.arm_travel_mm) / 1000.0),
                travel_mode=args.arm_travel_mode,
                lateral_guard_m=max(0.0, float(args.arm_lateral_guard_mm)) / 1000.0,
                gain_quality_guard=not args.no_arm_gain_quality_guard,
                neutral=neutral, fallback_hand=not args.no_arm_fallback,
                map_mode=args.arm_map, robot_links=robot_links,
                elbow_task=args.arm_elbow_task, elbow_gain=args.arm_elbow_gain,
                posture_gain=args.arm_posture_gain,
                max_direction_rate_deg_s=args.arm_direction_rate_deg_s,
                anchors=robot_anchors)
            print(f"arms: {', '.join(robots)} | ik_mode "
                  f"{({name: profiles[name].ik_mode for name in robots})} | "
                  f"orientation {args.arm_orientation} | travel limit "
                  f"{bridge.travel_limits()} m ({args.arm_travel_mode}) | "
                  f"lateral guard {args.arm_lateral_guard_mm} mm | "
                  f"neutral {'reused from ' + str(args.neutral_json) if neutral else str(args.arm_calibrate_frames) + ' frames'} | "
                  + (f"target slew {fixed_slew_m * 1000.0:.1f} mm/frame (fixed)"
                     if fixed_slew_m is not None else
                     f"target slew {args.arm_slew_mm_s:.0f} mm/s "
                     f"(ceiling {SLEW_MAX_STEP_M * 1000.0:.0f} mm/frame)"),
                  flush=True)
        if args.input_npz is not None:
            camera = ReplayFrameSource(args.input_npz, args.landmark_space)
            print(f"replay: {args.input_npz} ({camera.frames} frames, "
                  f"landmark_space={args.landmark_space}); no camera is opened", flush=True)
        else:
            camera = CameraFrameSource(asset, args.camera_index, args.width, args.height,
                                       args.fps, args.landmark_space,
                                       pose_asset if arms_wanted else None,
                                       args.pose_min_visibility)
        preview = build_arm_preview(args, camera, sides)
        if not args.headless:
            print(PREVIEW_HINT_ROBOTS, flush=True)
            print(f"camera window {ARM_PREVIEW_WINDOW!r}: raw frame + 21 hand points + both "
                  f"arms (shoulder/elbow/wrist); the PyBullet window shows the three robots.",
                  flush=True)
        if bridge is not None:
            print("keep a neutral arm pose (elbows slightly bent, hands in front of the chest) "
                  f"for about {args.arm_calibrate_frames} frames per side; until then the arms hold "
                  "their working point.", flush=True)
        started = time.perf_counter()

        while True:
            if args.frames and processed >= args.frames:
                break
            if not scene.alive():
                print("PyBullet window closed; stopping.", flush=True)
                break
            window = camera.next_frame()
            if getattr(camera, "exhausted", False):
                print("replay finished (all recorded frames were replayed).", flush=True)
                break
            processed += 1
            raw_frame = camera.last_raw_frame
            for side in sides:
                if raw_frame is not None and raw_frame.get(side) is not None:
                    raw_seen[side] += 1
            latest = {side: None for side in sides}
            if window is not None:
                result = retargeter.retarget(window)
                for side in sides:
                    angles = getattr(result, side + "_angles")
                    if angles is None:
                        invalid[side] += 1
                        continue
                    values = np.asarray(angles.values, dtype=np.float64)
                    if values.shape != (HAND_ANGLE_DIM,) or not np.isfinite(values).all():
                        raise ValueError(f"invalid {side} angles: shape={values.shape}")
                    scene.apply(side, values)
                    samples[side].append(values)
                    latest[side] = values
            targets = {}
            if bridge is not None:
                # 这一帧的真实时长先算出来：斜坡的速率口径与关键点跳变护栏都要用它
                # （现场摄像头 8-12 fps，名义 1/30 s 会把 600 度/秒算成 20 度/帧）。
                frame_dt = _median_interval(marks) or NOMINAL_FRAME_DT
                targets = bridge.update(arms_world=camera.arms(), hands=raw_frame,
                                        frame_dt=frame_dt)
                orientation_by_side = bridge.orientation_target()
                # 斜坡：速率（mm/s）-> 这一帧最多走多少米。现场摄像头循环帧率不固定
                # （8-12 fps），固定 mm/帧 会把跟随速度绑在帧率上，摆动被整段削掉（§8.9）。
                # 第一帧拿不到间隔，用名义 1/30 s（= 旧口径 20 mm/帧），不是"关掉斜坡"。
                driver.update_slew(frame_dt)
                for side in sides:
                    per_side = {name: entry[side] for name, entry in targets.items()}
                    positions = {name: target.position for name, target in per_side.items()
                                 if target.position is not None}
                    if len(positions) != len(per_side):
                        continue        # 还有机器人没目标：这一侧整侧跳过（保持上一帧）
                    # 肘位置任务（--arm-elbow-task）：只给量到锚点的那几台，缺的退回只喂腕。
                    elbows = {name: target.elbow for name, target in per_side.items()
                              if getattr(target, "elbow", None) is not None} or None
                    orientation = orientation_by_side[side]
                    try:
                        # orientation 是**相对中性**的四元数（同一份给三台，各自乘自己的
                        # 中性 EE 朝向 —— orientation_frame="neutral" 就是这件事）。
                        # 取自 bridge.orientation_target()：不能从某一台的 target 里挑，
                        # 那样会挑到 position-only 的那台（None，姿态被静默丢掉）。
                        driver.apply_each(side, positions, source="base",
                                          orientation=orientation,
                                          orientation_frame="neutral",
                                          elbows=elbows)
                    except Exception as error:  # noqa: BLE001 - 一台坏不该拖垮整场
                        if side not in arm_failures:
                            arm_failures[side] = f"{type(error).__name__}: {error}"
                            print(f"[WARN] {side}: arm IK failed, the arms keep their last "
                                  f"command: {arm_failures[side]}", flush=True)
                driver.step(1)
            else:
                scene.step()
            if args.dump_npz is not None:
                _record_frame(dumped, raw_frame, camera.arms())
            if preview is not None:
                body = camera.arms()
                # 不在这里挑手臂点：arm_preview.update() 自己把形状不对 / 含 NaN 的一侧
                # 当作"没有"（与桥的丢帧保持同口径），这里只负责把 None 原样传下去。
                preview.update(raw_frame, latest, arm_hud(bridge, driver, targets),
                               arms=None if body is None else body.arms,
                               lines=arm_lines(bridge, sides))
                if preview.quit_requested:
                    print("camera window closed; stopping.", flush=True)
                    break
            marks.append(time.perf_counter())
            if args.print_every and processed % args.print_every == 0:
                counts = " ".join(f"{side}={len(samples[side])}" for side in sides)
                print(f"frame={processed} fps={_loop_fps(marks):.1f} hand valid={counts} "
                      f"invalid={sum(invalid.values())} | {arm_hud(bridge, driver, targets)}",
                      flush=True)
                if args.arm_trace and bridge is not None:
                    print_arm_state(processed, bridge, driver, targets, profiles)
        final = {side: scene.joint_values(side) for side in sides}
        # 注意：这些都要在 finally 断开之前读 —— PyBullet 断连后再 getLinkState 会抛
        # "Not connected to physics server"。
        arm_residuals = {} if driver is None else driver.residuals_mm()
        arm_hold_drift = {} if driver is None else driver.hold_drift()
        arm_posture = {} if driver is None else driver.posture_summary()
        arm_gap = {} if driver is None else driver.forearm_gap_summary()
        arm_summary = None if driver is None else driver.summary()
        scene_summary = scene.summary() if hasattr(scene, "summary") else None
        elapsed = time.perf_counter() - started
    finally:
        if preview is not None:
            _drain_in_background(preview.close, "camera preview close")
        if camera is not None:
            _drain_in_background(camera.release, "camera release")
        if scene is not None:
            _drain_in_background(scene.close, "PyBullet disconnect")

    fps = _loop_fps(marks)
    print(f"\nprocessed={processed} frames in {elapsed:.2f} s  fps={fps:.1f}  "
          f"median frame interval {_median_interval(marks) * 1000:.1f} ms")
    if scene_summary is not None:
        print(f"scene={scene_summary['scene']} sim_frames={scene_summary['sim_frames']} "
              f"applied={scene_summary['applied_frames']} "
              f"clipped={scene_summary['clipped_dimensions']}", flush=True)
    for robot, entry in sorted(arm_residuals.items()):
        if entry:
            text = " ".join(f"{side}={value:.1f}" for side, value in sorted(entry.items()))
            print(f"  arm {robot}: wrist residual mm {text}", flush=True)
    for robot, per_side in sorted(arm_posture.items()):
        for side, entry in sorted(per_side.items()):
            if not entry or entry.get("frames", 0) < 2:
                continue
            bend = entry["elbow_bend_deg"]
            print(f"  arm {robot}/{side}: shoulder travel {entry['shoulder_travel_deg']:.1f} deg, "
                  f"upper arm swept {entry['upper_arm_dir_travel_deg']:.1f} deg, "
                  f"elbow moved {entry['elbow_span_mm']:.0f} mm, "
                  f"elbow bend {bend['min']}-{bend['max']} deg", flush=True)
    if arm_gap:
        text = " ".join(f"{robot}={value:.0f}" for robot, value in sorted(
            (arm_gap.get("min_mm") or {}).items()))
        print(f"  arm forearm gap (worst): {text} mm "
              f"[worst {arm_gap.get('worst_mm')} mm over {arm_gap.get('frames')} frames]",
              flush=True)
    if arm_hold_drift:
        print(f"  arm hold drift: {json.dumps(arm_hold_drift, ensure_ascii=False)}", flush=True)
    if bridge is not None:
        for side, entry in sorted((bridge.summary().get("arm_length_trust") or {}).items()):
            if not entry.get("trusted"):
                print(f"  arm gain guard [{side}]: {entry.get('reason')}; using "
                      f"profile.metric_scale instead of arm-length normalisation", flush=True)
        arm_guarded_total_print = sum((bridge.counts.get("guarded") or {}).values())
        if arm_guarded_total_print:
            print(f"  arm lateral guard: {arm_guarded_total_print} frames had a wrist target "
                  f"pushed back to its own side "
                  f"(limit {1000.0 * bridge.lateral_guard_m:.0f} mm)", flush=True)
    if preview is not None:
        print(f"camera preview: window={'on' if preview.show else 'off'} "
              f"mirror={'on' if preview.mirror else 'off'} "
              f"arm_overlay={'on' if getattr(preview, 'overlay', False) else 'off'} "
              f"dumped_png={preview.dumped}"
              f"{'' if preview.rect is None else f' rect={preview.rect}'}"
              f"{'' if not preview.note else ' note=' + preview.note}", flush=True)
    arm_valid_total = 0 if bridge is None else sum(bridge.counts["valid"].values())
    arm_clamped_total = 0 if bridge is None else sum(bridge.counts["clamped"].values())
    arm_guarded_total = 0 if bridge is None else sum(bridge.counts["guarded"].values())
    arm_gain_guarded_total = (0 if bridge is None
                              else sum(bridge.counts["gain_guarded"].values()))
    report = {
        "entry": ARM_ENTRY,
        "backend": backend,
        "landmark_space": args.landmark_space,
        "length_unit": "m" if args.landmark_space == "world" else "normalized",
        "backend_note": note,
        "model_asset": None if asset is None else str(asset),
        "pose_model_asset": None if not arms_wanted else str(pose_asset),
        "camera_index": args.camera_index,
        "input_npz": None if args.input_npz is None else str(args.input_npz),
        "headless": bool(args.headless),
        "scene": args.scene,
        "scene_summary": scene_summary,
        "arms_enabled": bool(arms_wanted),
        "robots": list(robots),
        "arm_failures": arm_failures,
        "arm_targets_valid_frames": arm_valid_total,
        "arm_clamped_frames": arm_clamped_total,
        "arm_guarded_frames": arm_guarded_total,
        "arm_gain_guarded_frames": arm_gain_guarded_total,
        "arm_travel_mode": args.arm_travel_mode,
        "arm_lateral_guard_mm": float(args.arm_lateral_guard_mm),
        "arm_slew_mm_per_s": (None if fixed_slew_m is not None
                              else float(args.arm_slew_mm_s)),
        "arm_slew_mm_fixed": (None if fixed_slew_m is None else fixed_slew_m * 1000.0),
        "arm_slew_ceiling_mm": SLEW_MAX_STEP_M * 1000.0,
        "arm_slew_mm_last_frame": (None if driver is None
                                   else round(1000.0 * driver.slew_m, 3)),
        "arm": arm_summary,
        "arm_posture": arm_posture,
        "arm_forearm_gap_mm": arm_gap,
        "bridge": (None if bridge is None else bridge.summary()),
        "arm_last_targets": {
            robot: {side: target.describe() for side, target in sorted(per_side.items())}
            for robot, per_side in sorted(targets.items())},
        "preview": None if preview is None else {
            "window": bool(preview.show),
            "mirrored": bool(preview.mirror),
            "arm_overlay": bool(getattr(preview, "overlay", False)),
            "dumped_png": int(preview.dumped),
            "dump_dir": None if preview.dump_dir is None else str(preview.dump_dir),
            "window_name": preview.window_name,
            "window_pos": None if preview.window_pos is None else list(preview.window_pos),
            "window_rect": None if preview.rect is None else list(preview.rect),
            "topmost": bool(preview.topmost),
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
        print(f"  hand {side}: raw={entry['raw_detected_frames']}/{processed} "
              f"valid={entry['valid_frames']} invalid={entry['invalid_frames']}")
    if bridge is not None:
        trace = bridge.trace()
        for side, entry in sorted(trace.items()):
            print(f"  arm  {side}: source={entry['source']} "
                  f"delta_op_cm={entry['delta_op_cm']} reach_m={entry['operator_reach_m']} "
                  f"elbow_deg={entry['elbow_bend_deg']}")
    if args.save_neutral and bridge is not None:
        if bridge.calibrating:
            print("[WARN] --save-neutral skipped: the neutral calibration never completed",
                  flush=True)
        else:
            path = save_neutral(args.save_neutral, bridge.neutral_record(),
                                {"entry": ARM_ENTRY, "axis_map": bridge.axis_map.describe()})
            print(f"neutral pose saved: {path}", flush=True)
    if args.dump_npz is not None and dumped["timestamps"]:
        print(f"raw frames dumped: {_save_dump(args.dump_npz, dumped)} "
              f"({len(dumped['timestamps'])} frames)", flush=True)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {path}")
    if not any(samples[side] for side in sides) and arm_valid_total == 0:
        if any(raw_seen[side] for side in sides):
            print("[FAIL] MediaPipe detected a hand but produced no angles: put the whole hand "
                  "in the middle of the frame and hold it open for about "
                  f"{args.calibration_frames} frames to finish calibration.")
        else:
            print("[FAIL] no raw hand detection at all: check the camera window - the hand must "
                  "really be visible (lighting, distance, not only at the frame edge).")
        return 1
    return 0




def main(args=None) -> int:
    """CLI 边界：参数补齐 -> run()；异常在这里变成 ``[FAIL]`` + 退出码 1。"""
    if args is None:
        args = build_parser().parse_args()
    else:
        # 被 launcher / 测试用现成的 args 调用时，补齐它没声明的可选参数，
        # 保证两条调用方式行为一致（与基线入口同一套做法）。
        for name, value in vars(build_parser().parse_args([])).items():
            if not hasattr(args, name):
                setattr(args, name, value)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nCapture stopped by user.", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary reports instead of raising
        print(f"[FAIL] realtime teleop stopped: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    # 与基线入口一致：先跑完 main()（统计/报告都写完），flush 之后直接 os._exit，
    # 免得卡在 MediaPipe landmarker.close()（本机实测约 42 s）上。
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)








