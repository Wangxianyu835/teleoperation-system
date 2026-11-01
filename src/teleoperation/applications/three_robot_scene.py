"""Whole-body robot scene executing named native hand commands.

HandSpec defines every movable hand joint, independent coordinates and coupling.
Non-hand joints remain held until an arm driver explicitly takes control.
"""

from __future__ import annotations

import time

import numpy as np


# 三台机器人的横向摆位（米）与标签色，与 tools show-all-hands 保持一致
ROBOT_ORDER = ("h1_2", "gr1_t2", "g1")
LAYOUT_X = {"h1_2": -2.4, "gr1_t2": 0.0, "g1": 2.4}
LABEL_COLOR = {"h1_2": [1.0, 0.35, 0.35],
               "gr1_t2": [0.35, 0.5, 1.0],
               "g1": [0.35, 1.0, 0.45]}

PHYSICS_DT = 1.0 / 240.0
DEFAULT_STEPS_PER_FRAME = 8      # 33 ms 一帧 = 8 x 1/240 s
HAND_FORCE = 200.0               # 手指：与 show-all-hands 相同的力矩
HOLD_FORCE = 500.0               # 锁姿态：必须比手指大，否则手臂被手指拖动


def joint_extent(p, body, cid):
    """合并某机器人所有 link 的 AABB -> (lo, hi)。"""
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for index in range(-1, p.getNumJoints(body, physicsClientId=cid)):
        try:
            low, high = p.getAABB(body, index, physicsClientId=cid)
        except p.error:
            continue
        lo = np.minimum(lo, np.asarray(low, dtype=float))
        hi = np.maximum(hi, np.asarray(high, dtype=float))
    return lo, hi


class ThreeRobotHandScene:
    """H1-2 / GR1-T2 / G1 并排；每侧原装手接收带规格校验的原生关节命令。"""

    def __init__(self, sides=("left", "right"), gui=True,
                 steps_per_frame=DEFAULT_STEPS_PER_FRAME,
                 label_every=15, gravity=-9.81, ground_clearance=0.03, robot_types=None):
        self.cid = -1
        try:
            self._initialize(sides,gui,steps_per_frame,label_every,gravity,ground_clearance,robot_types)
        except BaseException:
            if self.cid >= 0 and hasattr(self,'p'):
                self.close()
            raise

    def _initialize(self, sides, gui, steps_per_frame, label_every, gravity, ground_clearance, robot_types):
        import pybullet as p
        import pybullet_data

        from teleoperation.robots.hand_spec import load_hand_spec
        from teleoperation.simulation import RobotLoader
        from teleoperation.simulation.robot_loader import read_joint_ranges

        if not sides or len(set(sides)) != len(sides) or set(sides) - {'left', 'right'}:
            raise ValueError("at least one hand side is required")
        self.p = p
        self.sides = tuple(sides)
        self.gui = bool(gui)
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.label_every = max(1, int(label_every))
        self.gravity = float(gravity)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0 or not p.isConnected(self.cid):
            raise RuntimeError(
                "PyBullet GUI connection could not be created (no display or a "
                "non-interactive session); rerun with --headless."
            )
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        self._prepare_physics(p.setGravity, (0, 0, self.gravity))
        p.setTimeStep(PHYSICS_DT, physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        loader = RobotLoader(self.cid)
        self.robots = []
        selected = ROBOT_ORDER if robot_types is None else tuple(robot_types)
        if not selected or any(name not in ROBOT_ORDER for name in selected):
            p.disconnect(self.cid)
            raise ValueError('Unknown scene robot selection')
        for robot_type in selected:
            try:
                body = loader.load_robot(robot_type)["robot"]
            except Exception as error:
                raise RuntimeError(f"Selected robot {robot_type} could not be loaded: {error}") from error
            base, orientation = p.getBasePositionAndOrientation(
                body, physicsClientId=self.cid)
            p.resetBasePositionAndOrientation(
                body, [base[0] + LAYOUT_X[robot_type], base[1], base[2]], orientation,
                physicsClientId=self.cid)
            ranges, names = read_joint_ranges(body, self.cid)
            plans = {}
            for side in self.sides:
                spec = load_hand_spec(robot_type, side)
                if not set(spec.joint_names) <= set(names):
                    raise ValueError(f"Loaded robot lacks native hand joints: {robot_type}/{side}")
                for name, (low, high) in zip(spec.joint_names, spec.joint_limits):
                    if not np.allclose(ranges[name], (low, high), atol=1e-6, rtol=0):
                        raise ValueError(f"Loaded hand limits differ from HandSpec: {name}")
                plans[side] = (spec, names)
            hand_ids = {names[joint] for spec, names in plans.values()
                        for joint in spec.joint_names}
            hold_ids, hold_targets = [], []
            for index in range(p.getNumJoints(body, physicsClientId=self.cid)):
                info = p.getJointInfo(body, index, physicsClientId=self.cid)
                if info[2] == p.JOINT_FIXED or index in hand_ids:
                    continue
                hold_ids.append(index)
                hold_targets.append(float(p.getJointState(
                    body, index, physicsClientId=self.cid)[0]))
            low, high = joint_extent(p, body, self.cid)
            text = "  ".join(
                f"{side[0].upper()} {spec.dof} DOF / {len(spec.joint_names)} joints"
                for side, (spec, _names) in plans.items())
            self.robots.append({
                "type": robot_type,
                "body": body,
                "plans": plans,
                "hold_ids": hold_ids,
                "hold_targets": hold_targets,
                "label_position": [float((low[0] + high[0]) / 2),
                                   float((low[1] + high[1]) / 2),
                                   float(high[2] + 0.30)],
                "hand_description": text,
            })
            print(f"  {robot_type:<8} 摆位 x={LAYOUT_X[robot_type]:+.1f}  原装手 {text}  "
                  f"锁住 {len(hold_ids)} 个非手部关节", flush=True)
        if not self.robots:
            p.disconnect(self.cid)
            raise RuntimeError(
                "no robot could be loaded; check that assets/robots "
                "contains h1_2 / gr1 / g1"
            )
        self.ground_lift = self._lift_clear_of_ground(ground_clearance)

        self.applied = {side: 0 for side in self.sides}
        self.frames = 0
        self.loss_reported = False
        self.camera_target = [0.0, 0.0, 0.0]
        self.camera_distance = None
        self.label_ids = []
        # 无头（DIRECT）也要算一次取景：离屏渲染与 GUI 用同一套 target / distance。
        self.camera_distance = self.frame_scene()
        if self.gui:
            for robot in self.robots:
                self.label_ids.append(p.addUserDebugText(
                    f"{robot['type'].upper()}\n{robot['hand_description']}",
                    robot["label_position"], textSize=1.1,
                    textColorRGB=LABEL_COLOR[robot["type"]], lifeTime=0,
                    physicsClientId=self.cid))

    def _lift_clear_of_ground(self, clearance: float = 0.03) -> dict:
        """把每台机器人抬到「最低点刚好离开地面」，避免初始穿透的接触冲击踢动腿关节。

        [实测] 直接按 URDF 自带的 base_z 加载时脚底与 plane 有初始穿透：1.5 s
        （45 帧 x 8 步）后 h1_2 的 left_ankle_roll_joint 被顶动 0.664 rad、
        GR1-T2 的 right_hip_pitch_joint 0.700 rad，画面上像"机器人自己在歪"。
        抬高 0.15 m 后两者都降到 <= 0.007 rad。重力 -9.81 与 0 的漂移完全一样，
        所以原因只是接触冲量，不是重力。
        """
        p, cid = self.p, self.cid
        lifted = {}
        for robot in self.robots:
            low, _high = joint_extent(p, robot["body"], cid)
            base, orientation = p.getBasePositionAndOrientation(
                robot["body"], physicsClientId=cid)
            raise_by = float(clearance - low[2])
            if raise_by > 0.0:
                p.resetBasePositionAndOrientation(
                    robot["body"], [base[0], base[1], base[2] + raise_by], orientation,
                    physicsClientId=cid)
                lifted[robot["type"]] = round(raise_by, 4)
        if lifted:
            print(f"  抬离地面（避免初始穿透）：{lifted} m", flush=True)
        return lifted

    def _prepare_physics(self, call, positional, timeout: float = 15.0) -> None:
        """Call a PyBullet setup function, retrying until the server answers.

        ``p.connect(p.GUI)`` returns before the GUI server is ready, so the first
        call can raise ``Not connected to physics server``. DIRECT 客户端会立刻应答。
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

    def frame_scene(self, view: str = "full"):
        """算三台整机的取景并（GUI 下）设置相机；返回相机距离。

        按「水平跨度」和「高度」两个约束取更远的那个：pybullet 窗口通常是 16:9，
        只按正方形 60 度 FOV 估距离会退得太远、机器人显得很小。
        无头（DIRECT）下只计算并把结果留在 ``camera_target`` / ``camera_distance``，
        离屏渲染（``getCameraImage``）可以复用同一套取景。
        """
        p, cid = self.p, self.cid
        low_all = np.array([np.inf] * 3)
        high_all = np.array([-np.inf] * 3)
        for robot in self.robots:
            low, high = joint_extent(p, robot["body"], cid)
            low_all = np.minimum(low_all, low)
            high_all = np.maximum(high_all, high)
        center = (low_all + high_all) / 2.0
        aspect = 16.0 / 9.0
        half_v = float(np.radians(30.0))
        half_h = float(np.arctan(aspect * np.tan(half_v)))
        width = float(high_all[0] - low_all[0])
        height = float(high_all[2] - low_all[2])
        pad = 0.45
        # 斜视（yaw 135）时三台沿 X 排开，最近那台比目标中心近约 (宽/2)*sin45；
        # 只按中心距离取景会把最边上一台裁掉（实测：iso 视角只看得见两台）。
        azimuth = np.radians(45.0) if view == "full" else 0.0
        depth_half = 0.5 * width * abs(float(np.sin(azimuth)))
        side_half = 0.5 * width * abs(float(np.cos(azimuth)))
        distance = float(max(depth_half + (side_half + pad) / np.tan(half_h),
                             depth_half + (height / 2.0 + pad) / np.tan(half_v),
                             2.5))
        self.camera_target = [float(center[0]), float(center[1]), float(center[2])]
        self.camera_distance = distance
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=round(distance, 2),
                cameraYaw=135.0 if view == "full" else 0.0, cameraPitch=-10,
                cameraTargetPosition=list(self.camera_target),
                physicsClientId=cid)
        return distance

    def apply_native(self, result) -> None:
        """Write one robot/side's named native command, without the L21 adapter."""
        from teleoperation.contracts.hand_control import HandJointCommand
        if not isinstance(result, HandJointCommand):
            raise ValueError("Scene requires a valid native command")
        if result.side not in self.sides or not self.alive():
            raise ValueError("Native command side is not active")
        robot = next((r for r in self.robots if r["type"] == result.hand_model), None)
        if robot is None:
            raise ValueError("Native command robot is not loaded")
        spec, names = robot["plans"][result.side]
        if result.spec_hash != spec.spec_hash or result.joint_names != spec.joint_names:
            raise ValueError("Command target specification differs from the loaded hand")
        values = result.named_positions()
        independent = np.array([values[n] for n in spec.independent_joint_names])
        expanded = spec.expand(independent)
        if any(abs(values[n]-expanded[n]) > 1e-6 for n in spec.joint_names):
            raise ValueError("Command violates native hand coupling")
        for name, (low, high) in zip(spec.joint_names, spec.joint_limits):
            if not low - 1e-6 <= values[name] <= high + 1e-6:
                raise ValueError(f"Native scene limit exceeded: {name}")
        for name, value in values.items():
            self.p.setJointMotorControl2(robot["body"], names[name], self.p.POSITION_CONTROL,
                targetPosition=float(value), force=HAND_FORCE, physicsClientId=self.cid)
        if not hasattr(self, "native_commands"):
            self.native_commands = {}
            self.native_applied = {}
        self.native_commands[result.key] = dict(values)
        key = result.hand_model + "/" + result.side
        self.native_applied[key] = self.native_applied.get(key, 0) + 1
        self.applied[result.side] += 1

    def step(self) -> None:
        """锁住非手部关节，然后推进 ``steps_per_frame`` 步物理。"""
        if not self.alive():
            if not self.loss_reported:
                self.loss_reported = True
                print("PyBullet connection lost (GUI window closed or no display in this "
                      "session); stopping and still writing the summary. Use --headless "
                      "when the GUI cannot open here.", flush=True)
            return
        p, cid = self.p, self.cid
        for robot in self.robots:
            for index, target in zip(robot["hold_ids"], robot["hold_targets"]):
                p.setJointMotorControl2(robot["body"], index, p.POSITION_CONTROL,
                                        targetPosition=target, force=HOLD_FORCE,
                                        physicsClientId=cid)
        for _ in range(self.steps_per_frame):
            p.stepSimulation(physicsClientId=cid)
        self.frames += 1
        if self.gui and self.frames % self.label_every == 0:
            self._refresh_labels()

    def _refresh_labels(self) -> None:
        """每 ``label_every`` 帧刷新头顶 3D 文字（机器人名 + 已写命令数 + 原生维度）。"""
        p, cid = self.p, self.cid
        for index, robot in enumerate(self.robots):
            counts = "  ".join(f"{side[0].upper()}{self.applied[side]}"
                               for side in self.sides)
            self.label_ids[index] = p.addUserDebugText(
                f"{robot['type'].upper()}  {counts}\n{robot['hand_description']}",
                robot["label_position"], textSize=1.1,
                textColorRGB=LABEL_COLOR[robot["type"]], lifeTime=0,
                replaceItemUniqueId=self.label_ids[index], physicsClientId=cid)

    def alive(self) -> bool:
        """False once the user closes the GUI window."""
        return bool(self.p.isConnected(self.cid))

    def close(self) -> None:
        if self.p.isConnected(self.cid):
            self.p.disconnect(self.cid)

    def joint_values(self, side: str) -> dict:
        """这一侧三台机器人原装手的实际角度，键为 ``'机器人.关节名'``。"""
        if side not in self.sides or not self.alive():
            return {}
        p, cid = self.p, self.cid
        values = {}
        for robot in self.robots:
            plan = robot["plans"].get(side)
            if plan is None:
                continue
            spec, names = plan
            for joint_name in spec.joint_names:
                values[f"{robot['type']}.{joint_name}"] = float(
                    p.getJointState(robot["body"], names[joint_name],
                                    physicsClientId=cid)[0])
        return values

    def summary(self) -> dict:
        """给 ``--report`` 用的场景摘要（哪三台、各写了几帧、被限位截断多少次）。"""
        return {
            "scene": "three_robots",
            "steps_per_frame": self.steps_per_frame,
            "physics_dt": PHYSICS_DT,
            "sim_frames": self.frames,
            "applied_frames": dict(self.applied),
            "native_applied_frames": dict(getattr(self, "native_applied", {})),
            "ground_lift": dict(self.ground_lift),
            "robots": [
                {"robot": robot["type"],
                 "layout_x": LAYOUT_X[robot["type"]],
                 "native_hands": {side: {"dof": spec.dof, "joint_names": list(spec.joint_names),
                                          "spec_hash": spec.spec_hash}
                                  for side, (spec, _names) in robot["plans"].items()},
                 "held_joints": len(robot["hold_ids"]),
                 "native_joints": {side: len(spec.joint_names)
                                   for side, (spec, _n) in robot["plans"].items()}}
                for robot in self.robots
            ],
        }
