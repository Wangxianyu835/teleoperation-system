"""任务基类 - 所有30个任务都继承此类

2026-11-02 物理一致性修复（本文件新增的机制，四个问题都在旧版实测复现过）：

1. **穿模弹飞**：旧版 30 个任务把物体中心统一写成 z=0.42/0.43（桌面 0.42），
   忽略了自己的半高，圆柱/高箱体实际嵌进桌面 2~10 cm；PyBullet 会把它们
   弹起来再落地。实测「完全不给动作」的 1 秒内 21/30 任务物体自移动 > 2 cm
   （closedrawer/drawerbook 0.280 m、plateoven 0.149 m）。
   -> 新增几何登记 + 姿态感知半高 + snap_to_supports()（只抬不压）。
2. **t=0 白送成功**：旧判据是绝对阈值，3/30 任务在 t=0 就 check_success()=True
   （liftmug / opendrawer / openmicrowave）。
   -> 新增 initial_poses + 相对判据助手 displacement/rise/tilt_deg/yaw_change_deg。
3. **门/抽屉是自由刚体**：落桌后被挤出 0.28 m。
   -> 新增 load_hinged_panel（竖直轴，重力不产生力矩）与 load_sliding_panel
      （水平轴，重力不产生沿轨道的力）：静止时不会自己动，必须由机器人推。
4. **box 没有内腔**：「放进烤箱/柜子」这类任务物理上不可达。
   -> 新增 load_open_box（5 块薄板拼一个开口壳）。
"""

import numpy as np
import pybullet as p
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


# load_fixed_table() 的默认桌面：中心 z=0.4、半厚 0.02 -> 桌面高 0.42 m
DEFAULT_TABLE_POSITION = [0.5, 0.0, 0.4]
DEFAULT_TABLE_HALF_EXTENTS = [0.5, 0.8, 0.02]

# 贴面吸附的容差：小于 1 mm 的偏差不算穿模
SNAP_TOLERANCE = 1e-3


def _quat_matrix(orn) -> np.ndarray:
    """四元数 -> 3x3 旋转矩阵（行优先，R[i] 是局部 i 轴在世界系的坐标）。"""
    return np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)


def _z_half_extent(geometry: tuple, orn) -> float:
    """物体沿世界 z 的半高（考虑当前姿态）。

    圆柱侧倒时沿 z 的半高会从 height/2 变成 radius，所以必须按姿态算，
    不能直接用建体时的尺寸。
    """
    kind = geometry[0]
    if kind == 'box':
        half = np.asarray(geometry[1], dtype=float)
        return float(np.abs(_quat_matrix(orn)[2, :]) @ half)
    if kind == 'cylinder':
        radius, height = float(geometry[1]), float(geometry[2])
        axis_z = abs(float(_quat_matrix(orn)[2, 2]))
        return axis_z * height / 2.0 + np.sqrt(max(0.0, 1.0 - axis_z * axis_z)) * radius
    if kind == 'sphere':
        return float(geometry[1])
    return 0.0


def _xy_half_extents(geometry: tuple, orn) -> np.ndarray:
    """物体在水平面上的半尺寸（用世界系 AABB 的半宽，粗但单调）。"""
    kind = geometry[0]
    R = _quat_matrix(orn)
    if kind == 'box':
        return np.abs(R[:2, :]) @ np.asarray(geometry[1], dtype=float)
    if kind == 'cylinder':
        radius, height = float(geometry[1]), float(geometry[2])
        axis = np.abs(R[:2, 2])
        return radius * np.sqrt(np.clip(1.0 - axis ** 2, 0.0, None)) + axis * height / 2.0
    if kind == 'sphere':
        return np.full(2, float(geometry[1]))
    return np.zeros(2)


def _quat_angle_deg(orn_a, orn_b) -> float:
    """两个姿态之间的最小旋转角（度）。"""
    dot = abs(float(np.dot(np.asarray(orn_a, float), np.asarray(orn_b, float))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def _yaw_deg(orn) -> float:
    return float(np.degrees(p.getEulerFromQuaternion(orn)[2]))


def _wrap_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


class BaseTask(ABC):
    """双臂灵巧操作任务的抽象基类"""

    def __init__(self, physics_client_id: int):
        self.client = physics_client_id
        self.robot_id: Optional[int] = None
        self.objects: Dict[str, int] = {}       # {物体名: pybullet ID}
        self.constraints: List[int] = []         # 约束ID列表
        self.success = False
        self.max_time = 300.0                    # 超时时间（秒）

        # 成功判定阈值
        self.pos_threshold = 0.05    # 位置阈值（米）
        self.angle_threshold = 5.0   # 角度阈值（度）

        # ---- 物理一致性 / 相对判据用的登记表（reset() 里清空重建） ----
        self._geometry: Dict[str, tuple] = {}     # {名字: ('box', half) / ('cylinder', r, h) / ('sphere', r)}
        self._support_exclude = set()             # 不作为支撑面的名字（壳体侧壁等）
        self._no_snap = set()                     # 不参与贴面吸附（已由关节定位的门/抽屉）
        self._links: Dict[str, int] = {}          # {名字: 关节 link 序号}（门/抽屉是多体的 link）
        self._link_offsets: Dict[str, np.ndarray] = {}   # {名字: 面板中心相对关节原点的偏移}
        self._extra_supports: List[dict] = []      # [{'rect': (x0,x1,y0,y1), 'top': z}] 内腔底面等
        self.jointed_objects = set()              # 有约束的物体（域随机化时不动它们）
        self.panel_joints: Dict[str, dict] = {}    # {名字: {'kind','pivot','axis','limits'}} 供作动探针用
        self.cavities: Dict[str, dict] = {}        # {名字: {'cavity_min':..,'cavity_max':..}}
        self.initial_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # 桌面高度 / 覆盖范围（load_fixed_table() 会覆盖）
        self.table_top_z = DEFAULT_TABLE_POSITION[2] + DEFAULT_TABLE_HALF_EXTENTS[2]
        self.table_rect = (
            DEFAULT_TABLE_POSITION[0] - DEFAULT_TABLE_HALF_EXTENTS[0],
            DEFAULT_TABLE_POSITION[0] + DEFAULT_TABLE_HALF_EXTENTS[0],
            DEFAULT_TABLE_POSITION[1] - DEFAULT_TABLE_HALF_EXTENTS[1],
            DEFAULT_TABLE_POSITION[1] + DEFAULT_TABLE_HALF_EXTENTS[1],
        )

    def set_robot(self, robot_id: int):
        """设置机器人ID"""
        self.robot_id = robot_id

    def reset(self):
        """重置场景到初始状态"""
        self._clear_objects()
        self._geometry.clear()
        self._support_exclude.clear()
        self._no_snap.clear()
        self._links.clear()
        self._link_offsets.clear()
        self._extra_supports.clear()
        self.jointed_objects.clear()
        self.panel_joints.clear()
        self.cavities.clear()
        self.initial_poses.clear()
        self._load_objects()
        self._set_initial_positions()
        # 贴面吸附：把「陷进桌面/支撑面」的物体抬到支撑面上（只抬不压），
        # 否则 PyBullet 会把它们弹飞再落地，位移全成了「自己动」。
        self.snap_to_supports()
        self.capture_initial_state()
        self.success = False

    def snap_to_supports(self, tolerance: float = SNAP_TOLERANCE):
        """把陷进支撑面的物体抬到支撑面上（只抬不压）。

        支撑面候选 = 桌面/内腔底面 + 其它物体的顶面。按当前高度从高到低处理，
        迭代两轮以处理「a 落在 b 上、b 又被抬起」的链式情况。
        """
        names = [n for n in self.objects
                 if n != 'table' and n not in self._no_snap and n in self._geometry]
        for _ in range(2):
            order = sorted(names, key=lambda n: self.get_object_position(n)[2], reverse=True)
            for name in order:
                self._lift_onto_supports(name, tolerance)

    def _lift_onto_supports(self, name: str, tolerance: float) -> None:
        pos, orn = self.get_object_pose(name)
        half_h = _z_half_extent(self._geometry[name], orn)
        half_xy = _xy_half_extents(self._geometry[name], orn)

        # 1) 桌面（只有落在桌面包围盒里才算）
        support_top = None
        if (self.table_top_z is not None
                and self.table_rect[0] <= pos[0] <= self.table_rect[1]
                and self.table_rect[2] <= pos[1] <= self.table_rect[3]):
            support_top = self.table_top_z

        # 2) 显式登记的内腔底面
        for entry in self._extra_supports:
            x0, x1, y0, y1 = entry['rect']
            if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1:
                if support_top is None or entry['top'] > support_top:
                    support_top = entry['top']

        # 3) 其它物体/家具的顶面（必须真的「在它下面」才算支撑）
        for other, geom in self._geometry.items():
            if other == name or other == 'table' or other in self._support_exclude:
                continue
            o_pos, o_orn = self.get_object_pose(other)
            o_top = o_pos[2] + _z_half_extent(geom, o_orn)
            if o_top > pos[2] + half_h - tolerance:
                continue
            o_half_xy = _xy_half_extents(geom, o_orn)
            if abs(o_pos[0] - pos[0]) > (o_half_xy[0] + half_xy[0] - 1e-4):
                continue
            if abs(o_pos[1] - pos[1]) > (o_half_xy[1] + half_xy[1] - 1e-4):
                continue
            if support_top is None or o_top > support_top:
                support_top = o_top

        if support_top is None:
            return
        target_z = support_top + half_h
        if pos[2] < target_z - tolerance:
            p.resetBasePositionAndOrientation(
                self.objects[name], [float(pos[0]), float(pos[1]), float(target_z)],
                list(map(float, orn)), physicsClientId=self.client)

    def capture_initial_state(self):
        """记录当前位姿为「初始位姿」，相对判据都以此为基准。"""
        self.initial_poses = {name: self.get_object_pose(name) for name in self.objects}

    # ---- 相对判据助手（比绝对阈值可靠：t=0 一定全为 0 / False） ----
    def initial_position(self, name: str) -> np.ndarray:
        return self.initial_poses[name][0]

    def displacement(self, name: str) -> float:
        """相对初始位置的位移（米）。"""
        return float(np.linalg.norm(self.get_object_position(name) - self.initial_position(name)))

    def rise(self, name: str) -> float:
        """相对初始位置的抬升量（米，可为负）。"""
        return float(self.get_object_position(name)[2] - self.initial_position(name)[2])

    def tilt_deg(self, name: str) -> float:
        """相对初始姿态的旋转角（度，取绝对值）。"""
        _, orn = self.get_object_pose(name)
        return _quat_angle_deg(orn, self.initial_poses[name][1])

    def yaw_change_deg(self, name: str) -> float:
        """相对初始姿态的偏航角变化（度，带符号，范围 [-180, 180]）。"""
        _, orn = self.get_object_pose(name)
        return _wrap_deg(_yaw_deg(orn) - _yaw_deg(self.initial_poses[name][1]))

    def yaw_deg(self, name: str) -> float:
        """物体当前偏航角（度）。"""
        _, orn = self.get_object_pose(name)
        return _yaw_deg(orn)

    def is_over(self, obj: str, anchor: str, tol: float = 0.03) -> bool:
        """obj 的水平投影是否落在 anchor 上（不管谁更高）。"""
        p_obj = self.get_object_position(obj)
        p_anchor = self.get_object_position(anchor)
        return bool(np.linalg.norm(p_obj[:2] - p_anchor[:2]) < tol)

    def is_inside(self, name: str, region: dict, tol: float = 0.01) -> bool:
        """物体中心是否落在 load_open_box() 返回的内腔里。"""
        pos = self.get_object_position(name)
        lo = np.asarray(region['cavity_min'], dtype=float)
        hi = np.asarray(region['cavity_max'], dtype=float)
        return bool(np.all(pos[:2] > lo[:2] - tol) and np.all(pos[:2] < hi[:2] + tol)
                    and lo[2] - tol <= pos[2] <= hi[2] + tol)

    def _clear_objects(self):
        """清除所有场景物体"""
        for obj_id in self.objects.values():
            p.removeBody(obj_id, physicsClientId=self.client)
        self.objects.clear()

        for c_id in self.constraints:
            p.removeConstraint(c_id, physicsClientId=self.client)
        self.constraints.clear()

    @abstractmethod
    def _load_objects(self):
        """加载任务所需物体（子类实现）"""
        pass

    @abstractmethod
    def _set_initial_positions(self):
        """设置物体初始位置（子类实现）"""
        pass

    @abstractmethod
    def check_success(self) -> bool:
        """检查任务是否完成（子类实现）"""
        return False

    def apply_action(self, joint_positions, joint_indices=None,
                     force: float = 100.0):
        """把关节角度指令下发到机器人

        Args:
            joint_positions: 目标角度序列（rad）
            joint_indices:   ★ 动作向量第 i 个元素对应的 pybullet 关节索引。
                             强烈建议显式传入（来自 RobotLoader.action_joint_indices）。
                             不传则退化为「第 i 个动作 -> 第 i 个关节」—— 这是危险行为。
            force:            关节电机力矩上限

        注意 历史坑（已修）：旧版本固定按索引 i 映射，导致
            action[0]（本意是左肩）被送到了 left_hip_yaw_joint（左髋＝腿），
            整套动作全部错位 —— 表现为「手臂不动、腿在动」。
        """
        if self.robot_id is None:
            raise RuntimeError("请先调用 set_robot(robot_id) 再 apply_action()")

        num_joints = p.getNumJoints(self.robot_id,
                                    physicsClientId=self.client)
        if joint_indices is None:
            joint_indices = list(range(len(joint_positions)))

        n = min(len(joint_positions), len(joint_indices))
        for k in range(n):
            jidx = int(joint_indices[k])
            if not (0 <= jidx < num_joints):
                continue
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=jidx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(joint_positions[k]),
                force=force,
                physicsClientId=self.client,
            )

    def get_object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """获取物体的位置和朝向（门/抽屉这类 link 也支持）"""
        if name not in self.objects:
            return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        if name in self._links:
            state = p.getLinkState(self.objects[name], self._links[name],
                                   computeForwardKinematics=True,
                                   physicsClientId=self.client)
            pos = np.array(state[4])
            orn = np.array(state[5])
            offset = self._link_offsets.get(name)
            if offset is not None:
                # link 原点在关节轴上（pivot），面板中心要按关节姿态偏出去
                pos = pos + _quat_matrix(orn) @ offset
            return pos, orn
        pos, orn = p.getBasePositionAndOrientation(
            self.objects[name], physicsClientId=self.client
        )
        return np.array(pos), np.array(orn)

    def get_object_position(self, name: str) -> np.ndarray:
        """获取物体位置"""
        pos, _ = self.get_object_pose(name)
        return pos

    def _distance_between(self, obj1: str, obj2: str) -> float:
        """两个物体之间的欧氏距离"""
        p1 = self.get_object_position(obj1)
        p2 = self.get_object_position(obj2)
        return np.linalg.norm(p1 - p2)

    def _is_above(self, obj_top: str, obj_base: str, threshold: float = 0.03) -> bool:
        """判断 obj_top 是否在 obj_base 上方"""
        p_top = self.get_object_position(obj_top)
        p_base = self.get_object_position(obj_base)
        xy_dist = np.linalg.norm(p_top[:2] - p_base[:2])
        z_diff = p_top[2] - p_base[2]
        return xy_dist < threshold and z_diff > 0

    def load_box(self, name: str, half_extents: List[float],
                 position: List[float], color: List[float] = None,
                 mass: float = 0.1, friction: float = 0.5) -> int:
        """加载方块物体"""
        if color is None:
            color = [0.8, 0.6, 0.2, 1.0]
        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents,
            physicsClientId=self.client
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=color, physicsClientId=self.client
        )
        body_id = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id, basePosition=position,
            physicsClientId=self.client
        )
        p.changeDynamics(body_id, -1, lateralFriction=friction,
                         physicsClientId=self.client)
        self.objects[name] = body_id
        self._geometry[name] = ('box', [float(h) for h in half_extents])
        return body_id

    def load_cylinder(self, name: str, radius: float, height: float,
                      position: List[float], color: List[float] = None,
                      mass: float = 0.1, friction: float = 0.5) -> int:
        """加载圆柱体物体"""
        if color is None:
            color = [0.5, 0.5, 0.8, 1.0]
        col_id = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height,
            physicsClientId=self.client
        )
        vis_id = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height,
            rgbaColor=color, physicsClientId=self.client
        )
        body_id = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id, basePosition=position,
            physicsClientId=self.client
        )
        p.changeDynamics(body_id, -1, lateralFriction=friction,
                         physicsClientId=self.client)
        self.objects[name] = body_id
        self._geometry[name] = ('cylinder', float(radius), float(height))
        return body_id

    def load_sphere(self, name: str, radius: float,
                    position: List[float], color: List[float] = None,
                    mass: float = 0.05, friction: float = 0.5) -> int:
        """加载球体物体"""
        if color is None:
            color = [1.0, 0.3, 0.3, 1.0]
        col_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius,
            physicsClientId=self.client
        )
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius,
            rgbaColor=color, physicsClientId=self.client
        )
        body_id = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id, basePosition=position,
            physicsClientId=self.client
        )
        p.changeDynamics(body_id, -1, lateralFriction=friction,
                         physicsClientId=self.client)
        self.objects[name] = body_id
        self._geometry[name] = ('sphere', float(radius))
        return body_id

    def load_fixed_table(self, position: List[float] = None,
                         half_extents: List[float] = None):
        """加载固定操作台"""
        if position is None:
            position = [0.5, 0, 0.4]
        if half_extents is None:
            half_extents = [0.5, 0.8, 0.02]
        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents,
            physicsClientId=self.client
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.6, 0.4, 0.2, 1.0], physicsClientId=self.client
        )
        body_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id, basePosition=position,
            physicsClientId=self.client
        )
        self.objects['table'] = body_id
        # 桌面高度与覆盖范围：snap_to_supports() 用它们判断「谁该落在哪儿」
        self.table_top_z = float(position[2] + half_extents[2])
        self.table_rect = (
            float(position[0] - half_extents[0]), float(position[0] + half_extents[0]),
            float(position[1] - half_extents[1]), float(position[1] + half_extents[1]),
        )
        return body_id

    # ------------------------------------------------------------------
    # 物理一致性的场景构件：开口容器 / 竖直轴铰链门 / 水平滑轨抽屉
    # ------------------------------------------------------------------
    def load_open_box(self, name: str, outer_size: List[float], position: List[float],
                      wall: float = 0.01, open_side: str = 'z+',
                      color: List[float] = None) -> dict:
        """用 5 块薄板拼一个上方开口的容器（PyBullet 的 box 没有内腔）。

        没有它，「把盘子放进烤箱」「把书放进抽屉」这类任务物理上不可达 ——
        物体只会停在实心方块顶上。

        Args:
            outer_size: 外廓全尺寸 [sx, sy, sz]（米）
            position:   容器中心
            wall:       板厚
            open_side:  只有 'z+'（上方开口）被实现
        Returns:
            {'cavity_min': [x, y, z], 'cavity_max': [x, y, z]}  世界坐标内腔范围
        """
        if open_side != 'z+':
            raise ValueError("load_open_box 目前只支持上方开口 (open_side='z+')")
        if color is None:
            color = [0.55, 0.55, 0.55, 1.0]
        sx, sy, sz = (float(v) for v in outer_size)
        cx, cy, cz = (float(v) for v in position)
        half = np.array([sx / 2.0, sy / 2.0, sz / 2.0])

        panels = [
            # 底板
            ([half[0], half[1], wall / 2.0],
             [cx, cy, cz - half[2] + wall / 2.0]),
            # 左右侧板（沿 y 展开）
            ([wall / 2.0, half[1], half[2]],
             [cx - half[0] + wall / 2.0, cy, cz]),
            ([wall / 2.0, half[1], half[2]],
             [cx + half[0] - wall / 2.0, cy, cz]),
            # 前后侧板（沿 x 展开，扣掉左右板占的角）
            ([half[0] - wall, wall / 2.0, half[2]],
             [cx, cy - half[1] + wall / 2.0, cz]),
            ([half[0] - wall, wall / 2.0, half[2]],
             [cx, cy + half[1] - wall / 2.0, cz]),
        ]
        for index, (panel_half, panel_pos) in enumerate(panels):
            panel_name = f'{name}_panel{index}'
            self.load_box(panel_name, panel_half, panel_pos, color=color, mass=0.0)
            # 侧壁顶面不是支撑面（否则内腔里的物体会被抬到墙顶上）
            self._support_exclude.add(panel_name)

        cavity = {
            'cavity_min': [cx - half[0] + wall, cy - half[1] + wall,
                           cz - half[2] + wall],
            'cavity_max': [cx + half[0] - wall, cy + half[1] - wall,
                           cz + half[2]],
        }
        self.cavities[name] = cavity
        # 内腔底面作为支撑面登记（物体落进腔内时停在底板上，而不是穿透底板）
        self._extra_supports.append({
            'rect': (cavity['cavity_min'][0], cavity['cavity_max'][0],
                     cavity['cavity_min'][1], cavity['cavity_max'][1]),
            'top': cavity['cavity_min'][2],
        })
        return cavity

    def load_hinged_panel(self, name: str, half_extents: List[float],
                          position: List[float], pivot: List[float],
                          yaw_deg: float = 0.0, color: List[float] = None,
                          mass: float = 0.1, friction: float = 0.5,
                          hold_force: float = 0.5, axis=(0, 0, 1),
                          ignore_collision_with=None) -> int:
        """造一块带铰链的面板（微波炉门/柜门/笔记本屏/水龙头手柄）。

        实现：`createMultiBody` 一个「质量 0 的基座 + 1 个 REVOLUTE link」，
        基座放在 pivot 上，link 就是面板本身。
        ★ 为什么不用 p.createConstraint：本机 pybullet (202010061) 的
          createConstraint 只接受 JOINT_PRISMATIC / JOINT_FIXED，REVOLUTE 会
          直接 "createConstraint failed"（tmp_arm_main/probe_constraint4.py）。
        ★ 轴默认竖直 (0,0,1)：轴与重力平行 -> 重力对门不产生力矩，静止时不会
          自己开也不会自己关，必须由机器人推。旧版门是自由刚体，静置就被挤出
          0.28 m。水平轴（笔记本屏）靠 hold_force 当合页摩擦顶住重力。

        Args:
            position:    面板中心（yaw_deg=0，即「关闭」状态时的位置）
            pivot:       铰链轴经过的世界坐标点（通常是面板的一条竖边中心）
            yaw_deg:     初始关节角；非 0 表示「门一开始是开着/关着的」
            hold_force:  关节电机保持力矩（N·m），近似合页摩擦；0 = 不加电机
        """
        if color is None:
            color = [0.5, 0.5, 0.5, 1.0]
        half = [float(v) for v in half_extents]
        link_offset = [float(position[i]) - float(pivot[i]) for i in range(3)]
        base_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[1e-3, 1e-3, 1e-3], physicsClientId=self.client)
        # ★ 面板的几何整体偏到关节原点之外（linkPositions 保持 0），
        #   这样转起来才是「绕合页转」，而不是「绕面板自己中心转」
        link_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half, collisionFramePosition=link_offset,
            physicsClientId=self.client)
        link_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, visualFramePosition=link_offset,
            rgbaColor=color, physicsClientId=self.client)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=-1,
            basePosition=[float(v) for v in pivot],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=[float(mass)],
            linkCollisionShapeIndices=[link_collision],
            linkVisualShapeIndices=[link_visual],
            linkPositions=[[0.0, 0.0, 0.0]],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[link_offset],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[[float(v) for v in axis]],
            physicsClientId=self.client,
        )
        p.changeDynamics(body_id, 0, lateralFriction=float(friction),
                         physicsClientId=self.client)
        joint_angle = float(np.radians(yaw_deg))
        p.resetJointState(body_id, 0, targetValue=joint_angle, targetVelocity=0.0,
                          physicsClientId=self.client)
        if hold_force:
            p.setJointMotorControl2(body_id, 0, p.POSITION_CONTROL,
                                    targetPosition=joint_angle,
                                    force=float(hold_force),
                                    physicsClientId=self.client)
        self.objects[name] = body_id
        self._links[name] = 0
        self._geometry[name] = ('box', half)
        self.jointed_objects.add(name)
        self._no_snap.add(name)
        self.panel_joints[name] = {
            'kind': 'revolute',
            'pivot': np.asarray(pivot, dtype=float),
            'axis': np.asarray(axis, dtype=float),
        }
        self._link_offsets[name] = np.asarray(link_offset, dtype=float)
        for other in (ignore_collision_with or ()):
            if other in self.objects:
                p.setCollisionFilterPair(body_id, self.objects[other], 0, -1, 0,
                                         physicsClientId=self.client)
        return body_id

    def load_sliding_panel(self, name: str, half_extents: List[float],
                           position: List[float], axis=(1, 0, 0),
                           limits: Tuple[float, float] = (0.0, 0.2),
                           color: List[float] = None,
                           mass: float = 0.15, friction: float = 0.5,
                           ignore_collision_with=None) -> int:
        """造一块带滑轨的面板（抽屉/压杆）：质量 0 的基座 + 1 个 PRISMATIC link。

        ★ 轨道水平：重力沿 z，与轨道垂直，抽屉不会自己滑出/滑回，必须由
          机器人推。行程用 changeDynamics 的 jointLowerLimit/jointUpperLimit 限定。
        Args:
            limits: (下限, 上限) 相对 position 的位移，米。开抽屉用 (0, 0.2)，
                    关抽屉用 (-0.2, 0)。
        """
        if color is None:
            color = [0.7, 0.4, 0.2, 1.0]
        half = [float(v) for v in half_extents]
        base_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[1e-3, 1e-3, 1e-3], physicsClientId=self.client)
        link_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        link_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=-1,
            basePosition=[float(v) for v in position],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=[float(mass)],
            linkCollisionShapeIndices=[link_collision],
            linkVisualShapeIndices=[link_visual],
            linkPositions=[[0.0, 0.0, 0.0]],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_PRISMATIC],
            linkJointAxis=[[float(v) for v in axis]],
            physicsClientId=self.client,
        )
        p.changeDynamics(body_id, 0, lateralFriction=float(friction),
                         jointLowerLimit=float(min(limits)),
                         jointUpperLimit=float(max(limits)),
                         physicsClientId=self.client)
        self.objects[name] = body_id
        self._links[name] = 0
        self._geometry[name] = ('box', half)
        self.jointed_objects.add(name)
        self._no_snap.add(name)
        self.panel_joints[name] = {
            'kind': 'prismatic',
            'pivot': np.asarray(position, dtype=float),
            'axis': np.asarray(axis, dtype=float),
            'limits': (float(min(limits)), float(max(limits))),
        }
        for other in (ignore_collision_with or ()):
            if other in self.objects:
                p.setCollisionFilterPair(body_id, self.objects[other], 0, -1, 0,
                                         physicsClientId=self.client)
        return body_id
