"""任务基类 - 所有30个任务都继承此类"""

import numpy as np
import pybullet as p
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


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

    def set_robot(self, robot_id: int):
        """设置机器人ID"""
        self.robot_id = robot_id

    def reset(self):
        """重置场景到初始状态"""
        self._clear_objects()
        self._load_objects()
        self._set_initial_positions()
        self.success = False

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

    def apply_action(self, joint_positions: List[float]):
        """应用关节角度到机器人"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(min(len(joint_positions), num_joints)):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=100,
                physicsClientId=self.client,
            )

    def get_object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """获取物体的位置和朝向"""
        if name not in self.objects:
            return np.zeros(3), np.array([0, 0, 0, 1])
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
        return body_id