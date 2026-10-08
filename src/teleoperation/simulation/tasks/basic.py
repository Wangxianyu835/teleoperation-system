import numpy as np
import pybullet as p
from .base import BaseTask
from .registry import register_task
@register_task('pushcube')
class PushCube(BaseTask):
    """将方块推到蓝色目标区域"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cube', [0.03, 0.03, 0.03], [0, 0, 0],
                      color=[0.8, 0.4, 0.2, 1.0], mass=0.1, friction=0.4)
        # 目标区域
        self.target_center = np.array([0.3, 0.3, 0.43])
        self.load_box('target', [0.05, 0.05, 0.005], self.target_center,
                      color=[0.2, 0.4, 0.8, 0.5], mass=0)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cube'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._distance_between('cube', 'target') < self.pos_threshold


@register_task('pickcube')
class PickCube(BaseTask):
    """拿起方块并放下（提升>15cm）"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cube', [0.03, 0.03, 0.03], [0, 0, 0],
                      color=[0.9, 0.5, 0.1, 1.0], mass=0.08, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cube'], [0.5, 0, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        pos = self.get_object_position('cube')
        return pos[2] > 0.66  # 桌高0.4 + 方块半高0.03 + 提升0.15 ~= 0.58+


@register_task('pickplacecube')
class PickPlaceCube(BaseTask):
    """拿起方块放到盘子上"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cube', [0.03, 0.03, 0.03], [0, 0, 0],
                      color=[0.9, 0.5, 0.1, 1.0], mass=0.08, friction=0.5)
        # 盘子
        self.load_cylinder('plate', 0.08, 0.01, [0, 0, 0],
                           color=[0.9, 0.9, 0.9, 1.0], mass=0.05, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cube'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['plate'], [0.3, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._is_above('cube', 'plate', threshold=0.05)


@register_task('uprearcup')
class UpRearCup(BaseTask):
    """拿起杯子并竖直放置"""
    def _load_objects(self):
        self.load_fixed_table()
        # 杯子：一个圆柱体
        self.load_cylinder('cup', 0.04, 0.1, [0, 0, 0],
                           color=[0.3, 0.7, 0.3, 1.0], mass=0.08, friction=0.5)

    def _set_initial_positions(self):
        # 杯子侧放（绕Y轴旋转90度）
        p.resetBasePositionAndOrientation(
            self.objects['cup'], [0.5, 0, 0.43],
            p.getQuaternionFromEuler([0, np.pi / 2, 0]),
            physicsClientId=self.client)

    def check_success(self) -> bool:
        _, orn = self.get_object_pose('cup')
        euler = np.array(p.getEulerFromQuaternion(orn))
        # 杯子应该直立（roll和pitch接近0）
        return abs(euler[0]) < 0.2 and abs(euler[1]) < 0.2 and self.get_object_position('cup')[2] > 0.42


@register_task('balltrashcan')
class BallTrashCan(BaseTask):
    """拿起球放入垃圾桶"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_sphere('ball', 0.03, [0, 0, 0],
                         color=[1.0, 0.2, 0.2, 1.0], mass=0.03, friction=0.5)
        # 垃圾桶：空心圆柱（简化：一个开口向上的容器）
        self.load_cylinder('trashcan', 0.06, 0.15, [0, 0, 0],
                           color=[0.3, 0.3, 0.3, 1.0], mass=0.1, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['ball'], [0.5, -0.3, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['trashcan'], [0.6, 0.4, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_ball = self.get_object_position('ball')
        p_can = self.get_object_position('trashcan')
        xy_dist = np.linalg.norm(p_ball[:2] - p_can[:2])
        return xy_dist < 0.04 and p_ball[2] < p_can[2] + 0.08
