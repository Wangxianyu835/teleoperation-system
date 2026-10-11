import numpy as np
import pybullet as p
from .base import BaseTask
from .registry import register_task
@register_task('rotatefaucet')
class RotateFaucet(BaseTask):
    """旋转水龙头90度"""
    def _load_objects(self):
        self.load_fixed_table()
        # 水龙头基座
        self.load_cylinder('base', 0.03, 0.08, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0)
        # 水龙头手柄
        self.load_box('handle', [0.08, 0.015, 0.015], [0, 0, 0],
                      color=[0.7, 0.7, 0.7, 1.0], mass=0.05, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['base'], [0.5, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['handle'], [0.5, 0.3, 0.46],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client)

    def check_success(self) -> bool:
        _, orn = self.get_object_pose('handle')
        euler = np.array(p.getEulerFromQuaternion(orn))
        return abs(euler[2]) > np.pi / 2 - 0.1  # 绕Z轴旋转>90度


@register_task('rotatehearth')
class RotateHearth(BaseTask):
    """按下并旋转炉灶旋钮"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('knob', 0.025, 0.04, [0, 0, 0],
                           color=[0.2, 0.2, 0.2, 1.0], mass=0.05, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['knob'], [0.5, 0, 0.44], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        pos = self.get_object_position('knob')
        _, orn = self.get_object_pose('knob')
        euler = np.array(p.getEulerFromQuaternion(orn))
        pressed = pos[2] < 0.43  # 按下
        rotated = abs(euler[2]) > 0.5  # 旋转>30度
        return pressed and rotated


@register_task('openmicrowave')
class OpenMicrowave(BaseTask):
    """拉开微波炉门"""
    def _load_objects(self):
        self.load_fixed_table()
        # 微波炉本体
        self.load_box('microwave_body', [0.15, 0.2, 0.15], [0, 0, 0],
                      color=[0.8, 0.8, 0.8, 1.0], mass=0)
        # 微波炉门（铰链连接）
        self.load_box('door', [0.01, 0.2, 0.15], [0, 0, 0],
                      color=[0.5, 0.5, 0.5, 1.0], mass=0.2, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['microwave_body'], [0.5, 0.3, 0.5], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['door'], [0.65, 0.3, 0.5], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_body = self.get_object_position('microwave_body')
        p_door = self.get_object_position('door')
        return abs(p_door[0] - p_body[0]) > 0.12  # 门拉开距离


@register_task('closemicrowave')
class CloseMicrowave(BaseTask):
    """关上微波炉门"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('microwave_body', [0.15, 0.2, 0.15], [0, 0, 0],
                      color=[0.8, 0.8, 0.8, 1.0], mass=0)
        self.load_box('door', [0.01, 0.2, 0.15], [0, 0, 0],
                      color=[0.5, 0.5, 0.5, 1.0], mass=0.2, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['microwave_body'], [0.5, 0.3, 0.5], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 门初始打开
        p.resetBasePositionAndOrientation(
            self.objects['door'], [0.8, 0.3, 0.5], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_body = self.get_object_position('microwave_body')
        p_door = self.get_object_position('door')
        return abs(p_door[0] - p_body[0]) < 0.02


@register_task('opendrawer')
class OpenDrawer(BaseTask):
    """拉开抽屉"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cabinet', [0.15, 0.2, 0.25], [0, 0, 0],
                      color=[0.6, 0.3, 0.1, 1.0], mass=0)
        self.load_box('drawer', [0.14, 0.18, 0.1], [0, 0, 0],
                      color=[0.7, 0.4, 0.2, 1.0], mass=0.15, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cabinet'], [0.4, 0.3, 0.3], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['drawer'], [0.54, 0.3, 0.38], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_cabinet = self.get_object_position('cabinet')
        p_drawer = self.get_object_position('drawer')
        return abs(p_drawer[0] - p_cabinet[0]) > 0.12


@register_task('closedrawer')
class CloseDrawer(BaseTask):
    """关上抽屉"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cabinet', [0.15, 0.2, 0.25], [0, 0, 0],
                      color=[0.6, 0.3, 0.1, 1.0], mass=0)
        self.load_box('drawer', [0.14, 0.18, 0.1], [0, 0, 0],
                      color=[0.7, 0.4, 0.2, 1.0], mass=0.15, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cabinet'], [0.4, 0.3, 0.3], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 抽屉初始打开
        p.resetBasePositionAndOrientation(
            self.objects['drawer'], [0.7, 0.3, 0.38], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_cabinet = self.get_object_position('cabinet')
        p_drawer = self.get_object_position('drawer')
        return abs(p_drawer[0] - p_cabinet[0]) < 0.17


@register_task('liftmug')
class LiftMug(BaseTask):
    """提起杯盖"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('mug_body', 0.05, 0.1, [0, 0, 0],
                           color=[0.3, 0.6, 0.3, 1.0], mass=0.1, friction=0.6)
        self.load_cylinder('lid', 0.052, 0.01, [0, 0, 0],
                           color=[0.2, 0.5, 0.2, 1.0], mass=0.02, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['mug_body'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, 0, 0.475], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_lid = self.get_object_position('lid')
        p_mug = self.get_object_position('mug_body')
        return abs(p_lid[2] - p_mug[2]) > 0.05


@register_task('openlaptop')
class OpenLaptop(BaseTask):
    """打开笔记本电脑"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('laptop_base', [0.15, 0.1, 0.01], [0, 0, 0],
                      color=[0.3, 0.3, 0.3, 1.0], mass=0.5, friction=0.8)
        self.load_box('laptop_screen', [0.15, 0.1, 0.008], [0, 0, 0],
                      color=[0.1, 0.1, 0.2, 1.0], mass=0.3, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['laptop_base'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['laptop_screen'], [0.5, 0.1, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        _, orn = self.get_object_pose('laptop_screen')
        euler = np.array(p.getEulerFromQuaternion(orn))
        return abs(euler[0]) > np.pi / 3  # 屏幕打开>60度


@register_task('ballmug')
class BallMug(BaseTask):
    """拿起球放入杯子"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_sphere('ball', 0.025, [0, 0, 0],
                         color=[1.0, 0.3, 0.3, 1.0], mass=0.02, friction=0.5)
        self.load_cylinder('mug', 0.05, 0.1, [0, 0, 0],
                           color=[0.3, 0.6, 0.3, 1.0], mass=0.1, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['ball'], [0.5, -0.3, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['mug'], [0.3, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_ball = self.get_object_position('ball')
        p_mug = self.get_object_position('mug')
        xy_dist = np.linalg.norm(p_ball[:2] - p_mug[:2])
        return xy_dist < 0.03 and p_ball[2] > p_mug[2] + 0.04


@register_task('pourwater')
class PourWater(BaseTask):
    """从水壶倒水到杯子（检测倾斜角度）"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('kettle', 0.05, 0.12, [0, 0, 0],
                           color=[0.1, 0.5, 0.8, 1.0], mass=0.15, friction=0.6)
        self.load_cylinder('cup', 0.04, 0.08, [0, 0, 0],
                           color=[0.3, 0.7, 0.3, 1.0], mass=0.06, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['kettle'], [0.5, -0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['cup'], [0.35, 0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_kettle = self.get_object_position('kettle')
        _, orn = self.get_object_pose('kettle')
        euler = np.array(p.getEulerFromQuaternion(orn))
        return abs(euler[0]) > np.pi / 3 and p_kettle[2] > 0.5


@register_task('breadtoaster')
class BreadToaster(BaseTask):
    """把面包放入烤面包机并按下按钮"""
    def _load_objects(self):
        self.load_fixed_table()
        # 烤面包机
        self.load_box('toaster', [0.06, 0.1, 0.1], [0, 0, 0],
                      color=[0.7, 0.7, 0.7, 1.0], mass=0.3, friction=0.8)
        # 面包
        self.load_box('bread', [0.04, 0.04, 0.01], [0, 0, 0],
                      color=[0.9, 0.8, 0.5, 1.0], mass=0.02, friction=0.5)
        # 按钮
        self.load_cylinder('button', 0.01, 0.02, [0, 0, 0],
                           color=[1.0, 0, 0, 1.0], mass=0.01, friction=0.3)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['toaster'], [0.5, 0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['bread'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['button'], [0.5, 0.2, 0.48], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        # 面包在土司机上方 = 放入面包
        bread_above = self._is_above('bread', 'toaster', threshold=0.04)
        # 按钮被按下
        button_pos = self.get_object_position('button')
        button_pressed = button_pos[2] < 0.47
        return bread_above and button_pressed
