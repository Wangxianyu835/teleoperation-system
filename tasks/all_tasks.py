"""全部30个分层任务实现"""

import numpy as np
import pybullet as p
from .base_task import BaseTask
from .task_registry import register_task


# ============================================================
# Level 1: 基础拾取放置 (5个任务)
# ============================================================

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


# ============================================================
# Level 2: 工具操作 (11个任务)
# ============================================================

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


# ============================================================
# Level 3: 双手协作 (9个任务)
# ============================================================

@register_task('ballbimanual')
class BallBimanual(BaseTask):
    """将球从一只手传递到另一只手"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_sphere('ball', 0.03, [0, 0, 0],
                         color=[1.0, 0.2, 0.2, 1.0], mass=0.03, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['ball'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        # 球在中间位置且高度足够（被传递中）
        pos = self.get_object_position('ball')
        return abs(pos[1]) < 0.1 and pos[2] > 0.6


@register_task('potbimanual')
class PotBimanual(BaseTask):
    """双手抬起锅"""
    def _load_objects(self):
        self.load_fixed_table()
        # 锅本体
        self.load_cylinder('pot', 0.08, 0.06, [0, 0, 0],
                           color=[0.4, 0.4, 0.4, 1.0], mass=0.3, friction=0.5)
        # 两个手柄
        self.load_box('handle_l', [0.02, 0.04, 0.02], [0, 0, 0],
                      color=[0.3, 0.3, 0.3, 1.0], mass=0.02, friction=0.6)
        self.load_box('handle_r', [0.02, 0.04, 0.02], [0, 0, 0],
                      color=[0.3, 0.3, 0.3, 1.0], mass=0.02, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['pot'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['handle_l'], [0.42, 0, 0.44], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['handle_r'], [0.58, 0, 0.44], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_pot = self.get_object_position('pot')
        # 锅提升超过15cm
        return p_pot[2] > 0.58


@register_task('pottomato')
class PotTomato(BaseTask):
    """抬起锅并加入番茄"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('pot', 0.08, 0.06, [0, 0, 0],
                           color=[0.4, 0.4, 0.4, 1.0], mass=0.3, friction=0.5)
        self.load_sphere('tomato', 0.02, [0, 0, 0],
                         color=[1.0, 0, 0, 1.0], mass=0.02, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['pot'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['tomato'], [0.5, -0.3, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._is_above('tomato', 'pot', threshold=0.04)


@register_task('pottray')
class PotTray(BaseTask):
    """双手将锅放到托盘上"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('pot', 0.08, 0.06, [0, 0, 0],
                           color=[0.4, 0.4, 0.4, 1.0], mass=0.3, friction=0.5)
        self.load_box('tray', [0.15, 0.1, 0.01], [0, 0, 0],
                      color=[0.8, 0.8, 0.6, 1.0], mass=0.08, friction=0.7)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['pot'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['tray'], [0.3, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._is_above('pot', 'tray', threshold=0.06)


@register_task('stackboxes')
class StackBoxes(BaseTask):
    """箱子叠到另一个箱子上"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('box_a', [0.05, 0.05, 0.05], [0, 0, 0],
                      color=[0.2, 0.6, 0.2, 1.0], mass=0.1, friction=0.6)
        self.load_box('box_b', [0.05, 0.05, 0.05], [0, 0, 0],
                      color=[0.6, 0.2, 0.2, 1.0], mass=0.1, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['box_a'], [0.5, -0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['box_b'], [0.5, 0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_a = self.get_object_position('box_a')
        p_b = self.get_object_position('box_b')
        xy_dist = np.linalg.norm(p_a[:2] - p_b[:2])
        z_diff = abs(p_a[2] - p_b[2])
        return xy_dist < 0.04 and z_diff > 0.04 and z_diff < 0.07


@register_task('panhearth')
class PanHearth(BaseTask):
    """打开盖子把平底锅放到炉灶上"""
    def _load_objects(self):
        self.load_fixed_table()
        # 炉灶
        self.load_box('hearth', [0.12, 0.12, 0.03], [0, 0, 0],
                      color=[0.2, 0.2, 0.2, 1.0], mass=0)
        self.load_cylinder('pan', 0.07, 0.02, [0, 0, 0],
                           color=[0.3, 0.3, 0.3, 1.0], mass=0.15, friction=0.6)
        self.load_cylinder('lid', 0.072, 0.01, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0.03, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['hearth'], [0.5, 0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['pan'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, -0.2, 0.46], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        pan_on_hearth = self._is_above('pan', 'hearth', threshold=0.05)
        # 盖子被移开
        p_pan = self.get_object_position('pan')
        p_lid = self.get_object_position('lid')
        lid_off = np.linalg.norm(p_pan[:2] - p_lid[:2]) > 0.05
        return pan_on_hearth and lid_off


@register_task('tidyuptable')
class TidyUpTable(BaseTask):
    """按顺序放入篮子"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('basket', 0.1, 0.1, [0, 0, 0],
                           color=[0.6, 0.4, 0.2, 1.0], mass=0.1, friction=0.7)
        self.load_sphere('item1', 0.02, [0, 0, 0],
                         color=[1.0, 0.5, 0, 1.0], mass=0.02, friction=0.5)
        self.load_box('item2', [0.025, 0.025, 0.025], [0, 0, 0],
                      color=[0, 0.5, 1.0, 1.0], mass=0.03, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['basket'], [0.3, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['item1'], [0.5, -0.3, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['item2'], [0.5, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        item1_in = self._is_above('item1', 'basket', threshold=0.06)
        item2_in = self._is_above('item2', 'basket', threshold=0.06)
        return item1_in and item2_in


@register_task('pottomatoout')
class PotTomatoOut(BaseTask):
    """打开锅盖取出番茄放到桌面"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('pot', 0.08, 0.06, [0, 0, 0],
                           color=[0.4, 0.4, 0.4, 1.0], mass=0.3, friction=0.6)
        self.load_cylinder('lid', 0.082, 0.01, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0.03, friction=0.5)
        self.load_sphere('tomato', 0.02, [0, 0, 0],
                         color=[1.0, 0, 0, 1.0], mass=0.02, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['pot'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, 0, 0.455], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['tomato'], [0.5, 0, 0.44], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_tomato = self.get_object_position('tomato')
        # tomato离开锅的区域（不在锅上方）
        p_pot = self.get_object_position('pot')
        xy_dist = np.linalg.norm(p_tomato[:2] - p_pot[:2])
        return xy_dist > 0.06 and p_tomato[2] > 0.42


@register_task('plateoven')
class PlateOven(BaseTask):
    """打开烤箱放入盘子再关上"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('oven', [0.2, 0.2, 0.2], [0, 0, 0],
                      color=[0.5, 0.5, 0.5, 1.0], mass=0)
        self.load_box('oven_door', [0.01, 0.2, 0.2], [0, 0, 0],
                      color=[0.4, 0.4, 0.4, 1.0], mass=0.2, friction=0.5)
        self.load_cylinder('plate', 0.08, 0.01, [0, 0, 0],
                           color=[0.9, 0.9, 0.9, 1.0], mass=0.05, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['oven'], [0.5, 0.3, 0.45], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['oven_door'], [0.7, 0.3, 0.45], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['plate'], [0.5, -0.3, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_oven = self.get_object_position('oven')
        p_door = self.get_object_position('oven_door')
        p_plate = self.get_object_position('plate')
        # 盘子在里面
        plate_in = (abs(p_plate[0] - p_oven[0]) < 0.1 and
                    abs(p_plate[1] - p_oven[1]) < 0.1)
        # 门关上
        door_closed = abs(p_door[0] - p_oven[0]) < 0.22
        return plate_in and door_closed


# ============================================================
# Level 4: 长时域序列 (5个任务)
# ============================================================

@register_task('pottomatoplate')
class PotTomatoPlate(BaseTask):
    """打开盖子将番茄放到盘子上"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('pot', 0.08, 0.06, [0, 0, 0],
                           color=[0.4, 0.4, 0.4, 1.0], mass=0.3, friction=0.6)
        self.load_cylinder('lid', 0.082, 0.01, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0.03, friction=0.5)
        self.load_sphere('tomato', 0.02, [0, 0, 0],
                         color=[1.0, 0, 0, 1.0], mass=0.02, friction=0.5)
        self.load_cylinder('plate', 0.08, 0.01, [0, 0, 0],
                           color=[0.9, 0.9, 0.9, 1.0], mass=0.05, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['pot'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, 0, 0.455], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['tomato'], [0.5, 0, 0.44], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['plate'], [0.3, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._is_above('tomato', 'plate', threshold=0.04)


@register_task('penbrushpot')
class PenBrushPot(BaseTask):
    """拿起笔放入笔筒"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('pen', 0.01, 0.12, [0, 0, 0],
                           color=[0.1, 0.1, 0.8, 1.0], mass=0.01, friction=0.5)
        self.load_cylinder('pot_holder', 0.04, 0.1, [0, 0, 0],
                           color=[0.3, 0.3, 0.3, 1.0], mass=0.08, friction=0.7)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['pen'], [0.5, -0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['pot_holder'], [0.3, 0.3, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._is_above('pen', 'pot_holder', threshold=0.03)


@register_task('drawerbook')
class DrawerBook(BaseTask):
    """打开抽屉放入书本再关上"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cabinet', [0.15, 0.2, 0.25], [0, 0, 0],
                      color=[0.6, 0.3, 0.1, 1.0], mass=0)
        self.load_box('drawer', [0.14, 0.18, 0.1], [0, 0, 0],
                      color=[0.7, 0.4, 0.2, 1.0], mass=0.15, friction=0.5)
        self.load_box('book', [0.06, 0.09, 0.015], [0, 0, 0],
                      color=[0.2, 0.2, 0.6, 1.0], mass=0.15, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cabinet'], [0.4, 0.3, 0.3], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['drawer'], [0.54, 0.3, 0.38], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['book'], [0.5, -0.3, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_drawer = self.get_object_position('drawer')
        p_cabinet = self.get_object_position('cabinet')
        p_book = self.get_object_position('book')
        # 书在抽屉上方
        book_on = self._is_above('book', 'drawer', threshold=0.06)
        # 抽屉关上了
        drawer_closed = abs(p_drawer[0] - p_cabinet[0]) < 0.17
        return book_on and drawer_closed


@register_task('twistbottlecaps')
class TwistBottleCaps(BaseTask):
    """拿起瓶子拧上瓶盖"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('bottle', 0.03, 0.12, [0, 0, 0],
                           color=[0.2, 0.6, 0.8, 1.0], mass=0.1, friction=0.6)
        self.load_cylinder('cap', 0.032, 0.02, [0, 0, 0],
                           color=[0.8, 0.2, 0.2, 1.0], mass=0.01, friction=0.5)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['bottle'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['cap'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self._is_above('cap', 'bottle', threshold=0.02)


@register_task('stacktoyblocks')
class StackToyBlocks(BaseTask):
    """按顺序组装积木"""
    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('block_a', [0.04, 0.04, 0.04], [0, 0, 0],
                      color=[1.0, 0.8, 0, 1.0], mass=0.08, friction=0.6)
        self.load_box('block_b', [0.04, 0.04, 0.04], [0, 0, 0],
                      color=[0, 0.8, 1.0, 1.0], mass=0.08, friction=0.6)
        self.load_box('block_c', [0.04, 0.04, 0.04], [0, 0, 0],
                      color=[0.8, 0, 1.0, 1.0], mass=0.08, friction=0.6)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['block_a'], [0.5, -0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['block_b'], [0.5, 0, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['block_c'], [0.5, 0.2, 0.42], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        p_a = self.get_object_position('block_a')
        p_b = self.get_object_position('block_b')
        p_c = self.get_object_position('block_c')
        # 三个块近似垂直堆叠
        xy_dist = np.linalg.norm(p_a[:2] - p_b[:2]) + np.linalg.norm(p_b[:2] - p_c[:2])
        z_a, z_b, z_c = p_a[2], p_b[2], p_c[2]
        stacked = (max(z_a, z_b, z_c) - min(z_a, z_b, z_c)) > 0.06
        return xy_dist < 0.08 and stacked