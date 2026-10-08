import numpy as np
import pybullet as p
from .base import BaseTask
from .registry import register_task
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
