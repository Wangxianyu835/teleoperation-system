import numpy as np
import pybullet as p
from .base import BaseTask
from .registry import register_task
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
