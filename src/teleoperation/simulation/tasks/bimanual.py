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
        # 两个手柄贴在锅两侧的桌面上（锅半径 0.08，留 1 cm 间隙避免穿模）
        p.resetBasePositionAndOrientation(
            self.objects['handle_l'], [0.39, 0, 0.44], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['handle_r'], [0.61, 0, 0.44], [0, 0, 0, 1],
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
        # 锅盖压在平底锅上（锅顶面 0.44 + 盖半厚 0.005）
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, -0.2, 0.445], [0, 0, 0, 1],
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
    """打开锅盖把番茄从锅里拿出来（放到桌面上）

    旧版锅是实心圆柱、番茄摆在锅心（z=0.44）—— 物理上「在锅里」是不可能的，
    静置时番茄被挤出锅外，判据（离锅心 >6 cm）可能被这段乱动满足。
    新版锅是上方开口的壳：番茄真的在锅内，盖子真的压在锅沿上。
    """
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('lid', 0.082, 0.01, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0.03, friction=0.5)
        self.load_sphere('tomato', 0.02, [0, 0, 0],
                         color=[1.0, 0, 0, 1.0], mass=0.02, friction=0.5)

    def _set_initial_positions(self):
        # 锅：外廓 0.16 x 0.16 x 0.08，底面贴桌面 0.42，上方开口
        self.pot = self.load_open_box(
            'pot', [0.16, 0.16, 0.08], [0.5, 0, 0.46],
            wall=0.01, color=[0.4, 0.4, 0.4, 1.0])
        # 番茄在锅内腔里（腔底 0.43 + 半径 0.02）
        p.resetBasePositionAndOrientation(
            self.objects['tomato'], [0.5, 0, 0.45], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 盖子压在锅沿上（锅沿高 0.50）
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, 0, 0.505], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        # 番茄离开锅内腔、且仍在桌面以上（不是掉到地上）
        out_of_pot = not self.is_inside('tomato', self.pot, tol=0.0)
        return bool(out_of_pot and self.get_object_position('tomato')[2] > 0.42)


@register_task('plateoven')
class PlateOven(BaseTask):
    """把盘子放进烤箱再把烤箱门关上

    旧版烤箱是实心方块、门是自由刚体摆在方块内部高度（静置自己掉 19 cm），
    「盘子在里面」只是看水平距离，盘子可以停在烤箱顶上也算数。
    新版烤箱是上方开口的壳（盘子从上面放进去），门是竖直轴铰链（初始开着，
    推回去关上门才算完成）。
    """
    door_close_threshold_deg = 60.0

    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('plate', 0.08, 0.01, [0, 0, 0],
                           color=[0.9, 0.9, 0.9, 1.0], mass=0.05, friction=0.6)

    def _set_initial_positions(self):
        # 烤箱：外廓 0.40 x 0.40 x 0.40，底面贴桌面 0.42，上方开口
        self.oven = self.load_open_box(
            'oven', [0.40, 0.40, 0.40], [0.5, 0.3, 0.62],
            wall=0.01, color=[0.5, 0.5, 0.5, 1.0])
        # 门贴烤箱正面（x=0.70）外 2 cm，初始开 95 度（绕 -y 竖边往外转）
        self.load_hinged_panel(
            'oven_door', [0.01, 0.19, 0.19], [0.73, 0.3, 0.62],
            pivot=[0.73, 0.11, 0.62], yaw_deg=-95.0,
            color=[0.4, 0.4, 0.4, 1.0], mass=0.2, hold_force=5.0,
            ignore_collision_with=['oven_panel2'])
        # 盘子先放在旁边的桌面上
        p.resetBasePositionAndOrientation(
            self.objects['plate'], [0.5, -0.3, 0.425], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        plate_in = self.is_inside('plate', self.oven, tol=0.02)
        door_closed = self.yaw_change_deg('oven_door') > self.door_close_threshold_deg
        return bool(plate_in and door_closed)
