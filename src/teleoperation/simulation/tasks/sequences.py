import numpy as np
import pybullet as p
from .base import BaseTask
from .registry import register_task
@register_task('pottomatoplate')
class PotTomatoPlate(BaseTask):
    """打开锅盖把番茄放到盘子上

    旧版锅是实心圆柱、番茄摆在锅心，「在锅里」物理上不可能（静置被挤出）。
    新版锅是上方开口的壳（番茄真在锅内、盖子压在锅沿），判据 = 番茄落在
    盘子上方 且 已经不在锅内。
    """
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('lid', 0.082, 0.01, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0.03, friction=0.5)
        self.load_sphere('tomato', 0.02, [0, 0, 0],
                         color=[1.0, 0, 0, 1.0], mass=0.02, friction=0.5)
        self.load_cylinder('plate', 0.08, 0.01, [0, 0, 0],
                           color=[0.9, 0.9, 0.9, 1.0], mass=0.05, friction=0.6)

    def _set_initial_positions(self):
        # 锅：外廓 0.16 x 0.16 x 0.08，底面贴桌面 0.42，上方开口
        self.pot = self.load_open_box(
            'pot', [0.16, 0.16, 0.08], [0.5, 0, 0.46],
            wall=0.01, color=[0.4, 0.4, 0.4, 1.0])
        p.resetBasePositionAndOrientation(
            self.objects['tomato'], [0.5, 0, 0.45], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, 0, 0.505], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['plate'], [0.3, 0.3, 0.425], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        on_plate = self._is_above('tomato', 'plate', threshold=0.04)
        out_of_pot = not self.is_inside('tomato', self.pot, tol=0.0)
        return bool(on_plate and out_of_pot)


@register_task('penbrushpot')
class PenBrushPot(BaseTask):
    """拿起笔插进笔筒（笔筒是上方开口的壳，判据 = 笔真的在筒里）

    旧版笔筒是实心圆柱，「插进去」只能用「在它上方」近似，笔停在筒顶上也算成功。
    """
    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('pen', 0.01, 0.12, [0, 0, 0],
                           color=[0.1, 0.1, 0.8, 1.0], mass=0.01, friction=0.5)

    def _set_initial_positions(self):
        # 笔：底面贴桌面 0.42 -> 中心 0.48
        p.resetBasePositionAndOrientation(
            self.objects['pen'], [0.5, -0.3, 0.48], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 笔筒：外廓 0.08 x 0.08 x 0.10，底面贴桌面，上方开口（内腔 0.06 见方）
        self.holder = self.load_open_box(
            'pot_holder', [0.08, 0.08, 0.10], [0.3, 0.3, 0.47],
            wall=0.01, color=[0.3, 0.3, 0.3, 1.0])

    def check_success(self) -> bool:
        return self.is_inside('pen', self.holder, tol=0.01)


@register_task('drawerbook')
class DrawerBook(BaseTask):
    """把书放进（拉出来的）抽屉里

    旧版抽屉是自由刚体、埋在柜体里（静置被挤飞 0.28 m），「书在抽屉上方」只看
    水平距离，书停在抽屉顶上也算成功。
    新版抽屉是上方开口的托盘（跟着柜子固定在桌面），判据 = 书真的落进抽屉内腔。
    注：原先的「再把抽屉关上」子目标去掉了 —— 抽屉要能被关上就得是滑轨上的
        刚体，那它就不可能同时具备内腔（PyBullet 的 box 没有内部空腔）。
    """
    def _load_objects(self):
        self.load_fixed_table()
        # 柜体（固定）
        self.load_box('cabinet', [0.15, 0.2, 0.25], [0, 0, 0],
                      color=[0.6, 0.3, 0.1, 1.0], mass=0)
        # 书：先放在旁边桌面上
        self.load_box('book', [0.06, 0.09, 0.015], [0, 0, 0],
                      color=[0.2, 0.2, 0.6, 1.0], mass=0.15, friction=0.6)

    def _set_initial_positions(self):
        # 柜体立在桌面上：0.42 + 半高 0.25
        p.resetBasePositionAndOrientation(
            self.objects['cabinet'], [0.4, 0.3, 0.67], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 抽屉：外廓 0.28 x 0.36 x 0.20，贴着柜体正面摆在地上（上方开口）
        self.drawer = self.load_open_box(
            'drawer', [0.28, 0.36, 0.20], [0.695, 0.3, 0.52],
            wall=0.01, color=[0.7, 0.4, 0.2, 1.0])
        p.resetBasePositionAndOrientation(
            self.objects['book'], [0.5, -0.3, 0.435], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self.is_inside('book', self.drawer, tol=0.01)


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
