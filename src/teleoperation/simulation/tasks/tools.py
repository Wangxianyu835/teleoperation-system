import numpy as np
import pybullet as p
from .base import BaseTask
from .registry import register_task
@register_task('rotatefaucet')
class RotateFaucet(BaseTask):
    """旋转水龙头手柄 90 度（手柄 = 绕基座竖直轴转的杠杆）

    旧版手柄是自由刚体、悬在基座上方，落地会被挤开；判据看绝对欧拉角，
    平放的箱子随便一转就能蒙对。新版用竖直轴铰链 + 相对偏航角。
    """
    yaw_threshold_deg = 80.0

    def _load_objects(self):
        self.load_fixed_table()
        # 水龙头基座（固定）
        self.load_cylinder('base', 0.03, 0.08, [0, 0, 0],
                           color=[0.5, 0.5, 0.5, 1.0], mass=0)

    def _set_initial_positions(self):
        # 基座底面贴桌面：0.42 + 半高 0.04
        p.resetBasePositionAndOrientation(
            self.objects['base'], [0.5, 0.3, 0.46], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 手柄贴在基座顶面（0.50），铰链轴过基座中心 -> 绕竖直轴转
        self.load_hinged_panel(
            'handle', [0.08, 0.015, 0.015], [0.56, 0.3, 0.515],
            pivot=[0.5, 0.3, 0.515], color=[0.7, 0.7, 0.7, 1.0],
            mass=0.05, hold_force=3.0, ignore_collision_with=['base'])

    def check_success(self) -> bool:
        return abs(self.yaw_change_deg('handle')) > self.yaw_threshold_deg


@register_task('rotatehearth')
class RotateHearth(BaseTask):
    """旋转炉灶旋钮 >25 度（旋钮 = 绕竖直轴转的杠杆）

    旧判据要求旋钮「按下」到 z<0.43 —— 那是往桌面里按 3 cm，物理上不可能，
    这个任务旧版永远做不成。新版只保留可观测的旋转。
    """
    yaw_threshold_deg = 25.0

    def _load_objects(self):
        self.load_fixed_table()
        # 灶台面板（固定）
        self.load_cylinder('knob_base', 0.04, 0.02, [0, 0, 0],
                           color=[0.25, 0.25, 0.25, 1.0], mass=0)

    def _set_initial_positions(self):
        # 灶台面板底面贴桌面 0.42
        p.resetBasePositionAndOrientation(
            self.objects['knob_base'], [0.5, 0.1, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 旋钮手柄贴在面板顶面（0.44），铰链轴过面板中心
        self.load_hinged_panel(
            'knob', [0.03, 0.008, 0.008], [0.53, 0.1, 0.448],
            pivot=[0.5, 0.1, 0.448], color=[0.2, 0.2, 0.2, 1.0],
            mass=0.02, hold_force=1.5, ignore_collision_with=['knob_base'])

    def check_success(self) -> bool:
        return abs(self.yaw_change_deg('knob')) > self.yaw_threshold_deg


@register_task('openmicrowave')
class OpenMicrowave(BaseTask):
    """把微波炉门拉开 >60 度

    旧版门是自由刚体，摆在机身内部高度，静置 1 s 自己掉 14 cm；而且 t=0 的
    |门x - 机身x| = 0.15 已 > 0.12 -> 白送成功。新版门是竖直轴铰链：t=0 是
    关门状态，转开 60 度才算成功；轴与重力平行，不推就不会动。
    """
    open_threshold_deg = 60.0

    def _load_objects(self):
        self.load_fixed_table()
        # 微波炉机身（固定）
        self.load_box('microwave_body', [0.15, 0.2, 0.15], [0, 0, 0],
                      color=[0.8, 0.8, 0.8, 1.0], mass=0)

    def _set_initial_positions(self):
        # 机身底面贴桌面：0.42 + 半高 0.15
        p.resetBasePositionAndOrientation(
            self.objects['microwave_body'], [0.5, 0.3, 0.57], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 门贴机身正面（x=0.65）外 2 cm，铰链在门的 -y 竖边
        self.load_hinged_panel(
            'door', [0.01, 0.2, 0.14], [0.68, 0.3, 0.57],
            pivot=[0.68, 0.1, 0.57], color=[0.5, 0.5, 0.5, 1.0],
            mass=0.2, hold_force=5.0, ignore_collision_with=['microwave_body'])

    def check_success(self) -> bool:
        # 往外开 = 绕 -y 竖边负向转；开过 60 度算成功
        return self.yaw_change_deg('door') < -self.open_threshold_deg


@register_task('closemicrowave')
class CloseMicrowave(BaseTask):
    """把开着的微波炉门推回关上（初始开 95 度）

    旧版门是自由刚体摆在 x=0.8，落地过程可能恰好落进 2 cm 判据里。
    新版门是竖直轴铰链、初始开 95 度，判据要求偏航变化 > 60 度（推回）。
    """
    close_threshold_deg = 60.0

    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('microwave_body', [0.15, 0.2, 0.15], [0, 0, 0],
                      color=[0.8, 0.8, 0.8, 1.0], mass=0)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['microwave_body'], [0.5, 0.3, 0.57], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 门初始开着：绕 -y 竖边往外转 95 度（门距机身 2 cm，不与机身互相挤住）
        self.load_hinged_panel(
            'door', [0.01, 0.2, 0.14], [0.68, 0.3, 0.57],
            pivot=[0.68, 0.1, 0.57], yaw_deg=-95.0,
            color=[0.5, 0.5, 0.5, 1.0], mass=0.2, hold_force=5.0,
            ignore_collision_with=['microwave_body'])

    def check_success(self) -> bool:
        # 从 -95 度往 0 度推回去：偏航变化 > 60 度
        return self.yaw_change_deg('door') > self.close_threshold_deg


@register_task('opendrawer')
class OpenDrawer(BaseTask):
    """把抽屉拉出 >12 cm

    旧版抽屉是自由刚体，整个埋在柜体里（x 0.40~0.68 与柜体 0.25~0.55 重叠），
    静置就被挤飞 0.28 m；而且 t=0 的 |抽屉x-柜体x| = 0.14 已 > 0.12 -> 白送成功。
    新版抽屉挂水平滑轨（只沿 +x 平移 0~0.20 m），t=0 是「关到底」的位置。
    """
    pull_threshold_m = 0.12

    def _load_objects(self):
        self.load_fixed_table()
        # 柜体（固定）
        self.load_box('cabinet', [0.15, 0.2, 0.25], [0, 0, 0],
                      color=[0.6, 0.3, 0.1, 1.0], mass=0)

    def _set_initial_positions(self):
        # 柜体立在桌面上：0.42 + 半高 0.25
        p.resetBasePositionAndOrientation(
            self.objects['cabinet'], [0.4, 0.3, 0.67], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 抽屉贴着柜体正面（0.55），可沿 +x 拉出 20 cm；轨道水平 -> 不会自己滑
        self.load_sliding_panel(
            'drawer', [0.14, 0.18, 0.1], [0.695, 0.3, 0.52],
            axis=(1, 0, 0), limits=(0.0, 0.2),
            color=[0.7, 0.4, 0.2, 1.0], mass=0.15,
            ignore_collision_with=['cabinet'])

    def check_success(self) -> bool:
        pulled = (self.get_object_position('drawer')[0]
                  - self.initial_position('drawer')[0])
        return float(pulled) > self.pull_threshold_m


@register_task('closedrawer')
class CloseDrawer(BaseTask):
    """把拉出的抽屉推回去 >10 cm（t=0 时拉出 15 cm）

    旧版抽屉是自由刚体埋在柜体里，静置被挤飞 0.28 m 且判据随时可能蒙对。
    新版挂水平滑轨（初始在 +15 cm 处，只能往回推 20 cm）。
    """
    push_threshold_m = 0.10

    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('cabinet', [0.15, 0.2, 0.25], [0, 0, 0],
                      color=[0.6, 0.3, 0.1, 1.0], mass=0)

    def _set_initial_positions(self):
        p.resetBasePositionAndOrientation(
            self.objects['cabinet'], [0.4, 0.3, 0.67], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 抽屉初始拉出 15 cm（0.695 -> 0.845），可沿 -x 推回 20 cm
        self.load_sliding_panel(
            'drawer', [0.14, 0.18, 0.1], [0.845, 0.3, 0.52],
            axis=(1, 0, 0), limits=(-0.2, 0.0),
            color=[0.7, 0.4, 0.2, 1.0], mass=0.15,
            ignore_collision_with=['cabinet'])

    def check_success(self) -> bool:
        pushed = (self.initial_position('drawer')[0]
                  - self.get_object_position('drawer')[0])
        return float(pushed) > self.push_threshold_m


@register_task('liftmug')
class LiftMug(BaseTask):
    """把杯盖提起 >5 cm

    旧判据是 |杯盖z - 杯身z| > 0.05，t=0 恰好 0.055 -> 白送成功；杯身中心还
    摆在 0.42（h=0.1，嵌进桌面 5 cm）。新版：杯身贴桌面、杯盖贴杯顶，
    判据改成「杯盖相对初始位置抬升 > 5 cm」。
    """
    lift_threshold_m = 0.05

    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('mug_body', 0.05, 0.1, [0, 0, 0],
                           color=[0.3, 0.6, 0.3, 1.0], mass=0.1, friction=0.6)
        self.load_cylinder('lid', 0.052, 0.01, [0, 0, 0],
                           color=[0.2, 0.5, 0.2, 1.0], mass=0.02, friction=0.5)

    def _set_initial_positions(self):
        # 杯身：底面贴桌面 0.42 -> 中心 0.47
        p.resetBasePositionAndOrientation(
            self.objects['mug_body'], [0.5, 0, 0.47], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 杯盖：贴在杯身顶面 0.52 上
        p.resetBasePositionAndOrientation(
            self.objects['lid'], [0.5, 0, 0.525], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        return self.rise('lid') > self.lift_threshold_m


@register_task('openlaptop')
class OpenLaptop(BaseTask):
    """把笔记本屏幕掀开（屏幕抬起 >3 cm 且转过 >15 度）

    旧版屏幕是自由刚体、悬在底座上方，静置就倒；判据看绝对欧拉角，屏幕
    随便翻个面都可能蒙对。新版屏幕绕后缘水平轴（合盖状态 t=0），松手会自己
    合上，所以成功只在「撑住」的那一刻成立。

    ★ 铰链实测：这条水平轴最多能转到 24~33 度（该 build 上再大就顶住，
      原因未查清，见 docs/SIM_EVALUATION.md）。判据按实测可达范围定。
    """
    tilt_threshold_deg = 15.0
    lift_threshold_m = 0.03

    def _load_objects(self):
        self.load_fixed_table()
        self.load_box('laptop_base', [0.15, 0.1, 0.01], [0, 0, 0],
                      color=[0.3, 0.3, 0.3, 1.0], mass=0.5, friction=0.8)

    def _set_initial_positions(self):
        # 底座底面贴桌面 0.42 -> 中心 0.43
        p.resetBasePositionAndOrientation(
            self.objects['laptop_base'], [0.5, 0, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 屏幕合在底座上方 2.5 cm（底座顶面 0.44），铰链轴沿 x（宽度方向）过后缘
        self.load_hinged_panel(
            'laptop_screen', [0.15, 0.1, 0.008], [0.5, 0.0, 0.465],
            pivot=[0.5, -0.1, 0.465], axis=(1, 0, 0),
            color=[0.1, 0.1, 0.2, 1.0], mass=0.3, hold_force=5.0,
            ignore_collision_with=['laptop_base'])

    def check_success(self) -> bool:
        opened = self.tilt_deg('laptop_screen') > self.tilt_threshold_deg
        lifted = self.rise('laptop_screen') > self.lift_threshold_m
        return bool(opened and lifted)


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
    """把水壶倾到杯子上方倒水（倾斜 >60 度且壶口对着杯子）

    旧判据只要「倾 >60 度且壶高于 0.5」，不提杯子在哪儿，而且壶中心摆在
    0.42（h=0.12，嵌进桌面 6 cm）。新版贴桌面摆放 + 要求水平位置对上杯子。
    """
    tilt_threshold_deg = 60.0

    def _load_objects(self):
        self.load_fixed_table()
        self.load_cylinder('kettle', 0.05, 0.12, [0, 0, 0],
                           color=[0.1, 0.5, 0.8, 1.0], mass=0.15, friction=0.6)
        self.load_cylinder('cup', 0.04, 0.08, [0, 0, 0],
                           color=[0.3, 0.7, 0.3, 1.0], mass=0.06, friction=0.6)

    def _set_initial_positions(self):
        # 壶：底面贴桌面 0.42 -> 中心 0.48；杯子：中心 0.46
        p.resetBasePositionAndOrientation(
            self.objects['kettle'], [0.5, -0.2, 0.48], [0, 0, 0, 1],
            physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.objects['cup'], [0.35, 0.2, 0.46], [0, 0, 0, 1],
            physicsClientId=self.client)

    def check_success(self) -> bool:
        tilted = self.tilt_deg('kettle') > self.tilt_threshold_deg
        over_cup = self.is_over('kettle', 'cup', tol=0.08)
        return bool(tilted and over_cup)


@register_task('breadtoaster')
class BreadToaster(BaseTask):
    """把面包放进烤面包机并压下压杆

    旧版：烤面包机是实心箱体、按钮埋在箱体里（中心 z=0.48，箱体 0.42~0.62），
    判据要求按钮 z<0.47 —— 既做不成，按钮还会被挤出箱体乱飞。
    新版：机身是上方开口的壳（面包从上面放进去），侧面压杆挂水平滑轨，
    判据 = 面包落在内腔里 且 压杆被推 >2 cm。
    """
    lever_threshold_m = 0.02

    def _load_objects(self):
        self.load_fixed_table()
        # 面包：放在旁边桌面上（0.42 + 半高 0.01）
        self.load_box('bread', [0.04, 0.04, 0.01], [0, 0, 0],
                      color=[0.9, 0.8, 0.5, 1.0], mass=0.02, friction=0.5)

    def _set_initial_positions(self):
        # 烤面包机：外廓 0.12 x 0.20 x 0.20，底面贴桌面 0.42，上方开口
        self.toaster = self.load_open_box(
            'toaster', [0.12, 0.20, 0.20], [0.5, 0.2, 0.52],
            wall=0.01, color=[0.7, 0.7, 0.7, 1.0])
        p.resetBasePositionAndOrientation(
            self.objects['bread'], [0.5, -0.2, 0.43], [0, 0, 0, 1],
            physicsClientId=self.client)
        # 压杆：机身 -x 侧面外 1 cm，沿 -x 推 3 cm
        self.load_sliding_panel(
            'toaster_lever', [0.01, 0.02, 0.02], [0.42, 0.2, 0.5],
            axis=(-1, 0, 0), limits=(0.0, 0.03),
            color=[1.0, 0.0, 0.0, 1.0], mass=0.01)

    def check_success(self) -> bool:
        bread_in = self.is_inside('bread', self.toaster)
        lever_pushed = self.displacement('toaster_lever') > self.lever_threshold_m
        return bool(bread_in and lever_pushed)
