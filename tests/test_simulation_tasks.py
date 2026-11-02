"""30 个内置任务的回归规格：判据不能白送、场景不能自己动、关节必须推得动。

这些断言对应 docs/SIM_EVALUATION.md 里的三个失败模式：
  1. t=0 就 check_success()=True      -> 任务白送成功（旧版 3/30）
  2. 不给动作也自己动 > 2 cm          -> 摆位穿模被弹飞（旧版 21/30）
  3. 门/抽屉焊死                      -> 任务永远做不成
"""

import unittest

import numpy as np
import pybullet as p
import pybullet_data

from teleoperation.simulation.randomization import DomainRandomizer
from teleoperation.simulation.tasks import get_task, list_tasks
from teleoperation.simulation.tasks import register_builtin_tasks


SETTLE_STEPS = 120          # 0.5 s @ 240 Hz
MAX_DRIFT_M = 0.02          # 允许的自发位移


class SimulationTaskSceneTests(unittest.TestCase):
    """不加载机器人：任务场景本身（摆位 + 判据）的规格。"""

    @classmethod
    def setUpClass(cls):
        register_builtin_tasks()

    def setUp(self):
        self.client = p.connect(p.DIRECT)
        self._reset_world()

    def tearDown(self):
        p.disconnect(physicsClientId=self.client)

    def _reset_world(self):
        """每个任务都要清场：否则上一个任务的桌子/家具还留在世界里。"""
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.loadURDF('plane.urdf', physicsClientId=self.client)

    def _make_task(self, name):
        self._reset_world()
        task = get_task(name)(self.client)
        task.reset()
        return task

    def test_no_task_succeeds_at_reset(self):
        """t=0 时任何任务的判据都必须是 False（否则一次 episode 白送成功）。"""
        offenders = [name for name in list_tasks()
                     if self._make_task(name).check_success()]
        self.assertEqual(offenders, [], f'这些任务在 t=0 就成功了: {offenders}')

    def test_no_task_moves_or_succeeds_without_action(self):
        """不给动作静置 0.5 s：物体不能自己动 > 2 cm、不能掉到桌面以下，
        判据也不能被重力凑成 True。"""
        drift_offenders = []
        success_offenders = []
        for name in list_tasks():
            task = self._make_task(name)
            start = {obj: task.get_object_position(obj) for obj in task.objects
                     if obj != 'table'}
            for _ in range(SETTLE_STEPS):
                p.stepSimulation(physicsClientId=self.client)
            for obj, pos0 in start.items():
                pos1 = task.get_object_position(obj)
                drift = float(np.linalg.norm(pos1 - pos0))
                if drift > MAX_DRIFT_M:
                    drift_offenders.append(f'{name}.{obj}: {drift * 1000:.0f} mm')
                if pos1[2] < task.table_top_z - 0.05:
                    drift_offenders.append(f'{name}.{obj}: 掉到桌面以下 z={pos1[2]:.3f}')
            if task.check_success():
                success_offenders.append(name)
        self.assertEqual(drift_offenders, [],
                         '这些物体自己动了: ' + '; '.join(drift_offenders))
        self.assertEqual(success_offenders, [],
                         f'静置后自己完成的任务: {success_offenders}')

    def test_objects_rest_on_table_not_inside_it(self):
        """吸附之后物体底面必须贴在支撑面上（旧版嵌进桌面 2~10 cm 被弹飞）。"""
        task = self._make_task('liftmug')
        mug_z = float(task.get_object_position('mug_body')[2])
        self.assertAlmostEqual(mug_z, task.table_top_z + 0.05, places=3)  # 半高 0.05
        lid_z = float(task.get_object_position('lid')[2])
        self.assertAlmostEqual(lid_z, task.table_top_z + 0.10 + 0.005, places=3)

    def test_relative_criteria_need_actual_motion(self):
        """把杯盖挪起来 6 cm 才算成功：判据看的是「相对初始位置的抬升」。"""
        task = self._make_task('liftmug')
        lid, _ = task.get_object_pose('lid')
        self.assertFalse(task.check_success())
        p.resetBasePositionAndOrientation(
            task.objects['lid'], [float(lid[0]), float(lid[1]), float(lid[2] + 0.04)],
            [0, 0, 0, 1], physicsClientId=self.client)
        self.assertFalse(task.check_success(), '抬 4 cm 不该算成功（阈值 5 cm）')
        p.resetBasePositionAndOrientation(
            task.objects['lid'], [float(lid[0]), float(lid[1]), float(lid[2] + 0.06)],
            [0, 0, 0, 1], physicsClientId=self.client)
        self.assertTrue(task.check_success())
        self.assertGreater(task.rise('lid'), 0.05)

    def test_drawer_slides_and_satisfies_criterion(self):
        """抽屉挂在水平滑轨上：驱动关节能拉开 > 12 cm 并满足判据。"""
        task = self._make_task('opendrawer')
        body, link = task.objects['drawer'], task._links['drawer']
        p.setJointMotorControl2(body, link, p.POSITION_CONTROL,
                                targetPosition=0.2, force=200.0,
                                physicsClientId=self.client)
        for _ in range(180):
            p.stepSimulation(physicsClientId=self.client)
        self.assertGreater(task.get_object_position('drawer')[0]
                           - task.initial_position('drawer')[0], 0.12)
        self.assertTrue(task.check_success())

    def test_door_swings_about_its_hinge(self):
        """微波炉门绕竖直轴铰链转开：面板中心位移 > 10 cm 且判据成立。"""
        task = self._make_task('openmicrowave')
        body, link = task.objects['door'], task._links['door']
        p.setJointMotorControl2(body, link, p.POSITION_CONTROL,
                                targetPosition=-1.57, force=200.0,
                                physicsClientId=self.client)
        for _ in range(180):
            p.stepSimulation(physicsClientId=self.client)
        self.assertGreater(float(np.linalg.norm(
            task.get_object_position('door') - task.initial_position('door'))), 0.10)
        self.assertTrue(task.check_success())

    def test_randomizer_skips_jointed_panels(self):
        """域随机化不能搬动门/抽屉（搬了就把铰链/滑轨撕开）。"""
        task = self._make_task('opendrawer')
        before = task.get_object_position('drawer').copy()
        DomainRandomizer().randomize(task.objects, self.client,
                                     skip_names=task.jointed_objects)
        after = task.get_object_position('drawer')
        np.testing.assert_allclose(after, before, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
