"""批次 5（无人脚本档）的机制用例：可达半径 / 场景重摆 / 脚本接线。

这三条钉的是「以后谁改这几个数会立刻红」的地方，不需要摄像头、不需要人：
* ``sample_reach`` 必须比「连杆和」小（以前就是拿连杆和当半径，物体全被摆到够不着的地方）；
* ``_layout_for_arm`` 必须把桌面放到肩下、物体放到肩前、桌子近沿留在 0.24 m 外；
* ``ScriptedPolicy`` 必须真的把 30 个任务里的键接上（没有脚本的任务走 ``default_script``）。
"""

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVTOOLS = ROOT / "devtools"
for extra in (ROOT / "src", DEVTOOLS):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import pybullet as p  # noqa: E402

import audit_tasks as AUDIT  # noqa: E402
import task_eval as EVAL  # noqa: E402
import task_scripted as SCRIPT  # noqa: E402


class ReachMeasurementTests(unittest.TestCase):
    """可达半径只能用 FK 采样量：连杆和会把整幕摆到够不着的地方。"""

    def test_sampled_radius_is_smaller_than_link_sum(self):
        arm = EVAL.measure_arm("h1_2")
        link_sum = arm["reach_link_sum_m"]
        sampled = max(arm["reach_sampled_m"].values())
        self.assertAlmostEqual(sampled, arm["reach_m"], places=4)
        self.assertLess(sampled, link_sum,
                        "采样半径必须小于『肩 link 原点→肘 + 肘→腕』")
        self.assertGreater(sampled, 0.30, "h1_2 实测 0.4084 m")
        self.assertLess(sampled, 0.60)

    def test_cache_returns_a_copy(self):
        first = EVAL.measure_arm("h1_2")
        first["reach_m"] = -1.0
        second = EVAL.measure_arm("h1_2")
        self.assertGreater(second["reach_m"], 0.30,
                           "缓存被调用方改坏了 —— 返回的必须是副本")


class LayoutForArmTests(unittest.TestCase):
    """脚本档重摆：桌面比肩低、物体在肩前、桌子近沿不贴到机器人身上。"""

    @classmethod
    def setUpClass(cls):
        from teleoperation.simulation.environment import SimulationEnv
        from teleoperation.simulation.tasks import register_builtin_tasks

        register_builtin_tasks()
        cls.arm = EVAL.measure_arm("h1_2")
        cls.env = SimulationEnv(robot_type="h1_2", task_name="pushcube", render=False,
                                record=False,
                                data_dir=str(ROOT / "outputs" / "tmp_task_eval"))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _scene(self):
        self.env.reset(randomize=False)
        return self.env.task, EVAL.TaskScene(
            self.env.client, self.env.task, robot="h1_2",
            shoulder=self.arm["shoulder_m"], reach=self.arm["reach_m"],
            name="pushcube", robot_id=self.env.robot_id)


    def test_table_sits_below_the_shoulder_and_objects_in_front(self):
        task, scene = self._scene()
        scene._layout_for_arm(rebuild_bench=False)
        shoulder = np.asarray(self.arm["shoulder_m"], dtype=float)
        top = float(p.getAABB(task.objects["table"],
                              physicsClientId=self.env.client)[1][2])
        self.assertAlmostEqual(top, shoulder[2] - 0.22, delta=0.02)
        points = np.stack([EVAL.body_pose(self.env.client, task.objects[key])[0]
                           for key in ("cube", "target")])
        self.assertAlmostEqual(float(points[:, 0].mean()), shoulder[0] + 0.26,
                               delta=0.03)
        self.assertLessEqual(float(np.abs(points[:, 1] - shoulder[1]).max()), 0.15)

        low = float(p.getAABB(task.objects["table"],
                              physicsClientId=self.env.client)[0][0])
        self.assertGreaterEqual(low, 0.23, "桌子近沿不能伸进机器人身体")
        self.assertTrue(scene.report["notes"], "重摆要留一条 note 进报告")

    def test_relative_spread_is_compressed(self):
        task, scene = self._scene()
        before = np.linalg.norm(
            EVAL.body_pose(self.env.client, task.objects["cube"])[0][:2]
            - EVAL.body_pose(self.env.client, task.objects["target"])[0][:2])
        scene._layout_for_arm(rebuild_bench=False)
        after = np.linalg.norm(
            EVAL.body_pose(self.env.client, task.objects["cube"])[0][:2]
            - EVAL.body_pose(self.env.client, task.objects["target"])[0][:2])
        self.assertLess(after, before, "相对布置必须被压缩（0.41 m 的臂一次搬不完 0.54 m）")
        self.assertLessEqual(after, 0.25)



class ScriptTableTests(unittest.TestCase):
    """30 个任务名都要么有脚本、要么明确落到 ``default_script``（报告里能看出来）。"""

    def test_every_task_has_a_builder_or_falls_back(self):
        from teleoperation.simulation.tasks import list_tasks, register_builtin_tasks

        register_builtin_tasks()
        names = list_tasks()
        self.assertEqual(30, len(names), names)
        covered = sorted(set(names) & set(SCRIPT.SCRIPT_BUILDERS))
        missing = sorted(set(names) - set(SCRIPT.SCRIPT_BUILDERS))
        self.assertGreaterEqual(len(covered), 20, "至少 2/3 的任务要有真脚本")
        for name in covered:
            self.assertTrue(callable(SCRIPT.SCRIPT_BUILDERS[name]))
        # 没写脚本的任务必须能落到 default_script（有 fallback，不许 KeyError）
        self.assertTrue(callable(SCRIPT.default_script))
        self.assertIn("pushcube", covered)
        self.assertIn("pottomatoplate", covered)
        self.assertTrue(missing, "剩下的任务名要如实留在 missing 里，不装成已覆盖")


if __name__ == "__main__":
    unittest.main()
