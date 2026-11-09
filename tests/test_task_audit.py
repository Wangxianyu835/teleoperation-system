"""``devtools/audit_tasks.py`` 的口径保护：ORACLE 表要覆盖 30 个任务，检查逻辑要真能发现问题。

为什么需要这些用例
------------------
审计工具本身就是一把尺子：它说 OK，我们才敢把任务拿去给人做成功率。所以
最该被测试的不是任务，而是**尺子**：

1. 30 个任务名必须都有 ORACLE 条目 —— 队友改判据时这里先炸，而不是等到现场
   跑出 100% 成功率才发现；
2. ``pushcube`` 必须被判 OK（稳定 / 无假成功 / 判据可触发）；
3. 已知在 reset 时就成立的 ``liftmug`` 必须被判 FALSE_POSITIVE —— 这条保护的是
   **检测逻辑**：工具把坏的看成好的，比任务本身坏更危险。

运行::

    E:\\python3.11.7\\python.exe -m unittest tests.test_task_audit -v
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

AUDIT_PATH = Path(__file__).resolve().parents[1] / "devtools" / "audit_tasks.py"

#: op 名 -> 这条 op 应有的字段数（见 ORACLE 表上面的注释）
OP_ARITY = {"copy_pose": 3, "align": 3, "above": 4, "set_pos": 3,
            "set_z": 3, "raise": 3, "move_by": 3, "rotate": 3, "upright": 3}


def load_audit():
    """按路径加载 ``devtools/audit_tasks.py``（它不在安装包里，不靠 sys.path）。"""
    spec = importlib.util.spec_from_file_location("audit_tasks_under_test", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()


def registered_tasks():
    from teleoperation.simulation.tasks import list_tasks, register_builtin_tasks
    register_builtin_tasks()
    return list_tasks()


class AuditTableTests(unittest.TestCase):
    """ORACLE 表 = 30 个任务，且每条 op 的形状都对。"""

    def test_oracle_covers_every_registered_task(self):
        names = set(registered_tasks())
        self.assertEqual(30, len(names), "the benchmark is supposed to have 30 tasks")
        self.assertEqual(names, set(AUDIT.ORACLE),
                         "ORACLE is out of sync with the registered tasks")

    def test_oracle_ops_are_well_formed(self):
        for task, ops in AUDIT.ORACLE.items():
            self.assertTrue(ops, f"{task}: empty oracle")
            for op in ops:
                self.assertIn(op[0], OP_ARITY, f"{task}: unknown op {op!r}")
                self.assertEqual(OP_ARITY[op[0]], len(op),
                                 f"{task}: {op!r} has the wrong arity")
                self.assertIsInstance(op[1], str, f"{task}: object name not a str")

    def test_robots_are_the_loadable_ones(self):
        from teleoperation.simulation.arm_profiles import loadable_robots
        self.assertEqual(set(AUDIT.ROBOTS) - set(loadable_robots()), set(),
                         "audit asks for a robot the realtime stack cannot load")


class BestShiftTests(unittest.TestCase):
    """``best_scene_shift`` 是纯 numpy：平移搜索不能让最远距离变差。"""

    def test_shift_never_makes_the_farthest_worse(self):
        shoulder = np.array([0.0, 0.2, 1.4])
        cloud = [[0.5, 0.0, 0.45], [0.3, 0.3, 0.43], [0.6, 0.4, 0.5]]
        before = max(float(np.linalg.norm(np.asarray(point) - shoulder))
                     for point in cloud)
        best = AUDIT.best_scene_shift(shoulder, cloud, 0.58)
        self.assertLessEqual(best["farthest_m"], round(before, 4) + 1e-6)
        self.assertIn("inside", best)
        self.assertGreaterEqual(best["inside"], 0)

    def test_empty_cloud_and_missing_shoulder_are_handled(self):
        self.assertIsNone(AUDIT.best_scene_shift([0.0, 0.0, 0.0], [], 0.5)["farthest_m"])
        self.assertIsNone(AUDIT.best_scene_shift(None, [[0.5, 0.0, 0.4]], 0.5)["farthest_m"])


class TaskChecksTests(unittest.TestCase):
    """真连一个 DIRECT client 跑两个任务（--steps 120，秒级）。"""

    @classmethod
    def setUpClass(cls):
        import pybullet as p
        from teleoperation.simulation.tasks import register_builtin_tasks
        register_builtin_tasks()
        cls.client = p.connect(p.DIRECT)

    @classmethod
    def tearDownClass(cls):
        import pybullet as p
        if cls.client >= 0:
            p.disconnect(cls.client)

    def test_pushcube_passes_all_four_checks(self):
        row = AUDIT.check_task(self.client, "pushcube", 120)
        self.assertEqual([], row["flags"], f"pushcube is supposed to be evaluable: {row}")
        self.assertEqual("OK", row["verdict"])
        self.assertFalse(row["succ_at_reset"])
        self.assertTrue(row["succ_oracle"])

    def test_false_positive_is_detected_not_masked(self):
        row = AUDIT.check_task(self.client, "liftmug", 120)
        self.assertTrue(row["succ_at_reset"],
                        "liftmug is known to satisfy its criterion at reset")
        self.assertIn("FALSE_POSITIVE", row["flags"])
        self.assertEqual("FALSE_POSITIVE", row["verdict"])


if __name__ == "__main__":
    unittest.main()
