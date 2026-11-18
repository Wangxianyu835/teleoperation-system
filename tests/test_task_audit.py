"""``devtools/audit_tasks.py`` + ``devtools/task_eval.py`` 的口径保护。

为什么需要这些用例
------------------
审计工具本身就是一把尺子：它说 OK，我们才敢把任务拿去给人做成功率。所以
最该被测试的不是任务，而是**尺子**：

1. 30 个任务名必须都有 ORACLE 条目 —— 队友改判据时这里先炸，而不是等到现场
   跑出 100% 成功率才发现；
2. ``pushcube`` 必须被判 OK（稳定 / 无假成功 / 判据可触发）；
3. 已知在 reset 时就成立的 ``liftmug`` 必须被判 FALSE_POSITIVE —— 这条保护的是
   **检测逻辑**：工具把坏的看成好的，比任务本身坏更危险；
4. 批次 2 的场景层：``solve_shift`` 只许往里拉、不许往外推；套上场景层后
   ``pushcube`` 必须 稳定 + 无假成功 + 全在可达圈内；10 个代表任务**一个都不许**
   落进排除名单（否则批次 3 的试点就没任务了）。

运行::

    E:\\python3.11.7\\python.exe -m unittest tests.test_task_audit -v
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

AUDIT_PATH = Path(__file__).resolve().parents[1] / "devtools" / "audit_tasks.py"
EVAL_PATH = Path(__file__).resolve().parents[1] / "devtools" / "task_eval.py"

#: op 名 -> 这条 op 应有的字段数（见 ORACLE 表上面的注释）
OP_ARITY = {"copy_pose": 3, "align": 3, "above": 4, "set_pos": 3,
            "set_z": 3, "raise": 3, "move_by": 3, "rotate": 3, "upright": 3}


def load_audit():
    """按路径加载 ``devtools/audit_tasks.py``（它不在安装包里，不靠 sys.path）。"""
    spec = importlib.util.spec_from_file_location("audit_tasks_under_test", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval():
    """按路径加载 ``devtools/task_eval.py``（它 import audit_tasks，自己也加 devtools 到 sys.path）。"""
    spec = importlib.util.spec_from_file_location("task_eval_under_test", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()
EVAL = load_eval()


class CameraHandPolicyTests(unittest.TestCase):
    """摄像头任务策略必须把 18D 手角度落到原生手 joint，而不是只驱动手臂。"""

    def test_native_hand_targets_are_written_by_joint_name(self):
        class _Angles:
            values = np.zeros(18, dtype=np.float64)

            def native_mapping_dofs(self):
                return self.values[1:]

        class _Retargeter:
            def retarget(self, _window):
                return SimpleNamespace(right_angles=_Angles(), right_valid=True)

        class _Adapter:
            def map(self, _dofs, pinch=None):
                return {"R_index_proximal_joint": 0.7,
                        "R_middle_proximal_joint": -0.4}, 2

        policy = object.__new__(EVAL.CameraPolicy)
        policy.side = "right"
        policy.hand_enabled = True
        policy.hand_retargeter = _Retargeter()
        policy.hand_adapter = _Adapter()
        policy.hand_backend_name = "geometric"
        policy.hand_limit_mode = "clamp"
        policy.action = np.zeros(4, dtype=np.float64)
        policy.names = {"R_index_proximal_joint": 1,
                        "R_middle_proximal_joint": 3}
        policy.hand_valid_frames = 0
        policy.hand_invalid_frames = 0
        policy.hand_clipped = 0
        policy.last_hand = {}
        policy._pinch_signal = lambda _points: (None, 0.0, None)

        policy._update_hand(object())
        self.assertAlmostEqual(policy.action[1], 0.7)
        self.assertAlmostEqual(policy.action[3], -0.4)
        self.assertEqual(policy.hand_valid_frames, 1)
        self.assertEqual(policy.hand_clipped, 2)
        self.assertTrue(policy.last_hand["valid"])

    def test_cli_defaults_enable_camera_hands_with_faithful_limits(self):
        args = EVAL.build_parser().parse_args([])
        self.assertEqual(args.camera_hand, "on")
        self.assertEqual(args.hand_backend, "auto")
        self.assertEqual(args.hand_limit_mode, "clamp")
        self.assertEqual(args.camera_table_clearance_mm, 60.0)
        self.assertEqual(args.camera_arm_force, 200.0)
        self.assertEqual(args.camera_hand_force, 200.0)
        self.assertEqual(args.camera_ik_mode, "pose")
        self.assertEqual(args.camera_ik_solver, "paper")
        self.assertEqual(args.camera_wrist_pose, "on")
        self.assertEqual(args.camera_target_speed_mm_s, 600.0)
        self.assertFalse(args.camera_table_guard)

    def test_table_guard_only_lifts_targets_over_the_table_footprint(self):
        bounds = (np.array([0.2, -0.4, 0.0]), np.array([1.2, 0.4, 0.5]))
        guarded, active = EVAL.guard_table_target(
            [0.6, 0.0, 0.1], bounds, clearance_m=0.06)
        self.assertTrue(active)
        np.testing.assert_allclose(guarded, [0.6, 0.0, 0.56])
        # In front of the table, lowering the hand is legal.
        unchanged, active = EVAL.guard_table_target(
            [0.1, 0.0, 0.1], bounds, clearance_m=0.06)
        self.assertFalse(active)
        np.testing.assert_allclose(unchanged, [0.1, 0.0, 0.1])

    def test_a_mediapipe_pinch_drives_native_thumb_and_finger_closure(self):
        from teleoperation.robots.native_hand import map_dofs, pinch_signal

        points = np.zeros((21, 3), dtype=np.float64)
        points[0] = [0.0, 0.0, 0.0]
        points[5] = [0.0, 0.08, 0.0]
        points[17] = [0.0, -0.08, 0.0]
        points[4] = [0.01, 0.04, 0.0]
        points[8] = [0.01, 0.04, 0.0]
        points[9] = [0.0, 0.06, 0.0]
        points[12] = [0.0, -0.10, 0.0]
        points[13] = [0.0, -0.02, 0.0]
        points[16] = [0.0, -0.18, 0.0]
        points[20] = [0.0, -0.24, 0.0]
        finger, strength, ratio = pinch_signal(points)
        self.assertEqual(finger, "index")
        self.assertEqual(ratio, 0.0)
        self.assertEqual(strength, 1.0)
        mapping = [
            (2, "R_index_proximal_joint", 1.0, 0.0, 1.5),
            (16, "R_thumb_intermediate_joint", 1.0, 0.0, 1.2),
            (5, "R_middle_proximal_joint", 1.0, 0.0, 1.5),
        ]
        joints, _clipped = map_dofs(
            np.zeros(17), mapping, pinch=(finger, strength, ratio),
            robot_type="h1_2", side="right")
        self.assertEqual(joints["R_index_proximal_joint"], 1.5)
        self.assertEqual(joints["R_thumb_intermediate_joint"], 0.4)
        self.assertEqual(joints["R_middle_proximal_joint"], 0.0)

    def test_a_fist_triggers_all_native_fingers_even_without_thumb_tip_contact(self):
        from teleoperation.robots.native_hand import map_dofs, pinch_signal

        points = np.zeros((21, 3), dtype=np.float64)
        points[5] = [0.0, 0.08, 0.0]
        points[17] = [0.0, -0.08, 0.0]
        points[4] = [0.02, 0.02, 0.0]
        for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
            points[mcp] = [0.0, 0.04 if mcp < 13 else -0.04, 0.0]
            points[tip] = [0.01, 0.02 if tip < 13 else -0.02, 0.0]
        finger, strength, ratio = pinch_signal(points)
        self.assertEqual(finger, "fist")
        self.assertGreater(strength, 0.5)
        self.assertGreater(ratio, 0.0)

        mapping = [
            (2, "R_index_proximal_joint", 1.0, 0.0, 1.5),
            (5, "R_middle_proximal_joint", 1.0, 0.0, 1.5),
            (8, "R_ring_proximal_joint", 1.0, 0.0, 1.5),
            (11, "R_pinky_proximal_joint", 1.0, 0.0, 1.5),
            (16, "R_thumb_intermediate_joint", 1.0, 0.0, 1.2),
        ]
        joints, _ = map_dofs(
            np.zeros(17), mapping, pinch=(finger, strength, ratio),
            robot_type="h1_2", side="right")
        for name in ("R_index_proximal_joint", "R_middle_proximal_joint",
                     "R_ring_proximal_joint", "R_pinky_proximal_joint"):
            self.assertGreater(joints[name], 0.5)

    def test_native_pinch_uses_each_robot_hand_axis_not_farthest_limit(self):
        from teleoperation.robots.native_hand import map_dofs

        gr1 = [
            (2, "R_index_proximal_joint", -1.0, -1.57, 0.0),
            (3, "R_index_intermediate_joint", -1.0, -1.74, 0.0),
            (14, "R_thumb_proximal_yaw_joint", -1.0, -1.74, 0.0),
            (15, "R_thumb_proximal_pitch_joint", 1.0, 0.0, 1.22),
            (17, "R_thumb_distal_joint", 1.0, 0.0, 1.04),
        ]
        joints, _ = map_dofs(
            np.zeros(17), gr1, pinch=("index", 1.0, 0.0),
            robot_type="gr1_t2", side="right")
        self.assertEqual(joints["R_index_proximal_joint"], -0.785)
        self.assertEqual(joints["R_index_intermediate_joint"], -1.74)
        self.assertEqual(joints["R_thumb_proximal_yaw_joint"], -0.87)
        self.assertEqual(joints["R_thumb_distal_joint"], 1.04)

        g1 = [
            (2, "right_hand_index_0_joint", 1.0, 0.0, 1.5708),
            (3, "right_hand_index_1_joint", 1.0, 0.0, 1.7453),
            (14, "right_hand_thumb_0_joint", 1.0, -1.0472, 1.0472),
            (15, "right_hand_thumb_1_joint", 1.0, -1.0472, 0.7243),
            (16, "right_hand_thumb_2_joint", 1.0, -1.7453, 0.0),
        ]
        joints, _ = map_dofs(
            np.zeros(17), g1, pinch=("index", 1.0, 0.0),
            robot_type="g1", side="right")
        self.assertEqual(joints["right_hand_index_0_joint"], 1.5708)
        self.assertEqual(joints["right_hand_index_1_joint"], 0.0)
        self.assertEqual(joints["right_hand_thumb_0_joint"], 0.0)
        self.assertEqual(joints["right_hand_thumb_1_joint"], -1.0472)


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

class SceneShiftTests(unittest.TestCase):
    """``solve_shift`` 是纯 numpy：拉了之后必须进圈，已经进圈的不许乱动。"""

    def test_far_cloud_is_pulled_inside_the_reach_sphere(self):
        shoulder = np.array([0.0, 0.0, 1.43])
        cloud = [[0.5, -0.3, 0.45], [0.3, 0.3, 0.45]]
        info = EVAL.solve_shift(shoulder, cloud, 0.55, table_min_x=0.0)
        moved = np.asarray(cloud, float) + np.array([info["dx_obj"], 0.0,
                                                     info["dz_obj"]])
        farthest = float(np.linalg.norm(moved - shoulder, axis=1).max())
        self.assertLessEqual(farthest, 0.55 + 1e-6, info)
        self.assertEqual(2, info["inside"], info)
        self.assertGreater(info["dz_obj"], 0.0, "只抬不降（判据里有绝对 z）")

    def test_cloud_already_inside_is_not_moved(self):
        info = EVAL.solve_shift(np.array([0.0, 0.0, 1.0]), [[0.3, 0.0, 1.0]],
                                0.55, table_min_x=0.0)
        self.assertEqual(0.0, info["dx_obj"])
        self.assertEqual(0.0, info["dz_obj"])
        self.assertTrue(info["feasible"])
        self.assertAlmostEqual(0.24, info["dx_table"], places=6,
                               msg="桌子仍要让开机器人本体")

    def test_no_shoulder_means_no_shift(self):
        info = EVAL.solve_shift(None, [[0.5, 0.0, 0.4]], 0.5)
        self.assertEqual(0.0, info["dx_obj"])
        self.assertFalse(info["feasible"])


class SceneRegistryTests(unittest.TestCase):
    """排除名单 / 场景 op 必须对得上真实任务；10 个代表任务不许被排除。"""

    def test_every_registry_key_is_a_registered_task(self):
        names = set(registered_tasks())
        for registry in (EVAL.CRITERIA_INVALID, EVAL.CRITERIA_ABSOLUTE_Z,
                         EVAL.SCENE_UNSUPPORTED):
            self.assertEqual(set(), set(registry) - names)
        self.assertEqual(set(), set(EVAL.SCENE_OPS) - names)

    def test_representative_tasks_are_all_still_scoreable(self):
        excluded = (set(EVAL.CRITERIA_INVALID) | set(EVAL.CRITERIA_ABSOLUTE_Z)
                    | set(EVAL.SCENE_UNSUPPORTED))
        self.assertEqual(set(), set(EVAL.DEFAULT_TASKS) & excluded,
                         "10 个代表任务里出现了被排除的任务，批次 3 的试点会没任务")

    def test_scene_ops_are_well_formed(self):
        arity = {"resize": 3, "place": 3, "above": 4, "align": 3}
        for task, ops in EVAL.SCENE_OPS.items():
            self.assertTrue(ops, task)
            for op in ops:
                self.assertIn(op[0], arity, task)
                self.assertEqual(arity[op[0]], len(op), "%s: %r" % (task, op))

    def test_shift_oracle_only_moves_absolute_z(self):
        ops = EVAL.shift_oracle([("set_z", "cube", 0.66),
                                 ("above", "cube", "plate", 0.06),
                                 ("set_pos", "ball", (0.5, 0.0, 0.65))], 0.75)
        self.assertAlmostEqual(1.41, ops[0][2])
        self.assertEqual(("above", "cube", "plate", 0.06), ops[1])
        self.assertAlmostEqual(1.40, ops[2][2][2])


class SceneLayerTests(unittest.TestCase):
    """真连一个 DIRECT client：过完场景层，任务必须 稳定 / 无假成功 / 全在圈内。"""

    @classmethod
    def setUpClass(cls):
        import pybullet as p
        cls.client = p.connect(p.DIRECT)
        cls.arm = EVAL.measure_arm("h1_2")

    @classmethod
    def tearDownClass(cls):
        import pybullet as p
        if cls.client >= 0:
            p.disconnect(cls.client)

    def _prepare(self, name, steps=240):
        from teleoperation.simulation.tasks import get_task
        AUDIT.fresh(self.client)
        task = get_task(name)(self.client)
        task.reset()
        scene = EVAL.TaskScene(self.client, task, robot="h1_2",
                               shoulder=self.arm["shoulder_m"],
                               reach=self.arm["reach_m"], name=name)
        return task, scene.prepare(steps)

    def test_pushcube_becomes_stable_and_reachable(self):
        task, report = self._prepare("pushcube")
        try:
            self.assertTrue(report["settle"]["stable"], report["settle"])
            self.assertFalse(report["criteria_at_reset"])
            self.assertEqual(report["in_reach"]["total"], report["in_reach"]["inside"])
        finally:
            task._clear_objects()

    def test_objects_are_lifted_onto_the_bench(self):
        task, report = self._prepare("pickcube")
        try:
            self.assertIn("cube", report["rest_on_table"],
                          "baseline 把方块中心摆在台面高度，必须抬上来")
            self.assertGreater(report["bench_top_m"], 0.9)
        finally:
            task._clear_objects()

    def test_trial_runner_scores_idle_as_failure(self):
        report = EVAL.evaluate_task("h1_2", "pushcube", trials=1, policy="idle",
                                    max_steps=240, settle_steps=120,
                                    verbose=False)
        self.assertEqual(0, report["summary"]["successes"])
        trial = report["trials"][0]
        self.assertLess(trial["joint_travel_rad"], 0.1,
                        "idle 不该让手臂动起来")
        self.assertTrue(trial["scene"]["settle"]["stable"])




class TrialRunnerTests(unittest.TestCase):
    """``run_trial`` 对 callable 策略的支持（摄像头档的动作只能边走边算）与离屏取证图。"""

    def _env(self):
        from teleoperation.simulation.environment import SimulationEnv
        data_dir = EVAL.ROOT / "outputs" / "tmp_task_eval"
        env = SimulationEnv(robot_type="h1_2", task_name="pushcube", render=False,
                            record=False, data_dir=str(data_dir))
        env.reset(randomize=False)
        return env

    def test_run_trial_accepts_a_callable_policy(self):
        """callable 策略必须与"整段先算好"的数组策略走出同一条轨迹。"""
        env = self._env()
        try:
            actions = np.zeros((120, env.action_dim), dtype=np.float64)
            expected = EVAL.run_trial(env, actions, 120)
        finally:
            env.close()
        env = self._env()
        try:
            seen = []

            def policy(step):
                seen.append(step)
                return actions[min(step, len(actions) - 1)]

            result = EVAL.run_trial(env, policy, 120)
        finally:
            env.close()
        self.assertEqual(len(seen), result["steps"])
        self.assertEqual(result["steps"], expected["steps"])
        self.assertIn("wall_limit_hit", result)
        self.assertFalse(result["wall_limit_hit"])
        self.assertAlmostEqual(result["joint_travel_rad"], expected["joint_travel_rad"],
                               places=3)

    def test_run_trial_stops_when_the_policy_asks(self):
        env = self._env()
        try:
            class Stopping:
                stopped = False

                def __init__(self):
                    self.calls = 0

                def __call__(self, step):
                    self.calls += 1
                    if self.calls >= 10:
                        self.stopped = True
                    return np.zeros(env.action_dim, dtype=np.float64)

            policy = Stopping()
            result = EVAL.run_trial(env, policy, 480)
        finally:
            env.close()
        self.assertTrue(result["wall_limit_hit"], "用户按 q 应当结束 trial")
        self.assertLessEqual(result["steps"], 13, result)

    def test_snapshot_writes_a_real_png(self):
        env = self._env()
        try:
            target = EVAL.ROOT / "outputs" / "tmp_task_eval" / "snapshot_test.png"
            self.assertTrue(EVAL.snapshot(env, target), "离屏渲染要能出图")
            self.assertGreater(target.stat().st_size, 1000)
        finally:
            env.close()
        target.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
