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
from unittest.mock import patch

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

    def test_dex_backend_writes_native_joint_results_directly(self):
        from teleoperation.contracts.hand_control import HandJointResult

        result = HandJointResult(
            "h1_2", "right", "a" * 64, ("R_index_proximal_joint",),
            np.array([0.42]), 0.0, solve_ms=1.2)

        class _Retargeter:
            def retarget(self, _window):
                return {("h1_2", "right"): result}

        policy = object.__new__(EVAL.CameraPolicy)
        policy.side = "right"
        policy.hand_enabled = True
        policy.hand_backend_name = "dex"
        policy.hand_backend_note = "dex test"
        policy.hand_limit_mode = "native"
        policy.hand_retargeter = _Retargeter()
        policy.native_hand_spec = SimpleNamespace(
            expand=lambda positions: {
                "R_index_proximal_joint": float(positions[0])})
        policy.hand_names = ["R_index_proximal_joint"]
        policy.names = {"R_index_proximal_joint": 0}
        policy.action = np.zeros(1, dtype=np.float64)
        policy.env = SimpleNamespace(robot_type="h1_2")
        policy.source = SimpleNamespace(last_raw_frame={})
        policy._pinch_signal = lambda *_points: {
            "selected": ("index", 1.0, 0.0),
            "source": "test", "world": None, "image": None,
        }
        from teleoperation.robots.native_hand import PinchSignalStabilizer
        policy._pinch_stabilizer = PinchSignalStabilizer(hold_frames=8)
        policy.calibrated = True
        policy.tracking_started = True
        policy.hand_valid_frames = 0
        policy.hand_invalid_frames = 0
        policy.hand_clipped = 0
        policy.hand_calibration_status = {}
        policy.last_hand = {}
        policy.last_valid_hand = {}
        policy.pinch_finger_frames = {}
        policy.max_pinch_strength = 0.0
        policy.max_fist_strength = 0.0
        policy.last_pinch = None
        policy.last_raw_pinch = None

        policy._update_hand(object())
        self.assertAlmostEqual(0.42, policy.action[0])
        self.assertEqual("dex", policy.last_hand["backend"])
        self.assertIsNone(policy.last_hand["l21_angles"])
        self.assertEqual(1.2, policy.last_hand["solve_ms"])

    def test_dex_pinch_correction_moves_index_target_to_contact_pose(self):
        from teleoperation.contracts.hand_control import HandJointResult
        from teleoperation.robots.hand_spec import load_hand_spec

        spec = load_hand_spec("h1_2", "right")
        result = HandJointResult(
            "h1_2", "right", spec.spec_hash, spec.independent_joint_names,
            np.zeros(spec.dof), 0.0, solve_ms=1.2)
        endpoint = np.zeros(spec.dof)
        endpoint[spec.independent_joint_names.index(
            "R_index_proximal_joint")] = 1.7

        class _Retargeter:
            def retarget(self, _window):
                return {("h1_2", "right"): result}

        policy = object.__new__(EVAL.CameraPolicy)
        policy.side = "right"
        policy.hand_enabled = True
        policy.hand_backend_name = "dex"
        policy.hand_backend_note = "dex test"
        policy.hand_limit_mode = "native"
        policy.hand_retargeter = _Retargeter()
        policy.native_hand_spec = spec
        policy.pinch_contact_poses = {"index": endpoint}
        policy.pinch_correction_alpha = 0.0
        policy.pinch_correction_frames = 0
        policy.hand_names = list(spec.joint_names)
        policy.names = {name: index for index,
                        name in enumerate(spec.independent_joint_names)}
        policy.action = np.zeros(spec.dof, dtype=np.float64)
        policy.env = SimpleNamespace(robot_type="h1_2")
        policy.source = SimpleNamespace(last_raw_frame={})
        policy._pinch_signal = lambda *_points: {
            "selected": ("index", 1.0, 0.0),
            "source": "test", "world": None, "image": None,
        }
        from teleoperation.robots.native_hand import PinchSignalStabilizer
        policy._pinch_stabilizer = PinchSignalStabilizer(hold_frames=2)
        policy.calibrated = True
        policy.tracking_started = True
        policy.hand_valid_frames = 0
        policy.hand_invalid_frames = 0
        policy.hand_clipped = 0
        policy.hand_calibration_status = {}
        policy.last_hand = {}
        policy.last_valid_hand = {}
        policy.pinch_finger_frames = {}
        policy.max_pinch_strength = 0.0
        policy.max_fist_strength = 0.0
        policy.last_pinch = None
        policy.last_raw_pinch = None

        for _ in range(4):
            policy._update_hand(object())
        index = policy.names["R_index_proximal_joint"]
        self.assertAlmostEqual(1.7, policy.action[index])
        self.assertEqual(1.0, policy.last_hand["pinch_correction_alpha"])
        self.assertGreater(policy.pinch_correction_frames, 0)

    def test_closed_dex_fingers_select_the_full_fist_contact_pose(self):
        from teleoperation.contracts.hand_control import HandJointResult
        from teleoperation.robots.hand_spec import load_hand_spec

        spec = load_hand_spec("h1_2", "right")
        closed = np.array(spec.q_reference)
        for index, name in enumerate(spec.independent_joint_names):
            if any(finger in name.lower() for finger in
                   ("index", "middle", "ring", "pinky")):
                closed[index] = spec.bounds[index, 1]
        result = HandJointResult(
            "h1_2", "right", spec.spec_hash, spec.independent_joint_names,
            closed, 0.0, solve_ms=1.0)
        fist = np.array(closed)
        fist[spec.independent_joint_names.index(
            "R_thumb_proximal_yaw_joint")] = 1.3

        class _Retargeter:
            def retarget(self, _window):
                return {("h1_2", "right"): result}

        policy = object.__new__(EVAL.CameraPolicy)
        policy.side = "right"
        policy.hand_enabled = True
        policy.hand_backend_name = "dex"
        policy.hand_backend_note = "dex test"
        policy.hand_limit_mode = "native"
        policy.hand_retargeter = _Retargeter()
        policy.native_hand_spec = spec
        policy.pinch_contact_poses = {"fist": fist}
        policy.pinch_correction_alpha = 0.0
        policy.pinch_correction_frames = 0
        policy.max_dex_raw_closure = 0.0
        policy.max_dex_corrected_closure = 0.0
        policy.hand_names = list(spec.joint_names)
        policy.names = {name: index for index,
                        name in enumerate(spec.independent_joint_names)}
        policy.action = np.zeros(spec.dof, dtype=np.float64)
        policy.env = SimpleNamespace(robot_type="h1_2")
        policy.source = SimpleNamespace(last_raw_frame={})
        policy._pinch_signal = lambda *_points: {
            "selected": ("index", 1.0, 0.0),
            "source": "test", "world": None, "image": None,
        }
        from teleoperation.robots.native_hand import PinchSignalStabilizer
        policy._pinch_stabilizer = PinchSignalStabilizer(hold_frames=2)
        policy.calibrated = True
        policy.tracking_started = True
        policy.hand_valid_frames = 0
        policy.hand_invalid_frames = 0
        policy.hand_clipped = 0
        policy.hand_calibration_status = {}
        policy.last_hand = {}
        policy.last_valid_hand = {}
        policy.pinch_finger_frames = {}
        policy.max_pinch_strength = 0.0
        policy.max_fist_strength = 0.0
        policy.last_pinch = None
        policy.last_raw_pinch = None

        for _ in range(3):
            policy._update_hand(object())
        self.assertEqual("fist", policy.last_hand["pinch_correction_gesture"])
        self.assertGreaterEqual(policy.last_hand["dex_raw_finger_closure"], 0.8)
        self.assertGreater(policy.pinch_correction_alpha, 0.0)

    def test_dex_open_guard_reopens_without_confirmed_grip_intent(self):
        from teleoperation.contracts.hand_control import HandJointResult
        from teleoperation.robots.hand_spec import load_hand_spec

        spec = load_hand_spec("h1_2", "right")
        closed = np.array(spec.q_reference)
        for index, name in enumerate(spec.independent_joint_names):
            if any(finger in name.lower() for finger in
                   ("index", "middle", "ring", "pinky")):
                closed[index] = spec.bounds[index, 1]
        result = HandJointResult(
            "h1_2", "right", spec.spec_hash, spec.independent_joint_names,
            closed, 0.0, solve_ms=1.0)
        fist = np.array(closed)

        class _Retargeter:
            def retarget(self, _window):
                return {("h1_2", "right"): result}

        policy = object.__new__(EVAL.CameraPolicy)
        policy.side = "right"
        policy.hand_enabled = True
        policy.hand_backend_name = "dex"
        policy.hand_backend_note = "dex test"
        policy.hand_limit_mode = "native"
        policy.hand_retargeter = _Retargeter()
        policy.native_hand_spec = spec
        policy.pinch_contact_poses = {"fist": fist}
        policy.pinch_correction_alpha = 0.0
        policy.pinch_correction_frames = 0
        policy.open_guard_alpha = 0.0
        policy.max_dex_raw_closure = 0.0
        policy.max_dex_corrected_closure = 0.0
        policy.hand_names = list(spec.joint_names)
        policy.names = {name: index for index,
                        name in enumerate(spec.independent_joint_names)}
        policy.action = np.zeros(spec.dof, dtype=np.float64)
        policy.env = SimpleNamespace(robot_type="h1_2")
        policy.source = SimpleNamespace(last_raw_frame={})
        policy._pinch_signal = lambda *_points: {
            "selected": (None, 0.0, 0.9),
            "source": "image-open", "world": None, "image": None,
        }
        from teleoperation.robots.native_hand import PinchSignalStabilizer
        policy._pinch_stabilizer = PinchSignalStabilizer(hold_frames=2)
        policy.calibrated = True
        policy.tracking_started = True
        policy.hand_valid_frames = 0
        policy.hand_invalid_frames = 0
        policy.hand_clipped = 0
        policy.hand_calibration_status = {}
        policy.last_hand = {}
        policy.last_valid_hand = {}
        policy.pinch_finger_frames = {}
        policy.max_pinch_strength = 0.0
        policy.max_fist_strength = 0.0
        policy.last_pinch = None
        policy.last_raw_pinch = None

        for _ in range(4):
            policy._update_hand(object())
        index = policy.names["R_index_proximal_joint"]
        self.assertIsNone(policy.last_hand["pinch_correction_gesture"])
        self.assertEqual(1.0, policy.last_hand["open_guard_alpha"])
        self.assertAlmostEqual(0.0, policy.action[index])

    def test_contact_grip_stops_fingers_without_attaching_the_object(self):
        policy = object.__new__(EVAL.CameraPolicy)
        policy.last_pinch = {"finger": "index", "strength": 1.0}
        policy.grasp_constraint = None
        policy.grasp_feedback = {"object": "cube"}
        policy.contact_grip_active = False
        policy.contact_grip_object = None
        policy.contact_grip_targets = {}
        policy.contact_grip_links = []
        policy.contact_grip_count = 0
        policy.contact_grip_open_steps = 0
        policy.grasp_release_confirm_steps = 3
        policy.hand_names = ["R_thumb_distal_joint", "R_index_intermediate_joint"]
        policy.names = {name: index for index, name in enumerate(policy.hand_names)}
        policy.action = np.zeros(3, dtype=np.float64)
        env = SimpleNamespace(
            robot_id=1, client=0, task=SimpleNamespace(objects={"cube": 2}),
            action_joint_indices=[10, 11])
        link_names = {10: b"R_thumb_distal", 11: b"R_index_intermediate"}

        def joint_info(_body, link, **_kwargs):
            info = [0] * 13
            info[12] = link_names[link]
            return info

        with patch.object(EVAL.p, "getContactPoints",
                          return_value=[{3: 10}, {3: 11}]), \
                patch.object(EVAL.p, "getJointInfo", side_effect=joint_info), \
                patch.object(EVAL.p, "getJointState",
                             return_value=[0.7, 0.0, 0.0]):
            policy._update_contact_grip(env)
        self.assertTrue(policy.contact_grip_active)
        self.assertEqual("cube", policy.contact_grip_object)
        self.assertEqual(1, policy.contact_grip_count)
        self.assertEqual(0.7, policy.action[0])
        self.assertIsNone(policy.grasp_constraint,
                          "contact grip must never create a hand-object constraint")

        policy.last_pinch = {"finger": None, "strength": 0.0}
        for _ in range(3):
            policy._update_contact_grip(env)
        self.assertFalse(policy.contact_grip_active)
        self.assertEqual({}, policy.contact_grip_targets)

    def test_contact_grip_requires_both_thumb_and_finger_links(self):
        policy = object.__new__(EVAL.CameraPolicy)
        policy.last_pinch = {"finger": "index", "strength": 1.0}
        policy.grasp_constraint = None
        policy.grasp_feedback = {"object": "cube"}
        policy.contact_grip_active = False
        policy.contact_grip_targets = {}
        policy.contact_grip_links = []
        policy.hand_names = ["R_index_proximal_joint"]
        policy.names = {"R_index_proximal_joint": 0}
        policy.action = np.zeros(1, dtype=np.float64)
        env = SimpleNamespace(
            robot_id=1, client=0, task=SimpleNamespace(objects={"cube": 2}),
            action_joint_indices=[10])

        def joint_info(_body, link, **_kwargs):
            info = [0] * 13
            info[12] = (b"R_index_intermediate" if link == 10
                        else b"R_index_proximal")
            return info

        with patch.object(EVAL.p, "getContactPoints",
                          return_value=[{3: 10}, {3: 11}]), \
                patch.object(EVAL.p, "getJointInfo", side_effect=joint_info):
            policy._update_contact_grip(env)
        self.assertFalse(policy.contact_grip_active)

    def test_camera_objects_stay_dynamic_in_manual_mouse_demo(self):
        self.assertFalse(EVAL.camera_objects_frozen("on", True))
        self.assertTrue(EVAL.camera_objects_frozen("on", False))
        self.assertFalse(EVAL.camera_objects_frozen("off", False))

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
        policy._pinch_signal = lambda *_points: {
            "selected": (None, 0.0, None),
            "source": "none", "world": None, "image": None,
        }
        from teleoperation.robots.native_hand import PinchSignalStabilizer
        policy._pinch_stabilizer = PinchSignalStabilizer(hold_frames=8)
        policy.calibrated = True
        policy.tracking_frames = policy.startup_hold_frames = 12
        policy.tracking_started = True

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
        self.assertEqual(args.camera_pinch_correction, "on")
        self.assertEqual(args.camera_table_clearance_mm, 60.0)
        self.assertEqual(args.camera_arm_force, 200.0)
        self.assertEqual(args.camera_hand_force, 200.0)
        self.assertEqual(args.camera_ik_mode, "pose")
        self.assertEqual(args.camera_ik_solver, "paper")
        self.assertEqual(args.camera_wrist_pose, "on")
        self.assertEqual(args.camera_grasp_assist, "on")
        self.assertEqual(args.camera_freeze_objects, "on")
        self.assertEqual(args.camera_startup_motion_mm, 50.0)
        self.assertEqual(args.camera_startup_confirm_frames, 3)
        self.assertEqual(args.camera_target_deadband_mm, 2.0)
        self.assertEqual(args.camera_target_speed_mm_s, 600.0)
        self.assertFalse(args.camera_table_guard)
        self.assertFalse(args.allow_gui_mouse_picking)
        self.assertFalse(args.setup_gui_mouse_picking)

    def test_pickplace_destination_plate_is_not_a_grasp_assist_candidate(self):
        self.assertEqual({"plate"}, set(EVAL.grasp_assist_excluded(
            "pickplacecube")))
        self.assertEqual(set(), set(EVAL.grasp_assist_excluded("pickcube")))

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

    def test_first_tracking_target_starts_from_the_measured_wrist(self):
        policy = object.__new__(EVAL.CameraPolicy)
        policy.last_commanded_target = None
        policy.target_speed_m_s = 0.6
        policy.last_frame_dt = 0.05
        commanded, limited = policy._slew_target(
            (0.40, 0.20, 1.10), start_position=(0.20, 0.00, 1.00))
        np.testing.assert_allclose(commanded, (0.20, 0.00, 1.00))
        self.assertFalse(limited)

    def test_startup_gate_requires_confirmed_motion_from_calibrated_target(self):
        policy = object.__new__(EVAL.CameraPolicy)
        policy.startup_target_position = None
        policy.tracking_started = False
        policy.startup_hold_frames = 2
        policy.startup_motion_m = 0.05
        policy.startup_confirm_frames = 3
        policy.startup_motion_frames = 0
        policy.startup_target_motion_max_m = 0.0
        policy.tracking_frames = 1

        opened, _distance = policy._update_startup_gate((0.2, 0.0, 1.0))
        self.assertFalse(opened)
        np.testing.assert_allclose(
            policy.startup_target_position, (0.2, 0.0, 1.0))

        policy.tracking_frames = 3
        for _ in range(2):
            opened, distance = policy._update_startup_gate((0.26, 0.0, 1.0))
            self.assertFalse(opened)
            self.assertAlmostEqual(distance, 0.06)
        opened, _distance = policy._update_startup_gate((0.26, 0.0, 1.0))
        self.assertTrue(opened)

    def test_startup_hold_uses_a_strong_motor_budget(self):
        policy = object.__new__(EVAL.CameraPolicy)
        policy.tracking_started = False
        policy.action = np.array([0.1, 0.2], dtype=np.float64)
        policy.arm_names = {"arm"}
        policy.arm_force = 200.0
        policy.hand_force = 120.0
        env = SimpleNamespace(robot_id=7, client=3,
                              action_joint_names=["arm", "finger"],
                              action_joint_indices=[4, 5])

        with patch.object(EVAL.p, "setJointMotorControl2") as motor:
            policy.apply_action(env)
        self.assertEqual([call.kwargs["force"] for call in motor.call_args_list],
                         [2000.0, 2000.0])

        policy.tracking_started = True
        policy._update_grasp_assist = lambda _env: None
        with patch.object(EVAL.p, "setJointMotorControl2") as motor:
            policy.apply_action(env)
        self.assertEqual([call.kwargs["force"] for call in motor.call_args_list],
                         [200.0, 120.0])

    def test_grasp_assist_requires_contact_confirmation_and_distance(self):
        policy = object.__new__(EVAL.CameraPolicy)
        policy.grasp_assist_enabled = True
        policy.grasp_link = 2
        policy.grasp_assist_excluded = set()
        policy.grasp_constraint = None
        policy.grasped_object = None
        policy.grasp_attach_count = 0
        policy.grasp_release_count = 0
        policy.grasp_attach_confirm_steps = 12
        policy.grasp_release_confirm_steps = 12
        policy.grasp_max_attach_gap_m = 0.12
        policy.grasp_candidate_steps = 0
        policy.grasp_candidate_key = None
        policy.grasp_open_steps = 0
        policy.grasp_rejected_distance = 0
        policy.last_pinch = {"strength": 0.5}
        policy.last_hand = {"valid": True}
        env = SimpleNamespace(
            client=3, robot_id=7,
            task=SimpleNamespace(objects={"cup": 11}))
        hand_state = (None, None, None, None,
                      (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), None)

        with patch.object(EVAL.p, "getDynamicsInfo",
                          return_value=(0.2,)), \
                patch.object(EVAL.p, "getContactPoints",
                             return_value=[1, 2]), \
                patch.object(EVAL.p, "getBasePositionAndOrientation",
                             return_value=((0.20, 0.0, 0.0),
                                           (0.0, 0.0, 0.0, 1.0))), \
                patch.object(EVAL.p, "getLinkState", return_value=hand_state):
            policy._update_grasp_assist(env)
        self.assertIsNone(policy.grasp_constraint)
        self.assertEqual(1, policy.grasp_rejected_distance)

        with patch.object(EVAL.p, "getDynamicsInfo",
                          return_value=(0.2,)), \
                patch.object(EVAL.p, "getContactPoints",
                             return_value=[1, 2]), \
                patch.object(EVAL.p, "getBasePositionAndOrientation",
                             return_value=((0.05, 0.0, 0.0),
                                           (0.0, 0.0, 0.0, 1.0))), \
                patch.object(EVAL.p, "getLinkState", return_value=hand_state), \
                patch.object(EVAL.p, "createConstraint", return_value=99), \
                patch.object(EVAL.p, "removeConstraint") as remove:
            for _ in range(11):
                policy._update_grasp_assist(env)
            self.assertIsNone(policy.grasp_constraint)
            policy._update_grasp_assist(env)
            self.assertEqual(99, policy.grasp_constraint)
            self.assertEqual("cup", policy.grasped_object)

            policy.last_pinch = {"strength": 0.0}
            for _ in range(11):
                policy._update_grasp_assist(env)
            self.assertEqual(99, policy.grasp_constraint)
            policy.last_hand = {"valid": False}
            for _ in range(30):
                policy._update_grasp_assist(env)
            self.assertEqual(99, policy.grasp_constraint,
                             "a temporary hand-tracking loss must not release")
            policy.last_hand = {"valid": True}
            for _ in range(12):
                policy._update_grasp_assist(env)
            self.assertIsNone(policy.grasp_constraint)
            self.assertEqual(1, policy.grasp_release_count)
            remove.assert_called_once()

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
        self.assertEqual(joints["R_thumb_intermediate_joint"], 0.0)
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

    def test_grasp_uses_a_shallow_finger_hook_instead_of_crushing_the_object(self):
        from teleoperation.robots.native_hand import map_dofs

        mapping = [
            (2, "R_index_proximal_joint", 1.0, 0.0, 1.7),
            (3, "R_index_intermediate_joint", 1.0, 0.0, 1.7),
            (5, "R_middle_proximal_joint", 1.0, 0.0, 1.7),
            (6, "R_middle_intermediate_joint", 1.0, 0.0, 1.7),
            (16, "R_thumb_intermediate_joint", 1.0, 0.0, 0.8),
        ]
        joints, _ = map_dofs(
            np.zeros(17), mapping, pinch=("grasp", 1.0, 0.5),
            robot_type="h1_2", side="right")
        self.assertAlmostEqual(joints["R_index_proximal_joint"], 1.7)
        self.assertEqual(joints["R_index_intermediate_joint"], 0.0)
        self.assertAlmostEqual(joints["R_middle_proximal_joint"], 1.7)
        self.assertEqual(joints["R_middle_intermediate_joint"], 0.0)

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

    def test_liftmug_no_longer_succeeds_with_the_lid_seated(self):
        row = AUDIT.check_task(self.client, "liftmug", 120)
        self.assertFalse(row["succ_at_reset"])
        self.assertTrue(row["succ_oracle"], row)
        self.assertEqual([], row["flags"], row)

    def test_corrected_tool_task_criteria_are_evaluable(self):
        for name in ("opendrawer", "openmicrowave", "breadtoaster",
                     "rotatehearth", "pottomatoout"):
            with self.subTest(task=name):
                row = AUDIT.check_task(self.client, name, 120)
                self.assertFalse(row["succ_at_reset"], row)
                self.assertTrue(row["succ_oracle"], row)
                self.assertEqual([], row["flags"], row)

    def test_pottomatoout_starts_with_tomato_inside_a_hollow_pot(self):
        from teleoperation.simulation.tasks import get_task

        task = get_task("pottomatoout")(self.client)
        try:
            task.reset()
            tomato = task.get_object_position("tomato")
            pot = task.get_object_position("pot")
            self.assertLess(float(np.linalg.norm(tomato[:2] - pot[:2])), 0.06)
            self.assertFalse(task.check_success())
            self.assertGreater(tomato[2], pot[2] - 0.04)
        finally:
            task._clear_objects()

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

    def test_pickcube_uses_the_native_hand_pre_grasp_region(self):
        from teleoperation.simulation.environment import SimulationEnv

        env = SimulationEnv(robot_type="h1_2", task_name="pickcube",
                            render=False, record=False,
                            data_dir=str(EVAL.ROOT / "outputs" / "tmp_task_eval"))
        try:
            env.reset(randomize=False)
            arm = EVAL.measure_arm("h1_2")
            scene = EVAL.TaskScene(
                env.client, env.task, robot="h1_2",
                shoulder=arm["shoulder_m"], reach=arm["reach_m"],
                name="pickcube", robot_id=env.robot_id)
            report = scene.prepare(
                120, ik_workspace=True, side="right", hold_robot=True,
                layout_forward=0.38, layout_drop=0.18,
                layout_lateral=0.15, layout_side_bias=-0.06,
                layout_table_min_x=0.29, camera_cube_pre_grasp=True,
                ik_comfort_fraction=0.98)
            self.assertIn("camera_cube_pre_grasp", report)
            cube = EVAL.body_pose(env.client, env.task.objects["cube"])[0]
            target = np.asarray(
                report["camera_cube_pre_grasp"]["target_center_m"])
            self.assertLessEqual(float(np.linalg.norm(cube - target)), 0.01)
            low, high = EVAL.aabb(env.client, env.task.objects["cube"])
            self.assertLess(float(high[0] - low[0]), 0.05,
                            "cube must fit inside the native pinch envelope")
        finally:
            env.close()

    def test_camera_side_bias_keeps_pickplace_objects_on_the_table(self):
        from teleoperation.simulation.environment import SimulationEnv

        env = SimulationEnv(robot_type="h1_2", task_name="pickplacecube",
                            render=False, record=False,
                            data_dir=str(EVAL.ROOT / "outputs" / "tmp_task_eval"))
        try:
            env.reset(randomize=False)
            arm = EVAL.measure_arm("h1_2")
            scene = EVAL.TaskScene(
                env.client, env.task, robot="h1_2",
                shoulder=arm["shoulder_m"], reach=arm["reach_m"],
                name="pickplacecube", robot_id=env.robot_id)
            scene.prepare(
                120, ik_workspace=True, side="right", hold_robot=True,
                layout_forward=0.38, layout_drop=0.18,
                layout_lateral=0.15, layout_side_bias=-0.06,
                layout_table_min_x=0.29,
                camera_pickplace_spacing=True,
                ik_comfort_fraction=0.98)
            supported, reason = EVAL.task_objects_supported(env)
            self.assertTrue(supported, reason)
            low, _high = EVAL.aabb(env.client, env.task.objects["table"])
            self.assertGreaterEqual(float(low[0]), 0.20)
            self.assertIn("camera_cube_pre_grasp", scene.report)
            cube = EVAL.body_pose(env.client, env.task.objects["cube"])[0]
            plate = EVAL.body_pose(env.client, env.task.objects["plate"])[0]
            self.assertGreaterEqual(
                scene.report["camera_pickplace_spacing_m"], 0.13,
                "抓取起点和放置目标必须留出足够的搬运距离")
            self.assertGreaterEqual(
                float(cube[0]), float(low[0]) + 0.054,
                "方块必须深入桌面，不能贴着机器人侧桌沿")
            self.assertGreaterEqual(
                float(cube[0]), float(scene.shoulder[0]) + 0.20,
                "预抓取方块不能退到机器人躯干内部")
            self.assertGreaterEqual(
                float(plate[0]), float(scene.shoulder[0]) + 0.33,
                "放置区必须明显位于机器人近侧抓取区之外")
            self.assertGreaterEqual(
                float(EVAL.p.getDynamicsInfo(
                    env.task.objects["plate"], -1,
                    physicsClientId=env.client)[0]), 0.0,
                "放置盘必须是固定目标，不能被机器人轻碰就滑走")
            for key, body in (("cube", env.task.objects["cube"]),
                              ("plate", env.task.objects["plate"])):
                residual = scene.report["ik_reach"]["objects"][key][
                    "residual_mm"]
                # The camera cube is moved to the native-hand pinch centre
                # after the generic IK pass; 2 cm covers that final shift.
                self.assertLessEqual(residual, 20.0, (key, residual))
        finally:
            env.close()

    def test_camera_balltrashcan_uses_a_real_pre_grasp_start(self):
        from teleoperation.simulation.environment import SimulationEnv

        env = SimulationEnv(robot_type="h1_2", task_name="balltrashcan",
                            render=False, record=False,
                            data_dir=str(EVAL.ROOT / "outputs" / "tmp_task_eval"))
        try:
            env.reset(randomize=False)
            arm = EVAL.measure_arm("h1_2")
            scene = EVAL.TaskScene(
                env.client, env.task, robot="h1_2",
                shoulder=arm["shoulder_m"], reach=arm["reach_m"],
                name="balltrashcan", robot_id=env.robot_id)
            report = scene.prepare(
                120, ik_workspace=True, side="right", hold_robot=True,
                layout_forward=0.38, layout_drop=0.18,
                layout_lateral=0.15, layout_side_bias=-0.06,
                layout_table_min_x=0.29,
                camera_balltrashcan_setup=True)
            supported, reason = EVAL.task_objects_supported(env)
            self.assertTrue(supported, reason)
            self.assertFalse(report["criteria_at_reset"])
            ball = EVAL.body_pose(env.client, env.task.objects["ball"])[0]
            target = np.asarray(
                report["camera_balltrashcan_setup"]["target_center_m"])
            np.testing.assert_allclose(ball, target, atol=0.01)
            self.assertEqual(
                0.0, float(EVAL.p.getDynamicsInfo(
                    env.task.objects["trashcan"], -1,
                    physicsClientId=env.client)[0]))
            self.assertGreaterEqual(
                float(EVAL.p.getDynamicsInfo(
                    env.task.objects["ball"], -1,
                    physicsClientId=env.client)[6]), 0.10)
            low, _high = EVAL.aabb(env.client, env.task.objects["table"])
            self.assertGreaterEqual(float(ball[0]) - float(low[0]), 0.05)

            can_position = EVAL.body_pose(
                env.client, env.task.objects["trashcan"])[0]
            EVAL.p.resetBasePositionAndOrientation(
                env.task.objects["ball"],
                [float(can_position[0]), float(can_position[1]), 1.23],
                [0.0, 0.0, 0.0, 1.0], physicsClientId=env.client)
            EVAL.p.resetBaseVelocity(
                env.task.objects["ball"], [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0], physicsClientId=env.client)
            for _ in range(480):
                EVAL.p.stepSimulation(physicsClientId=env.client)
            self.assertTrue(env.task.check_success())
            self.assertLess(
                float(EVAL.body_pose(
                    env.client, env.task.objects["ball"])[0][2]),
                float(can_position[2]) + 0.08)
        finally:
            env.close()

    def test_trial_runner_scores_idle_as_failure(self):
        report = EVAL.evaluate_task("h1_2", "pushcube", trials=1, policy="idle",
                                    max_steps=240, settle_steps=120,
                                    verbose=False)
        self.assertEqual(0, report["summary"]["successes"])
        trial = report["trials"][0]
        self.assertLess(trial["joint_travel_rad"], 0.1,
                        "idle 不该让手臂动起来")
        self.assertTrue(trial["scene"]["settle"]["stable"])

    def test_camera_uprearcup_has_rolling_resistance(self):
        from teleoperation.simulation.environment import SimulationEnv

        env = SimulationEnv(robot_type="h1_2", task_name="uprearcup",
                            render=False, record=False,
                            data_dir=str(EVAL.ROOT / "outputs" / "tmp_task_eval"))
        try:
            env.reset(randomize=False)
            arm = EVAL.measure_arm("h1_2")
            workspace = EVAL.camera_workspace_profile("uprearcup")
            position, orientation = EVAL.p.getBasePositionAndOrientation(
                env.robot_id, physicsClientId=env.client)
            EVAL.p.resetBasePositionAndOrientation(
                env.robot_id,
                [position[0], position[1],
                 position[2] - workspace["robot_drop_m"]],
                orientation, physicsClientId=env.client)
            shoulder = np.asarray(arm["shoulder_m"], dtype=float).copy()
            shoulder[2] -= workspace["robot_drop_m"]
            scene = EVAL.TaskScene(
                env.client, env.task, robot="h1_2",
                shoulder=shoulder, reach=arm["reach_m"],
                name="uprearcup", robot_id=env.robot_id)
            report = scene.prepare(
                120, ik_workspace=True, side="right", hold_robot=True,
                layout_forward=workspace["layout_forward_m"],
                layout_drop=(shoulder[2] - workspace["table_top_m"]),
                layout_lateral=0.15, layout_side_bias=-0.06,
                layout_table_min_x=workspace["layout_table_min_x_m"],
                camera_cup_stability=True)
            self.assertTrue(report["ik_reach"]["ok"], report["ik_reach"])
            _low, high = EVAL.aabb(env.client, env.task.objects["table"])
            self.assertAlmostEqual(workspace["table_top_m"], float(high[2]),
                                   delta=0.01)
            settings = report["camera_cup_stability"]
            self.assertEqual(0.0, settings["rollingFriction"])
            self.assertIn("compound_body", settings)
            self.assertGreaterEqual(settings["mass_kg"], 0.15)
            disabled = EVAL.disable_robot_object_collision_except_hand(
                env, "cup", side="right")
            self.assertGreater(disabled, 0)

            EVAL.p.resetBaseVelocity(
                env.task.objects["cup"], [1.0, 0.0, 2.0], [0.0, 0.0, 3.0],
                physicsClientId=env.client)
            self.assertTrue(EVAL.clamp_horizontal_velocity(
                env.client, env.task.objects["cup"]))
            velocity, angular = EVAL.p.getBaseVelocity(
                env.task.objects["cup"], physicsClientId=env.client)
            self.assertAlmostEqual(0.30, float(np.linalg.norm(velocity[:2])),
                                   places=5)
            self.assertAlmostEqual(1.5, float(np.linalg.norm(angular)),
                                   places=5)
            self.assertAlmostEqual(2.0, float(velocity[2]), places=5,
                                   msg="速度保护不能改变竖直重力方向")

            cup = env.task.objects["cup"]
            start, start_orientation = EVAL.body_pose(env.client, cup)
            start = start.copy()
            target = np.asarray(
                report["camera_cup_pre_grasp"]["target_center_m"])
            np.testing.assert_allclose(start, target, atol=0.01,
                                       err_msg="杯子必须停在右手预抓取中心")
            low, _high = EVAL.aabb(env.client, env.task.objects["table"])
            self.assertGreaterEqual(
                float(start[0]) - 0.05 - float(low[0]), 0.04,
                "杯体沿 x 的整个近端必须有桌面支撑")
            # Test rolling separately from the open-hand pre-grasp volume.
            roll_start = np.array([0.25, -0.06, float(start[2])])
            EVAL.p.resetBasePositionAndOrientation(
                cup, roll_start.tolist(), start_orientation,
                physicsClientId=env.client)
            start = roll_start
            start_orientation = EVAL.body_pose(env.client, cup)[1]
            tilted = EVAL.p.getQuaternionFromEuler([0.0, np.pi / 4.0, 0.0])
            EVAL.p.resetBasePositionAndOrientation(
                cup, [float(start[0]), float(start[1]), 1.22], tilted,
                physicsClientId=env.client)
            EVAL.p.resetBaseVelocity(
                cup, [0.0, 0.0, 0.0], [0.30, 0.0, 0.0],
                physicsClientId=env.client)
            EVAL.p.changeDynamics(
                cup, -1,
                activationState=(
                    EVAL.p.ACTIVATION_STATE_DISABLE_SLEEPING),
                physicsClientId=env.client)
            for _ in range(480):
                EVAL.p.stepSimulation(physicsClientId=env.client)
            _position, end_orientation = EVAL.body_pose(env.client, cup)
            difference = EVAL.p.getDifferenceQuaternion(
                tilted, end_orientation, physicsClientId=env.client)
            angle = 2.0 * np.arccos(min(
                1.0, abs(float(difference[3]))))
            self.assertGreater(
                float(np.degrees(angle)), 15.0,
                "斜立的杯子必须受重力自然倒下，不能被摩擦锁死")

            EVAL.p.resetBasePositionAndOrientation(
                cup, [float(start[0]), float(start[1]), float(start[2])],
                start_orientation, physicsClientId=env.client)
            EVAL.p.resetBaseVelocity(
                cup, [0.05, 0.0, 0.0], [0.0, 0.0, 1.0],
                physicsClientId=env.client)
            EVAL.p.changeDynamics(
                cup, -1,
                activationState=EVAL.p.ACTIVATION_STATE_WAKE_UP,
                physicsClientId=env.client)
            scene._hold_motors()
            for _ in range(480):
                EVAL.p.stepSimulation(physicsClientId=env.client)
            end = EVAL.body_pose(env.client, cup)[0]
            self.assertLess(float(np.linalg.norm((end - start)[:2])), 0.20,
                            "轻微扰动不应让杯子持续滚出桌面")
        finally:
            env.close()




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

    def test_transient_success_does_not_end_the_trial(self):
        env = self._env()
        try:
            calls = {"count": 0}

            def transient():
                calls["count"] += 1
                return calls["count"] <= 3

            env.task.check_success = transient
            result = EVAL.run_trial(
                env, np.zeros((60, env.action_dim), dtype=np.float64), 60,
                success_hold_steps=10)
        finally:
            env.close()
        self.assertFalse(result["success"], result)
        self.assertLess(result["success_streak_max_steps"], 10, result)

    def test_success_must_hold_for_the_configured_window(self):
        env = self._env()
        try:
            calls = {"count": 0}

            def sustained():
                calls["count"] += 1
                return calls["count"] >= 5

            env.task.check_success = sustained
            result = EVAL.run_trial(
                env, np.zeros((30, env.action_dim), dtype=np.float64), 30,
                success_hold_steps=10)
        finally:
            env.close()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["steps"], 14, result)

    def test_demo_mode_keeps_running_after_success(self):
        env = self._env()
        try:
            env.task.check_success = lambda: True
            result = EVAL.run_trial(
                env, np.zeros((30, env.action_dim), dtype=np.float64), 30,
                success_hold_steps=5, stop_on_success=False)
        finally:
            env.close()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["steps"], 30, result)

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
