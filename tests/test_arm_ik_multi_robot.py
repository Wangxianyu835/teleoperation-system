"""Multi-robot arm IK: profiles, per-robot joint selection, and headless tracking.

Covers the three layers added on the simulation side (none of them touch the
teammate's baseline files):

    teleoperation.simulation.arm_profiles     per-robot table (pure arithmetic)
    teleoperation.simulation.arm_ik           PyBullet IK per loaded robot
    teleoperation.simulation.arm_scene_driver release the scene's arm holds + drive

The numbers asserted here are the ones measured on this machine
(``devtools/verify_arm_multi_robot.py``, 2026-10-12, gravity -9.81, holds active,
full sweep per profile): h1_2 pose mean 17.0 / max 23.8 mm, gr1_t2 position mean 8.9
/ max 13.3 mm, g1 position mean 7.7 / max 14.9 mm, travel within 1 mm of the target.
The thresholds come from each profile, so a robot whose arm stops being tracked fails
this test instead of silently producing a frozen arm.
"""

import unittest
from pathlib import Path

import numpy as np

from teleoperation.simulation.arm_profiles import (
    ARM_PROFILES,
    SIDES,
    TRON2A,
    ArmProfile,
    all_robots,
    classify_arm_dof,
    describe,
    loadable_robots,
    profile_for,
    table,
)

ROOT = Path(__file__).resolve().parents[1]
ROBOTS_ROOT = ROOT / "assets" / "robots" / "from_teleopbench"
URDF_PATHS = {"h1_2": ROBOTS_ROOT / "h1_2" / "h1_2.urdf",
              "gr1_t2": ROBOTS_ROOT / "gr1" / "urdf" / "robot.urdf",
              "g1": ROBOTS_ROOT / "g1" / "g1_29dof_with_hand_lock_waist.urdf"}
ASSETS_AVAILABLE = all(path.is_file() for path in URDF_PATHS.values())
PAPER_ROBOTS = ("h1_2", "gr1_t2", "g1")
#: Hand mount each robot ends at (measured 2026-10-10); gr1_t2 has no wrist link.
EE_LINKS = {"h1_2": "right_wrist_yaw_link",
            "gr1_t2": "right_hand_pitch_link",
            "g1": "right_wrist_yaw_link"}
try:
    import pybullet  # noqa: F401  (import check only)

    PYBULLET_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without pybullet
    PYBULLET_AVAILABLE = False


class ArmProfileTableTests(unittest.TestCase):
    """The table is the single place that decides which arm a robot_type gets."""

    def test_every_paper_robot_is_loadable_and_tron2a_is_only_registered(self):
        self.assertEqual(set(loadable_robots()), set(PAPER_ROBOTS))
        self.assertIn(TRON2A, all_robots())
        self.assertFalse(ARM_PROFILES[TRON2A].loadable)
        self.assertNotIn(TRON2A, loadable_robots())

    def test_wrong_robot_type_fails_loudly_with_the_known_names(self):
        with self.assertRaisesRegex(KeyError, "h1_2"):
            profile_for("H1-2")
        with self.assertRaisesRegex(ValueError, "registered but not loadable"):
            profile_for(TRON2A)

    def test_ik_mode_is_driven_by_the_robot_not_by_the_caller(self):
        # h1_2 keeps the orientation constraint; the other two do not (gr1_t2 has no
        # wrist link, and both track the uncalibrated orientation convention poorly).
        self.assertEqual(profile_for("h1_2").ik_mode, "pose")
        self.assertEqual(profile_for("g1").ik_mode, "position")
        self.assertEqual(profile_for("gr1_t2").ik_mode, "position")

    def test_operator_targets_are_scaled_per_robot(self):
        for robot in PAPER_ROBOTS:
            profile = profile_for(robot)
            mapped = profile.base_target_from_operator((0.10, -0.05, 0.02))
            expected = tuple(offset + profile.metric_scale * value
                             for offset, value in zip(profile.metric_offset,
                                                      (0.10, -0.05, 0.02)))
            np.testing.assert_allclose(mapped, expected, atol=1e-12)
        # The same operator motion therefore reaches different base-frame targets.
        h1_2, g1 = profile_for("h1_2"), profile_for("g1")
        self.assertGreater(h1_2.metric_scale, g1.metric_scale)
        self.assertNotEqual(h1_2.base_target_from_operator((0.1, 0.0, 0.0)),
                            g1.base_target_from_operator((0.1, 0.0, 0.0)))

    def test_reach_box_is_centred_on_the_profile_offset(self):
        profile = profile_for("h1_2")
        low, high = profile.reach(0.05)
        np.testing.assert_allclose(low, np.asarray(profile.metric_offset) - 0.05)
        np.testing.assert_allclose(high, np.asarray(profile.metric_offset) + 0.05)

    def test_bad_profile_values_are_rejected_at_construction(self):
        with self.assertRaisesRegex(ValueError, "ik_mode"):
            ArmProfile(robot_type="x", ik_mode="twist")
        with self.assertRaisesRegex(ValueError, "metric_scale"):
            ArmProfile(robot_type="x", metric_scale=0.0)
        with self.assertRaisesRegex(ValueError, "metric_offset"):
            ArmProfile(robot_type="x", metric_offset=(0.1, 0.0))

    def test_describe_and_table_are_json_able(self):
        import json

        payload = json.dumps(table())
        self.assertIn("h1_2", payload)
        entry = describe("gr1_t2")
        self.assertEqual(entry["ik_mode"], "position")
        self.assertEqual(len(entry["metric_offset"]), 3)
        self.assertTrue(entry["loadable"])
        self.assertFalse(describe(TRON2A)["loadable"])



@unittest.skipUnless(ASSETS_AVAILABLE, "assets/robots/from_teleopbench is missing (not in git)")
@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet is not installed")
class ThreeRobotArmIkTests(unittest.TestCase):
    """One shared headless scene: every arm must track, everything else must hold."""

    @classmethod
    def setUpClass(cls):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

        # The scene is the teammate's class, built exactly as the realtime app builds
        # it (hands on, no display); only the arms are ours afterwards.
        cls.scene = ThreeRobotHandScene(sides=("right",), gui=False)
        cls.robots = {robot["type"]: robot for robot in cls.scene.robots}
        cls.held_before = {name: len(robot["hold_ids"])
                           for name, robot in cls.robots.items()}
        cls.driver = ArmSceneDriver(cls.scene, sides=("right",))
        cls.driver.step(5)

    @classmethod
    def tearDownClass(cls):
        cls.driver = None
        cls.gui = None
        cls.scene.close()

    def test_one_adapter_per_loaded_robot_with_seven_arm_joints(self):
        self.assertEqual(set(self.driver.ik), set(PAPER_ROBOTS))
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            with self.subTest(robot=robot):
                self.assertEqual(adapter.robot_type, robot)
                self.assertEqual(adapter.ik_mode, profile_for(robot).ik_mode)
                self.assertEqual(len(adapter.arm["right"]), 7)
                self.assertEqual(classify_arm_dof(adapter.body, adapter.cid),
                                 {"left": 7, "right": 7})
                self.assertEqual(adapter.ee["right"][1], EE_LINKS[robot])
                # h1_2 keeps the orientation constraint, the other two do not.
                self.assertEqual(adapter.ik_mode, profile_for(robot).ik_mode)

    def test_scene_releases_exactly_the_driven_arm_joints(self):
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            expected = sorted(name for _index, name in adapter.arm["right"])
            with self.subTest(robot=robot):
                released = self.driver.released[robot]
                self.assertEqual(released["count"], 7)
                self.assertEqual(sorted(released["joints"]), expected)
                self.assertEqual(released["held_after"],
                                 self.held_before[robot] - 7)
                self.assertGreater(released["held_after"], 0)
                for index in adapter.arm_ids["right"]:
                    self.assertNotIn(index, self.robots[robot]["hold_ids"])

    @staticmethod
    def reset_arm(adapter):
        """Put the arm back where the loader left it, so two solves start alike."""
        for index, angle in adapter.hold.items():
            if index in adapter.driven:
                pybullet.resetJointState(adapter.body, index, angle,
                                         physicsClientId=adapter.cid)

    def test_ik_mode_from_the_profile_decides_if_orientation_is_constrained(self):
        way_off = (0.0, 0.0, 0.0, 1.0)
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            profile = profile_for(robot)
            with self.subTest(robot=robot):
                self.reset_arm(adapter)
                values = adapter.solve("right", profile.metric_offset,
                                       orientation=way_off)
                self.assertEqual(sorted(values),
                                 sorted(name for _i, name in adapter.arm["right"]))
                if profile.ik_mode == "pose":
                    np.testing.assert_allclose(adapter.last_orientation["right"],
                                               way_off, atol=1e-12)
                else:
                    # A position-only profile must ignore the orientation argument:
                    # same start, same target -> same numbers (solve() warm-starts,
                    # so the arm is reset in between).
                    self.assertIsNone(adapter.last_orientation["right"])
                    self.reset_arm(adapter)
                    baseline = adapter.solve("right", profile.metric_offset)
                    np.testing.assert_allclose(list(values.values()),
                                               list(baseline.values()), atol=1e-9)

    def test_each_arm_tracks_its_own_base_frame_target_and_moves(self):
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            profile = profile_for(robot)
            half = 0.5 * profile.sweep_amplitude_m
            offset = np.asarray(profile.metric_offset)
            corners = [offset + np.array([half, 0.0, half]),
                       offset + np.array([-half, 0.0, -half])]
            positions = []
            for corner in corners:
                self.driver.apply("right", corner, source="base")
                self.driver.step(16)
                positions.append(np.asarray(adapter.measure("right")[0]))
            with self.subTest(robot=robot):
                residual = adapter.residual_mm("right")
                self.assertLess(residual, profile.max_residual_mm,
                                f"{robot}: residual {residual:.1f} mm")
                travel = float(np.ptp(np.asarray(positions), axis=0).max() * 1e3)
                self.assertGreater(travel, profile.min_travel_mm,
                                   f"{robot}: wrist travelled {travel:.1f} mm")

    def test_one_operator_motion_reaches_all_three_arms(self):
        operator = (0.05, 0.0, 0.0)
        reports = self.driver.apply_operator("right", operator)
        self.assertEqual(set(reports), set(PAPER_ROBOTS))
        world_x = {}
        for robot, report in reports.items():
            adapter = self.driver.ik[robot]
            profile = profile_for(robot)
            expected_base = profile.base_target_from_operator(operator)
            with self.subTest(robot=robot):
                np.testing.assert_allclose(adapter.last_target_world["right"],
                                           adapter.to_world(expected_base), atol=1e-9)
                # The operator's 5 cm is scaled and offset per robot, never metres-as-is.
                self.assertGreater(expected_base[0], operator[0])
                self.assertEqual(report["ee"], EE_LINKS[robot])
                world_x[robot] = adapter.last_target_world["right"][0]
        # One motion, three robots: the targets keep the scene's spacing.  The URDFs
        # carry a few centimetres of base X of their own, hence the loose delta.
        self.assertAlmostEqual(world_x["h1_2"] - world_x["gr1_t2"], -2.4, delta=0.25)
        self.assertAlmostEqual(world_x["g1"] - world_x["gr1_t2"], 2.4, delta=0.25)

    def test_world_space_targets_must_include_the_base_height(self):
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            profile = profile_for(robot)
            base = np.asarray(adapter.base_position())
            base_target = np.asarray(profile.metric_offset)
            with self.subTest(robot=robot):
                self.assertGreater(base[2], 0.5)
                np.testing.assert_allclose(adapter.target_in_world(base_target, "base"),
                                           base + base_target, atol=1e-9)
                naive = np.asarray(adapter.target_in_world(base_target, "world"))
                np.testing.assert_allclose(naive, base_target, atol=1e-9)
                # Taking base-frame numbers as world coordinates lands base_z low:
                # the measured trap is ~866 mm on h1_2 (base_z 1.00 + ground lift).
                self.assertGreater(abs(naive[2] - (base + base_target)[2]), 0.7)

    def test_a_relative_orientation_is_composed_with_the_neutral_pose(self):
        from teleoperation.simulation.arm_ik import quaternion_multiply

        adapter = self.driver.ik["h1_2"]                 # the pose-mode profile
        self.reset_arm(adapter)
        relative = (0.0, 0.130526192220052, 0.0, 0.991444861373810)   # 15 deg about y
        adapter.solve("right", profile_for("h1_2").metric_offset,
                      orientation=relative, orientation_frame="neutral")
        expected = quaternion_multiply(adapter.neutral_orientation["right"], relative)
        np.testing.assert_allclose(adapter.last_orientation["right"], expected, atol=1e-12)
        # A relative identity means "stay at the neutral pose", i.e. the composed
        # target is exactly the build-time orientation.
        self.reset_arm(adapter)
        adapter.solve("right", profile_for("h1_2").metric_offset,
                      orientation=(0.0, 0.0, 0.0, 1.0), orientation_frame="neutral")
        np.testing.assert_allclose(adapter.last_orientation["right"],
                                   adapter.neutral_orientation["right"], atol=1e-12)

    def test_an_unknown_orientation_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "orientation_frame"):
            self.driver.ik["h1_2"].solve("right", profile_for("h1_2").metric_offset,
                                         orientation=(0.0, 0.0, 0.0, 1.0),
                                         orientation_frame="laptop")

    def test_posture_metrics_see_the_shoulder_elbow_and_upper_arm_move(self):
        import json

        adapter = self.driver.ik["h1_2"]
        self.reset_arm(adapter)
        adapter.clear_posture()
        base = np.asarray(profile_for("h1_2").realtime_neutral("right"), dtype=np.float64)
        for offset in (np.zeros(3), np.array([0.12, 0.0, 0.08]),
                       np.array([0.02, -0.12, 0.12]), np.array([0.10, 0.10, 0.0])):
            self.driver.apply_each("right", {robot: tuple(base + offset)
                                             for robot in PAPER_ROBOTS})
            self.driver.step(6)
        entry = adapter.posture_summary("right")
        json.dumps(entry)
        # 单侧驱动时量不出双臂间隙（这个 driver 只驱动右侧）——哨兵值而不是 0。
        self.assertEqual(self.driver.forearm_gap_mm(), {})
        self.assertGreaterEqual(entry["frames"], 4)
        self.assertGreater(entry["shoulder_travel_deg"], 1.0, entry)
        self.assertGreater(entry["upper_arm_dir_travel_deg"], 1.0, entry)
        self.assertGreater(entry["elbow_span_mm"], 5.0, entry)
        self.assertGreaterEqual(len(entry["shoulder_range_deg"]), 3)
        self.assertLess(entry["elbow_bend_deg"]["min"], 180.0)
        # The three FK points are the real arm: H1-2 measures 331 mm + 229 mm
        # (2026-10-12), so a wrong link mapping would show up right here.
        points = adapter.arm_points("right")
        upper = float(np.linalg.norm(np.asarray(points["elbow"]) - np.asarray(points["shoulder"])))
        forearm = float(np.linalg.norm(np.asarray(points["wrist"]) - np.asarray(points["elbow"])))
        self.assertAlmostEqual(upper, 0.331, delta=0.02)
        self.assertAlmostEqual(forearm, 0.229, delta=0.02)

    def test_non_driven_joints_stay_held_while_the_arms_move(self):
        self.driver.apply_each("right", {robot: profile_for(robot).metric_offset
                                         for robot in PAPER_ROBOTS})
        self.driver.step(12)
        drift = self.driver.hold_drift()
        for robot in PAPER_ROBOTS:
            entry = drift[robot]
            with self.subTest(robot=robot):
                self.assertLess(entry["max_drift_rad"], 0.05, entry)
                self.assertGreater(entry["held_joints"], 0)
                # Held-but-unmapped hand joints droop under gravity; informational.
                self.assertLess(entry["hand_drift_rad"], 0.3, entry)
        for entry in self.scene.summary()["robots"]:
            self.assertEqual(entry["held_joints"],
                             self.held_before[entry["robot"]] - 7)

    def test_apply_each_needs_a_target_for_every_driven_robot(self):
        with self.assertRaisesRegex(ValueError, "no target given"):
            self.driver.apply_each("right", {"h1_2": (0.24, 0.0, 0.19)})
        with self.assertRaisesRegex(ValueError, "not driven"):
            self.driver.apply("left", (0.24, 0.0, 0.19))

    def test_summary_states_which_arm_each_robot_got(self):
        import json

        self.driver.apply("right", (0.2, 0.0, 0.15), source="base")
        self.driver.step(8)
        payload = self.driver.summary()
        json.dumps(payload)          # report payload must stay JSON-able
        self.assertEqual(payload["scene"], "three_robots+arm_ik")
        self.assertEqual(payload["released_arm_joints"],
                         {robot: 7 for robot in PAPER_ROBOTS})
        self.assertEqual(set(payload["residual_mm"]), set(PAPER_ROBOTS))
        self.assertIn("hands", payload)
        for robot in PAPER_ROBOTS:
            entry = payload["robots"][robot]
            with self.subTest(robot=robot):
                self.assertEqual(entry["ik_mode"], profile_for(robot).ik_mode)
                self.assertEqual(entry["driven_joints"], 7)
                self.assertEqual(entry["end_effector"]["right"]["link"],
                                 EE_LINKS[robot])
                self.assertIn("right", entry["arm_joints"])


class SegmentDistanceTests(unittest.TestCase):
    """Two-segment shortest distance: the geometry behind the forearm-gap metric."""

    def test_known_cases(self):
        from teleoperation.simulation.arm_scene_driver import _segment_distance

        # Parallel, 0.1 m apart
        self.assertAlmostEqual(_segment_distance((0, 0, 0), (1, 0, 0),
                                                 (0, 0.1, 0), (1, 0.1, 0)), 0.1, places=12)
        # Crossing at right angles -> touching
        self.assertAlmostEqual(_segment_distance((0, 0, 0), (1, 0, 0),
                                                 (0.5, -1, 0), (0.5, 1, 0)), 0.0, places=12)
        # Collinear but disjoint -> end-to-end distance
        self.assertAlmostEqual(_segment_distance((0, 0, 0), (1, 0, 0),
                                                 (2, 0, 0), (3, 0, 0)), 1.0, places=12)
        # Degenerate: a zero-length segment on either side is a point
        self.assertAlmostEqual(_segment_distance((0, 0, 0), (0, 0, 0),
                                                 (0, 0, 3), (0, 0, 3)), 3.0, places=12)
        self.assertAlmostEqual(_segment_distance((0, 0, 0), (0, 0, 0),
                                                 (0, 0, 3), (1, 0, 3)), 3.0, places=12)


@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet is not installed")
class TwoSidedForearmGapTests(unittest.TestCase):
    """Both arms driven: the left/right forearm gap (the crossing / midline signal)."""

    @classmethod
    def setUpClass(cls):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

        cls.scene = ThreeRobotHandScene(sides=("left", "right"), gui=False)
        cls.driver = ArmSceneDriver(cls.scene, sides=("left", "right"))
        cls.driver.step(5)

    @classmethod
    def tearDownClass(cls):
        cls.driver = None
        cls.scene.close()

    def test_the_gap_is_measured_per_robot_and_summarised(self):
        import json

        for _ in range(4):
            for side in ("left", "right"):
                self.driver.apply_each(
                    side, {robot: profile_for(robot).realtime_neutral(side)
                           for robot in PAPER_ROBOTS})
            self.driver.step(6)
        gap = self.driver.forearm_gap_mm()
        self.assertEqual(set(gap), set(PAPER_ROBOTS))
        for robot, value in sorted(gap.items()):
            with self.subTest(robot=robot):
                self.assertGreater(value, 0.0)
        summary = self.driver.forearm_gap_summary()
        json.dumps(summary)
        self.assertGreaterEqual(summary["frames"], 4)
        self.assertEqual(set(summary["min_mm"]), set(PAPER_ROBOTS))
        self.assertEqual(summary["worst_mm"], min(summary["min_mm"].values()))
        posture = self.driver.posture_summary()
        self.assertEqual(set(posture), set(PAPER_ROBOTS))
        self.assertEqual(set(posture["h1_2"]), {"left", "right"})
        json.dumps(self.driver.summary())      # the report payload stays JSON-able


@unittest.skipUnless(ASSETS_AVAILABLE, "assets/robots/from_teleopbench is missing (not in git)")
@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet is not installed")
class ArmTargetSlewTests(unittest.TestCase):
    """The slew cap has to be real, and the working point has to stay reachable apart.

    ``PyBulletArmIK.solve`` starts from the body's current angles, so a target that
    arrives in one large step can land the two-pass solver in a bad branch and it keeps
    solving into that branch every frame.  Measured on this machine 2026-10-13 (gravity
    -9.81, 60 iterations / 1e-4, the app's own settings):

        one command then hold   h1_2 47.8 mm, forearms  6.0 mm apart (hands on the midline)
                                gr1_t2 31.6 mm, g1 68.7 mm -- all three over the 30 mm gate
        20 mm/frame walk-in     h1_2  8.9 mm, forearms 83.5 mm apart
                                gr1_t2 6.7 mm / 200.0 mm, g1 8.9 mm / 186.8 mm
        replayed live session   h1_2 steady 45 mm -> 9 mm (frames 105-165)

    The fixture commands every frame like the app does, so here the *cap itself* is the
    regression (``test_a_single_call_moves_a_target_at_most_the_cap``); the second test
    then pins the property that matters to the eye: both wrists on their own side of the
    robot, on target, with the forearms clearly apart.
    """

    SLEW_M = 0.02

    @classmethod
    def setUpClass(cls):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

        # Same scene the app builds (hands on, headless), fresh so the arms start where
        # a live run starts: hanging at their load angles.  The IK settings are the
        # app's own defaults (realtime_teleop --arm-ik-iterations/--arm-ik-residual-
        # threshold): the redundant 7 DoF chain settles into a different null-space
        # pose at 200/1e-5, which moves the measured forearm gap without failing.
        cls.scene = ThreeRobotHandScene(sides=SIDES, gui=False)
        cls.driver = ArmSceneDriver(cls.scene, sides=SIDES, ik_iterations=60,
                                    residual_threshold=1e-4, slew_m=cls.SLEW_M)
        cls.bodies = {robot["type"]: robot["body"] for robot in cls.scene.robots}

    @classmethod
    def tearDownClass(cls):
        cls.driver = None
        cls.scene.close()

    def wrist_in_base(self, adapter, side: str):
        """Wrist position in that robot's own base frame (where the profiles live)."""
        p, cid = self.driver.p, self.driver.cid
        body = adapter.body
        world = p.getLinkState(body, adapter.ee[side][0],
                               physicsClientId=cid)[4]
        base, orientation = p.getBasePositionAndOrientation(body, physicsClientId=cid)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation),
                              dtype=np.float64).reshape(3, 3)
        return rotation.T @ (np.asarray(world, dtype=np.float64) - np.asarray(base))

    def test_a_single_call_moves_a_target_at_most_the_cap(self):
        targets = {robot: profile_for(robot).realtime_neutral("left")
                   for robot in PAPER_ROBOTS}
        before = {robot: self.wrist_in_base(self.driver.ik[robot], "left")
                  for robot in PAPER_ROBOTS}
        self.driver.apply_each("left", targets, source="base")
        for robot in PAPER_ROBOTS:
            commanded = np.asarray(self.driver.commanded[(robot, "left")])
            wanted = np.asarray(targets[robot], dtype=np.float64)
            towards = wanted - before[robot]
            with self.subTest(robot=robot):
                # The first call starts from the wrist itself and takes one capped step.
                self.assertLessEqual(float(np.linalg.norm(commanded - before[robot])),
                                     self.SLEW_M + 1e-9)
                # A step towards the working point, never past it or away from it.
                self.assertGreater(float((commanded - before[robot]) @ towards), 0.0)
                self.assertLessEqual(float(np.linalg.norm(commanded - wanted)),
                                     float(np.linalg.norm(towards)))
        self.assertGreater(self.driver.slew_limited, 0)
        self.assertEqual(self.driver.describe()["target_slew_mm"], 20.0)

    def test_the_speed_form_survives_a_slow_camera_loop(self):
        """2026-10-13：斜坡改成速率口径（mm/s），每帧按实测间隔换算。

        为什么：现场摄像头循环只有 8-12 fps，写死的 20 mm/帧 = 0.20 m/s，而录制
        （``outputs/tmp_realtime/live_long_frames.npz``，10.05 fps）里操作者手的每帧位移是
        中位 37 mm / p90 147 mm —— 斜坡帧帧生效，机器人被按 13% 的手速拖着走，正弦实测
        横向摆幅传递比只有 0.43（换成 40 mm/帧后 0.80）。数字见 docs/multi-robot-arm.md 8.8。
        """
        from teleoperation.simulation.arm_scene_driver import (
            SLEW_MAX_STEP_M,
            slew_per_frame,
        )

        # 600 mm/s 在 30 fps 下正好是旧口径的 20 mm/帧；10 fps 下按帧换算成 60 mm/帧、
        # 被天花板压到 40 mm/帧（实测拐点：再快横摆不再变大而残差变大）。
        self.assertAlmostEqual(slew_per_frame(0.6, 1.0 / 30.0) * 1000.0, 20.0, places=9)
        self.assertAlmostEqual(slew_per_frame(0.6, 1.0 / 10.0) * 1000.0, 40.0, places=9)
        self.assertAlmostEqual(slew_per_frame(0.6, 0.5) * 1000.0,
                               SLEW_MAX_STEP_M * 1000.0, places=9)
        self.assertEqual(slew_per_frame(0.0, 0.1), 0.0)     # 0 = 关掉斜坡
        self.assertEqual(slew_per_frame(0.6, 0.0), 0.0)     # 还没有帧间隔
        self.driver.slew_m_per_s = 0.6
        self.assertAlmostEqual(self.driver.update_slew(1.0 / 10.0), 0.04, places=9)
        self.assertAlmostEqual(self.driver.update_slew(1.0 / 30.0), 0.02, places=9)
        described = self.driver.describe()
        self.assertEqual(described["target_slew_mm_per_s"], 600.0)
        self.assertEqual(described["target_slew_ceiling_mm"], 40.0)
        # 回归（端到端跑 `--arm-slew-mm 20` 时抓到）：速率是 0 说明用的是**固定 mm/帧**
        # 口径，update_slew 不能把 slew_m 改写成 0（那等于悄悄关掉斜坡）。
        self.driver.slew_m_per_s = 0.0
        self.driver.slew_m = self.SLEW_M
        self.assertAlmostEqual(self.driver.update_slew(1.0 / 10.0), self.SLEW_M, places=9)
        self.assertAlmostEqual(self.driver.update_slew(0.0), self.SLEW_M, places=9)
        self.driver.slew_m_per_s = 0.0
        self.driver.slew_m = self.SLEW_M

    def cold_start(self):
        """Put every joint back at its load angle: the state a fresh app starts in.

        ``PyBulletArmIK.solve`` seeds from the body's current angles, so this is what
        makes the measurement below deterministic instead of dependent on which test
        ran first.  Clearing ``commanded`` re-seeds the ease-in from the wrist itself,
        which is what the first frame of a live run does.
        """
        for adapter in self.driver.ik.values():
            for index in adapter.indices:
                self.driver.p.resetJointState(adapter.body, index,
                                              float(adapter.hold[index]),
                                              physicsClientId=self.driver.cid)
        self.driver.commanded.clear()

    def test_the_working_point_walks_in_and_the_two_wrists_stay_apart(self):
        self.cold_start()
        wanted = {robot: {side: profile_for(robot).realtime_neutral(side)
                          for side in SIDES} for robot in PAPER_ROBOTS}
        for _ in range(60):                 # one command per frame, like the app loop
            for side in SIDES:
                self.driver.apply_each(side, {robot: wanted[robot][side]
                                              for robot in PAPER_ROBOTS}, source="base")
            self.driver.step(1)
        gaps = self.driver.forearm_gap_mm()
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            with self.subTest(robot=robot):
                actual = {}
                for side in SIDES:
                    target = np.asarray(wanted[robot][side], dtype=np.float64)
                    points = self.wrist_in_base(adapter, side)
                    actual[side] = points
                    # The documented gate is 30 mm (arm_profiles.max_residual_mm).
                    self.assertLess(float(np.linalg.norm(points - target)) * 1000.0, 30.0,
                                    f"{robot}/{side}: {points} vs {target}")
                    # "left" must stay on that robot's own +y side: the crossing check.
                    self.assertGreater(points[1] * (1.0 if side == "left" else -1.0), 0.0)
                # Hands on the centre line measured +-3 mm (6 mm apart) before the ease-in.
                self.assertGreater(abs(actual["left"][1] - actual["right"][1]), 0.06)
                # h1_2 measures 83.5 mm of forearm gap here (6.0 mm before the ease-in);
                # the curve is in E:\\cache\\tmp\\probe_slew_curve.py.
                self.assertGreater(gaps[robot], 60.0, gaps)



class FieldArmDefaultsTests(unittest.TestCase):
    """2026-10-14 field defaults: the entry point the operator runs ships the fix.

    ``PyBulletArmIK`` / ``ArmSceneDriver`` keep the historical defaults
    (``lock_others=False``, anchor key ``"shoulder"`` still reachable) so the
    ``devtools/verify_*.py`` numbers and the tests above keep matching old records;
    what the field actually launches (``realtime_teleop.py``, and through it
    ``devtools/run_realtime_arm_windows.py``) must default to the rigid anchor plus the
    arm-only solve. See docs/multi-robot-arm.md 8.13.4.
    """

    def test_realtime_cli_defaults_to_the_rigid_anchor_and_the_arm_only_solve(self):
        from teleoperation.simulation import arm_ik
        from teleoperation.simulation.realtime_teleop import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.arm_anchor, "upper")
        self.assertEqual(args.arm_ik_scope, "arm")
        self.assertEqual(args.arm_elbow_recovery_margin_deg,
                         arm_ik.ELBOW_RECOVERY_MARGIN_DEG)
        self.assertEqual(args.arm_elbow_recovery_frames,
                         arm_ik.ELBOW_RECOVERY_COOLDOWN_FRAMES)
        self.assertGreater(args.arm_elbow_recovery_deg, 0.0)

    def test_both_switches_still_reach_the_historical_behaviour(self):
        from teleoperation.simulation.realtime_teleop import build_parser

        args = build_parser().parse_args(["--arm-ik-scope", "body",
                                          "--arm-anchor", "shoulder",
                                          "--arm-elbow-recovery-margin-deg", "0",
                                          "--arm-elbow-recovery-frames", "0"])
        self.assertEqual(args.arm_ik_scope, "body")
        self.assertEqual(args.arm_anchor, "shoulder")
        self.assertEqual(args.arm_elbow_recovery_margin_deg, 0.0)
        self.assertEqual(args.arm_elbow_recovery_frames, 0)


@unittest.skipUnless(ASSETS_AVAILABLE, "assets/robots/from_teleopbench is missing (not in git)")
@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet is not installed")
class RigidShoulderAnchorTests(unittest.TestCase):
    """The shoulder end of the upper arm must be rigid to the elbow (8.13.2).

    ``arm_points()["shoulder"]`` is a name match (the first link whose name contains
    "shoulder"): on h1_2 / g1 that is the pitch link, on gr1_t2 the upper-arm pitch link.
    Turning the arm's roll joint moves it 48.5-101.7 mm across the six sides, so a recipe
    built on it ("shoulder + L_u * direction") asks for an elbow target off the arm's
    reachable shell -- the GR1-T2 reversed-forearm root cause. The parent link of the
    elbow joint is rigid instead: 0.00 mm.
    """

    @classmethod
    def setUpClass(cls):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

        cls.scene = ThreeRobotHandScene(sides=("right",), gui=False)
        cls.driver = ArmSceneDriver(cls.scene, sides=("right",))
        cls.driver.step(5)

    @classmethod
    def tearDownClass(cls):
        cls.driver = None
        cls.scene.close()

    @staticmethod
    def _span(points, end, start):
        """Distance (m) between two ``arm_points()`` entries."""
        return float(np.linalg.norm(np.asarray(points[end], dtype=np.float64)
                                    - np.asarray(points[start], dtype=np.float64)))

    def test_default_anchor_key_is_upper_and_bad_keys_are_rejected(self):
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

        self.assertEqual(self.driver.shoulder_anchor_key, "upper")
        with self.assertRaisesRegex(ValueError, "shoulder_anchor_key"):
            ArmSceneDriver(self.scene, sides=("right",), shoulder_anchor_key="elbow")

    def test_parent_link_anchor_is_rigid_while_the_name_match_drifts(self):
        for robot in PAPER_ROBOTS:
            adapter = self.driver.ik[robot]
            roll = next(index for index, name in adapter.arm["right"]
                        if "roll" in name.lower()
                        and ("shoulder" in name.lower() or "upper_arm" in name.lower()))
            with self.subTest(robot=robot):
                start = float(pybullet.getJointState(adapter.body, roll,
                                                     physicsClientId=adapter.cid)[0])
                before = adapter.arm_points("right")
                rigid_before = self._span(before, "elbow", "upper")
                named_before = self._span(before, "elbow", "shoulder")
                # 1.4 rad = 80 deg of roll: measured drift 39-81 mm, see 8.13.2.
                pybullet.resetJointState(adapter.body, roll, start + 1.4,
                                         physicsClientId=adapter.cid)
                after = adapter.arm_points("right")
                # The rigid span may not move at all (measured 0.00 mm; 0.01 mm allowed).
                self.assertAlmostEqual(self._span(after, "elbow", "upper"),
                                       rigid_before, delta=1e-5)
                # The name-matched one does move: that is the whole bug.
                self.assertGreater(abs(self._span(after, "elbow", "shoulder")
                                       - named_before), 0.02)
                pybullet.resetJointState(adapter.body, roll, start,
                                         physicsClientId=adapter.cid)

