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
