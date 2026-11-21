"""Native-hand pinch must bring the visible fingertips together.

Joint-angle tests alone cannot catch a wrong opposition axis: every joint can be
at a plausible angle while the mesh fingertips move farther apart. This test
drives the three real robot hands in PyBullet and measures the closest collision
shape distance.
"""

import unittest

import numpy as np


class NativeHandPinchSignalTests(unittest.TestCase):
    @staticmethod
    def _hand(thumb, index, *, curled=False):
        points = np.zeros((21, 3), dtype=np.float64)
        points[0] = [0.00, 0.50, 0.0]
        points[5] = [0.00, 0.40, 0.0]
        points[9] = [0.02, 0.38, 0.0]
        points[17] = [0.10, 0.50, 0.0]
        points[4] = thumb
        points[8] = index
        points[12] = [0.12, 0.40, 0.0]
        for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
            if curled:
                points[tip] = np.asarray(points[mcp]) + [0.01, 0.01, 0.0]
            elif tip not in (8, 12):
                points[tip] = np.asarray(points[mcp]) + [0.0, 0.10, 0.0]
            elif tip == 12:
                points[tip] = [0.12, 0.40, 0.0]
        return points

    def test_image_space_contact_recovers_a_bad_world_tip_distance(self):
        from teleoperation.robots.native_hand import pinch_signal_details

        world = self._hand([0.00, 0.55, 0.05], [0.15, 0.30, 0.00])
        image = self._hand([0.00, 0.50, 0.0], [0.00, 0.50, 0.0])
        details = pinch_signal_details(world, image)
        finger, strength, ratio = details["selected"]
        self.assertEqual(details["source"], "image")
        self.assertEqual(finger, "index")
        self.assertEqual(strength, 1.0)
        self.assertLess(ratio, 0.10)

    def test_open_image_overrides_a_false_world_fist(self):
        from teleoperation.robots.native_hand import pinch_signal_details

        world = self._hand([0.00, 0.50, 0.0], [0.00, 0.49, 0.0],
                           curled=True)
        image = self._hand([-0.08, 0.52, 0.0], [0.08, 0.42, 0.0])
        details = pinch_signal_details(world, image)
        self.assertEqual(details["world"]["finger"], "fist")
        self.assertEqual(details["source"], "image-open")
        self.assertEqual(details["selected"][:2], (None, 0.0))

    def test_contact_hold_bridges_occlusion_then_releases(self):
        from teleoperation.robots.native_hand import PinchSignalStabilizer

        tracker = PinchSignalStabilizer(hold_frames=3, decay=0.9)
        self.assertEqual(tracker.update(("index", 1.0, 0.0))[:2],
                         ("index", 1.0))
        for _ in range(3):
            self.assertEqual(tracker.update((None, 0.0, 0.8))[:2],
                             ("index", 1.0))
        self.assertEqual(tracker.update((None, 0.0, 0.8)),
                         (None, 0.0, 0.8))

    def test_image_space_cannot_turn_a_forward_pointing_hand_into_a_fist(self):
        from teleoperation.robots.native_hand import pinch_signal

        image = self._hand([-0.05, 0.50, 0.0], [0.03, 0.38, 0.0],
                           curled=True)
        finger, strength, _ratio = pinch_signal(None, image)
        self.assertNotEqual(finger, "fist")
        self.assertEqual(finger, None)
        self.assertEqual(strength, 0.0)

    def test_open_hand_approach_is_not_commanded_as_a_partial_pinch(self):
        from teleoperation.robots.native_hand import pinch_signal

        image = self._hand([0.02, 0.52, 0.0], [0.04, 0.42, 0.0])
        finger, strength, _ratio = pinch_signal(None, image)
        self.assertIsNone(finger)
        self.assertEqual(strength, 0.0)


class NativeHandPinchSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene

        cls.scene = ThreeRobotHandScene(
            sides=("right",), gui=False, steps_per_frame=8)

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    def _link(self, body, name):
        p = self.scene.p
        for index in range(p.getNumJoints(body, physicsClientId=self.scene.cid)):
            info = p.getJointInfo(body, index, physicsClientId=self.scene.cid)
            link_name = info[12].decode() if isinstance(info[12], bytes) else str(info[12])
            if link_name == name:
                return index
        self.fail("link %r was not loaded" % name)

    def test_index_pinch_makes_each_native_pair_contact(self):
        p = self.scene.p
        for _frame in range(30):
            self.scene.apply(
                "right", np.zeros(18, dtype=np.float64),
                pinch=("index", 1.0, 0.0))
            self.scene.step()

        pairs = {
            "h1_2": ("R_thumb_distal", "R_index_intermediate"),
            "gr1_t2": ("R_thumb_distal", "R_index_intermediate"),
            "g1": ("right_hand_thumb_2_link", "right_hand_index_1_link"),
        }
        for robot in self.scene.robots:
            thumb_name, finger_name = pairs[robot["type"]]
            body = robot["body"]
            points = p.getClosestPoints(
                body, body, 1.0,
                linkIndexA=self._link(body, thumb_name),
                linkIndexB=self._link(body, finger_name),
                physicsClientId=self.scene.cid)
            gaps_mm = [abs(float(point[8])) * 1000.0 for point in points]
            with self.subTest(robot=robot["type"]):
                self.assertTrue(gaps_mm, "no pinch geometry was measured")
                self.assertLess(min(gaps_mm), 5.0, gaps_mm)

    def test_fist_curls_every_finger_on_each_native_hand(self):
        for _frame in range(30):
            self.scene.apply(
                "right", np.zeros(18, dtype=np.float64),
                pinch=("fist", 1.0, 0.5))
            self.scene.step()

        expected = {
            "h1_2": ("R_index_proximal_joint", 1),
            "gr1_t2": ("R_index_proximal_joint", -1),
            "g1": ("right_hand_index_0_joint", 1),
        }
        for joint, value in self.scene.joint_values("right").items():
            robot = joint.split(".", 1)[0]
            name, sign = expected[robot]
            if joint.endswith(name):
                with self.subTest(robot=robot, joint=joint):
                    self.assertGreater(value * sign, 1.0)


if __name__ == "__main__":
    unittest.main()
