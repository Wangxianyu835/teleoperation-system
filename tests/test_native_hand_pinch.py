"""Native-hand pinch must bring the visible fingertips together.

Joint-angle tests alone cannot catch a wrong opposition axis: every joint can be
at a plausible angle while the mesh fingertips move farther apart. This test
drives the three real robot hands in PyBullet and measures the closest collision
shape distance.
"""

import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
