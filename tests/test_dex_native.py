"""Native dex contracts and boundaries; optional solver checks live in acceptance."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleoperation.robots.native_hand import NativeHandModel


class NativeHandModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def model(self, robot="h1_2", side="right"):
        return NativeHandModel(robot, side, Path(self.directory.name))

    def test_six_models_cover_named_joints_and_fingertips(self):
        for robot, counts in (("h1_2", (12, 6, 5)), ("gr1_t2", (11, 6, 5)), ("g1", (7, 7, 3))):
            for side in ("left", "right"):
                model = self.model(robot, side)
                self.assertEqual((len(model.joint_names), len(model.independent_names), len(model.tip_frames)), counts)
                self.assertEqual(len(set(model.joint_names)), counts[0])
                self.assertEqual(set(model.expand(model.bounds.mean(1))), set(model.joint_names))

    def test_mimic_limits_and_velocities_are_pulled_back_to_drivers(self):
        model = self.model()
        index = model.independent_names.index("R_thumb_proximal_pitch_joint")
        np.testing.assert_allclose(model.bounds[index], [0, .5])
        self.assertAlmostEqual(model.velocity_limits[index], .5 / 2.4)
        expanded = model.expand(model.bounds[:, 1])
        self.assertAlmostEqual(expanded["R_thumb_distal_joint"], 1.2)
        self.assertAlmostEqual(expanded["R_thumb_intermediate_joint"], .8)

    def test_all_coupled_boundary_commands_obey_original_limits(self):
        for robot in ("h1_2", "gr1_t2", "g1"):
            for side in ("left", "right"):
                model = self.model(robot, side)
                for values in model.bounds.T:
                    for name, value in model.expand(values).items():
                        low, high = model.limits[name]
                        self.assertGreaterEqual(value, low - 1e-12)
                        self.assertLessEqual(value, high + 1e-12)

    def test_bad_commands_are_rejected_without_implicit_clamping(self):
        model = self.model()
        for values in (np.zeros(18), np.full(6, np.nan), model.bounds[:, 1] + 1):
            with self.assertRaises(ValueError):
                model.expand(values)

    def test_configuration_is_reproducible(self):
        first, second = self.model(), self.model()
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(first.urdf_path.read_bytes(), second.urdf_path.read_bytes())
