"""Target-specific joint spaces shared by every retargeting method."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from teleoperation.contracts.hand_control import HandJointResult
from teleoperation.robots.hand_spec import HandSpec, load_hand_spec


def coupled_spec(folder):
    path = Path(folder) / "hand.urdf"
    path.write_text('<robot name="fixture"><link name="palm"/></robot>', encoding="utf-8")
    return HandSpec(
        hand_model="fixture", side="right", urdf_path=path, root_link="palm",
        joint_names=("driver", "follower"), independent_joint_names=("driver",),
        joint_limits=((-2., 2.), (-1., 1.)), joint_velocities=(3., .5),
        mimic_rules=(("follower", "driver", -2., .2),),
    )


class HandSpecificationTests(unittest.TestCase):
    def test_registered_reference_geometry_uses_real_hand_frames(self):
        for model in ("l21", "h1_2", "gr1_t2", "g1"):
            for side in ("left", "right"):
                with self.subTest(model=model, side=side):
                    spec = load_hand_spec(model, side)
                    self.assertGreater(spec.reference_palm_length_m, 1e-6)
                    np.testing.assert_allclose(spec.native_to_canonical.T @ spec.native_to_canonical,
                                               np.eye(3), atol=1e-6)
                    self.assertEqual(spec.q_reference.shape, (spec.dof,))
                    self.assertTrue(np.all(spec.q_reference >= spec.bounds[:, 0]))
                    self.assertTrue(np.all(spec.q_reference <= spec.bounds[:, 1]))

    def test_missing_reference_frame_and_degenerate_palm_are_rejected(self):
        spec = load_hand_spec("l21", "right")
        for frames in (("hand_base_link", "missing", "index_tip"),
                       ("hand_base_link", "index_tip", "index_tip")):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                replace(spec, palm_frames=frames)

    def test_registered_hands_expose_independent_coordinates_and_urdf_speeds(self):
        for model, dof, count in (("l21", 17, 17), ("h1_2", 6, 12),
                                  ("gr1_t2", 6, 11), ("g1", 7, 7)):
            for side in ("left", "right"):
                with self.subTest(model=model, side=side):
                    spec = load_hand_spec(model, side)
                    self.assertEqual(spec.dof, dof)
                    self.assertEqual(len(spec.joint_names), count)
                    self.assertEqual(len(spec.human_tip_indices), 3 if model == "g1" else 5)
                    self.assertEqual(set(spec.expand(spec.bounds.mean(1))), set(spec.joint_names))
                    if model == "l21":
                        np.testing.assert_array_equal(spec.velocity_limits, np.ones(17))
                        self.assertEqual(spec.independent_joint_names[0], "index_mcp_roll")

    def test_negative_mimic_pulls_back_bounds_and_follower_velocity(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = coupled_spec(folder)
            np.testing.assert_allclose(spec.bounds, [[-.4, .6]], atol=1e-12)
            np.testing.assert_allclose(spec.velocity_limits, [.25], atol=1e-12)
            self.assertAlmostEqual(spec.expand(np.array([.3]))["follower"], -.4)
            with self.assertRaises(ValueError):
                spec.expand(np.array([.7]))

    def test_cyclic_mimic_duplicate_joints_and_infeasible_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = coupled_spec(folder)
            bad = (
                dict(joint_names=("driver", "driver")),
                dict(independent_joint_names=("missing",)),
                dict(mimic_rules=(("follower", "follower", 1., 0.),)),
                dict(mimic_rules=(("follower", "missing", 1., 0.),)),
                dict(mimic_rules=(("follower", "driver", 0., 2.),)),
                dict(joint_limits=((1., 2.), (-1., 1.))),
                dict(joint_velocities=(3., float("nan"))),
            )
            for changes in bad:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    replace(spec, **changes)

    def test_spec_hash_identifies_semantics_and_asset_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = coupled_spec(folder)
            self.assertEqual(spec.spec_hash, coupled_spec(folder).spec_hash)
            self.assertNotEqual(spec.spec_hash, replace(spec, side="left").spec_hash)
            self.assertNotEqual(spec.spec_hash, replace(spec, joint_velocities=(2., .5)).spec_hash)
            before = spec.spec_hash
            spec.urdf_path.write_text('<robot name="changed"><link name="palm"/></robot>', encoding="utf-8")
            self.assertNotEqual(before, replace(spec).spec_hash)

    def test_twenty_one_independent_coordinates_are_not_assumed_to_be_l21(self):
        with tempfile.TemporaryDirectory() as folder:
            base = coupled_spec(folder)
            names = tuple(f"joint_{i}" for i in range(21))
            spec = replace(base, hand_model="synthetic21", joint_names=names,
                           independent_joint_names=names, joint_limits=((-1., 1.),) * 21,
                           joint_velocities=(1.,) * 21, mimic_rules=())
            self.assertEqual(spec.dof, 21)
            self.assertEqual(tuple(spec.expand(np.zeros(21))), names)
            for values in (np.zeros(18), np.full(21, np.nan), np.full(21, 2.)):
                with self.assertRaises(ValueError):
                    spec.expand(values)

    def test_unknown_target_and_side_fail_without_creating_robot_resources(self):
        for model, side in (("unknown", "left"), ("l21", "both")):
            with self.assertRaises(ValueError):
                load_hand_spec(model, side)


class HandJointResultTests(unittest.TestCase):
    def test_result_keeps_ordered_independent_coordinates_and_owns_its_values(self):
        values = np.array([.1, .2])
        result = HandJointResult("fixture", "right", "a" * 64,
                                 ("joint_b", "joint_a"), values, 1.)
        values[:] = 0.
        np.testing.assert_array_equal(result.positions, [.1, .2])
        self.assertEqual(result.joint_names, ("joint_b", "joint_a"))
        with self.assertRaises(ValueError):
            result.positions[0] = 2.

    def test_invalid_result_requires_no_placeholder_or_guessed_angles(self):
        result = HandJointResult("fixture", "left", "b" * 64,
                                 ("joint",), None, 2., valid=False, error="missing input")
        self.assertIsNone(result.positions)
        self.assertFalse(result.valid)
        with self.assertRaises(ValueError):
            replace(result, valid=True)
        for changes in (dict(positions=np.zeros(2)), dict(positions=np.array([np.nan])),
                        dict(timestamp=float("inf")), dict(joint_names=("a", "a")),
                        dict(side="both"), dict(spec_hash="")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(result, **changes)


if __name__ == "__main__":
    unittest.main()
