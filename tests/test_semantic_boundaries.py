"""New semantic boundaries and rejected actions; run during manual acceptance."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from teleoperation.contracts.arm import ArmIKResult, ArmTargetPose
from teleoperation.contracts.hand import HandWindow, L21HandAngles, L21HandDOFs
from teleoperation.contracts.observations import RawHandFrame, RawTeleopObservation, UpperBodyObservation
from teleoperation.inputs.replay import NpyReplayInput
from teleoperation.inputs.visionpro import VisionProInput
from teleoperation.robots.action_mapping import H1ActionAdapter, validate_native_action, pack_tron2a_l21_action
from teleoperation.robots.partial import PartialLimbVector, pack_partial_limb_vector
from teleoperation.simulation.environment import SimulationEnv


class SemanticBoundaryTests(unittest.TestCase):
    def test_hand_values_preserve_precision_and_drop_only_the_placeholder(self):
        values = np.arange(18, dtype=np.float64) / 7
        angles = L21HandAngles(values)
        self.assertIs(angles.values, values)
        dofs = angles.native_mapping_dofs()
        self.assertIsInstance(dofs, L21HandDOFs)
        self.assertEqual(dofs.values.dtype, np.float64)
        np.testing.assert_array_equal(dofs.values, values[1:])

    def test_raw_combination_rejects_a_model_window(self):
        window = HandWindow({"left": np.ones((3, 25, 3)), "right": None})
        body = UpperBodyObservation({"left": None, "right": None}, {"left": False, "right": False})
        with self.assertRaises(TypeError): RawTeleopObservation(window, body)

    def test_arm_stages_have_distinct_shapes_and_unknown_units_remain_unknown(self):
        body = UpperBodyObservation({"left": np.ones((3, 3)), "right": None}, {"left": True, "right": False})
        self.assertEqual(body.units, "unknown")
        self.assertEqual(body.coordinate_frame, "unknown")
        with self.assertRaises(ValueError): ArmTargetPose("left", body.arms["left"])
        with self.assertRaises(ValueError): UpperBodyObservation({"left": np.zeros(7), "right": None}, body.valid)

    def test_failed_ik_can_preserve_previous_q_and_infinite_errors(self):
        previous = np.arange(7, dtype=np.float32)
        result = ArmIKResult(False, previous, float("inf"), float("inf"), side="left")
        self.assertIs(result.q, previous)
        self.assertFalse(result.success)
        self.assertTrue(np.isinf(result.position_error))

    def test_npy_returns_the_first_raw_frame_then_eof(self):
        points = np.arange(63, dtype=np.float64).reshape(21, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npy"
            np.save(path, np.array([{"left_hand": points, "right_hand": None, "frame_id": 7}], dtype=object))
            source = NpyReplayInput(path)
            frame = source.next_observation()
            self.assertIsInstance(frame, RawHandFrame)
            np.testing.assert_array_equal(frame.hands["left"], points)
            self.assertEqual(frame.hands["left"].dtype, np.float64)
            self.assertEqual(frame.metadata["frame_id"], 7)
            with self.assertRaises(StopIteration): source.next_observation()
            source.close(); source.close()

    def test_visionpro_returns_unchanged_matrices_and_none_for_no_new_sample(self):
        transforms = np.arange(400, dtype=np.float64).reshape(25, 4, 4)
        stream = Mock()
        stream.get_latest.side_effect = [{"left_fingers": transforms}, {}]
        source = VisionProInput("unused", streamer=stream)
        frame = source.next_observation()
        self.assertEqual(frame.geometry_encoding["left"], "transforms25")
        np.testing.assert_array_equal(frame.hands["left"], transforms)
        self.assertIsNone(source.next_observation())
        source.close(); source.close()
        stream.close.assert_called_once()

    def test_partial_keeps_values_and_cannot_be_encoded_even_if_it_matches_a_dimension(self):
        partial = pack_partial_limb_vector(np.arange(7), None, np.arange(7), None)
        np.testing.assert_array_equal(partial.values, np.r_[np.arange(7), np.arange(7)].astype(np.float32))
        self.assertEqual(partial.limbs, ("left_arm", "right_arm"))
        self.assertFalse(hasattr(partial, "__array__"))
        coincidental = PartialLimbVector(np.zeros(38), ("left_arm",))
        with self.assertRaises(TypeError): validate_native_action(coincidental, 38)
        adapter = H1ActionAdapter([f"native_joint_{i}" for i in range(38)])
        with self.assertRaises(TypeError): adapter.encode(coincidental)

    def test_strict_tron_pack_requires_all_four_limbs(self):
        action = pack_tron2a_l21_action(np.ones(7), np.ones(17) * 2, np.ones(7) * 3, np.ones(17) * 4)
        np.testing.assert_array_equal(action, np.r_[np.ones(7), np.ones(17) * 2, np.ones(7) * 3, np.ones(17) * 4].astype(np.float32))
        for left_arm in (None, np.zeros(8)):
            with self.assertRaises(ValueError): pack_tron2a_l21_action(left_arm, np.zeros(17), np.zeros(7), np.zeros(17))

    def test_invalid_actions_fail_before_joint_application_or_physics(self):
        env = SimulationEnv.__new__(SimulationEnv)
        env.action_dim = 38
        env.task = Mock()
        for action in (PartialLimbVector(np.zeros(38), ("left_arm",)), np.zeros(37), np.full(38, np.nan)):
            with self.subTest(action=type(action).__name__), patch("teleoperation.simulation.environment.p.stepSimulation") as step:
                with self.assertRaises((TypeError, ValueError)): env.step(action)
                env.task.apply_action.assert_not_called()
                step.assert_not_called()
