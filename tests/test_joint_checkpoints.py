"""Target identity and preprocessing are part of neural weight compatibility."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from teleoperation.paths import training_run_directory
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.predictor import build_joint_model
from teleoperation.retargeting.hand.checkpoint import (
    save_joint_checkpoint,
    shared_joint_checkpoint_payload,
    load_shared_joint_checkpoint,
)


SMALL = dict(embed_dim_ratio=8, spatial_depth=1, temporal_depth=1, num_heads=2, drop_path_rate=0.)


class JointCheckpointTests(unittest.TestCase):
    def test_dynamic_heads_and_strict_identity_reload(self):
        torch.set_num_threads(1)
        for name, dof in (("g1", 7), ("l21", 17)):
            spec = load_hand_spec(name, "right")
            model = build_joint_model(spec, SMALL).eval()
            values = torch.zeros((2, 3, 25, 3))
            self.assertEqual(model(values).shape, (2, dof))
            payload = shared_joint_checkpoint_payload(model, spec)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"model.pth"
                save_joint_checkpoint(path, payload)
                loaded, metadata = load_shared_joint_checkpoint(path, spec, "cpu")
                torch.testing.assert_close(model(values), loaded(values), atol=0, rtol=0)
                self.assertEqual(metadata['side'], 'shared')
                for changes in ({"hand_model":"other"}, {"spec_hash":"a"*64},
                                {"joint_names":list(reversed(spec.independent_joint_names))},
                                {"input_profile":"old_metres"},
                                {"reference_geometry":{"root_link":"different_root"}}):
                    save_joint_checkpoint(path, {**payload, **changes})
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        load_shared_joint_checkpoint(path, spec, "cpu")

    def test_new_twenty_one_dof_network_needs_no_l21_placeholder(self):
        spec = load_hand_spec("l21", "right")
        # This only checks the network head; it is not a real new hand asset.
        names = tuple(f"joint_{i}" for i in range(21))
        spec = replace(spec, hand_model="synthetic21", joint_names=names, independent_joint_names=names,
                       joint_limits=((-1.,1.),)*21, joint_velocities=(1.,)*21, mimic_rules=(), palm_frames=())
        model = build_joint_model(spec, SMALL).eval()
        self.assertEqual(model(torch.zeros((1,3,25,3))).shape, (1,21))

    def test_paths_are_target_specific_and_reject_traversal(self):
        root = Path("outputs/models")
        path = training_run_directory(root, "l21", "right", "camera_palm_length_window3_v1", "run01")
        self.assertEqual(path, root/"l21/right/nn/camera_palm_length_window3_v1/run01")
        shared = training_run_directory(root, "l21", "shared", "camera_palm_length_window3_v1", "run01")
        self.assertEqual(shared, root/"l21/shared/nn/camera_palm_length_window3_v1/run01")
        for run in ("../escape", "a/b", "a\\b", "", ".."):
            with self.assertRaises(ValueError):
                training_run_directory(root, "l21", "right", "camera_palm_length_window3_v1", run)

    def test_shared_checkpoint_uses_right_canonical_spec_and_loads_once_for_both_sides(self):
        spec = load_hand_spec("l21", "right")
        model = build_joint_model(spec, SMALL).eval()
        payload = shared_joint_checkpoint_payload(model, spec)
        self.assertEqual(payload["side"], "shared")
        self.assertEqual(payload["hand_sides"], ["left", "right"])
        self.assertEqual(payload["training_robot"], "right")
        self.assertEqual(payload["angle_convention"], "l21_canonical_right_angles_v1")
        self.assertEqual(payload["output_coordinates"], "canonical_right")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pth"
            save_joint_checkpoint(path, payload)
            loaded, metadata = load_shared_joint_checkpoint(path, spec, "cpu")
            values = torch.zeros((2, 3, 25, 3))
            torch.testing.assert_close(model(values), loaded(values), atol=0, rtol=0)
            self.assertEqual(metadata["side"], "shared")
            with self.assertRaisesRegex(ValueError, "(?i)shared"):
                load_shared_joint_checkpoint(
                    path,
                    load_hand_spec("l21", "left"),
                    "cpu",
                )

    def test_checkpoint_cannot_override_declared_effective_output_limits(self):
        spec=load_hand_spec('g1','right')
        payload=shared_joint_checkpoint_payload(build_joint_model(spec,SMALL),spec)
        payload['state_dict']['angle_clamper.angle_min']=torch.full((spec.dof,),-100.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.pth'
            save_joint_checkpoint(path,payload)
            with self.assertRaisesRegex(ValueError,'limits'):
                load_shared_joint_checkpoint(path,spec,'cpu')


if __name__ == "__main__":
    unittest.main()
