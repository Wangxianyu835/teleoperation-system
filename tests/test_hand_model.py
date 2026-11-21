"""Independent physical FK and gradient checks for target-specific training."""
import contextlib
import io
import unittest

import numpy as np
import torch

from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.hand_model import DifferentiableHandModel, build_hand_targets
from teleoperation.retargeting.hand.normalization import normalize_hand_window
from tests.test_palm_length_normalization import metric_window


class HandModelTests(unittest.TestCase):
    def test_fk_matches_pybullet_link_frames_at_zero_and_nonzero_positions(self):
        import pybullet as p
        client = p.connect(p.DIRECT)
        self.addCleanup(lambda: p.disconnect(client))
        rng = np.random.default_rng(61)
        for name in ("l21", "h1_2", "gr1_t2", "g1"):
            for side in ("left", "right"):
                with self.subTest(name=name, side=side):
                    spec = load_hand_spec(name, side)
                    fk = DifferentiableHandModel(spec)
                    body = p.loadURDF(str(spec.urdf_path), useFixedBase=True, physicsClientId=client)
                    info = [p.getJointInfo(body, i, physicsClientId=client) for i in range(p.getNumJoints(body, physicsClientId=client))]
                    joints = {item[1].decode(): i for i, item in enumerate(info)}
                    links = {item[12].decode(): i for i, item in enumerate(info)}
                    def pose(frame):
                        if frame in links:
                            state = p.getLinkState(body, links[frame], computeForwardKinematics=True, physicsClientId=client)
                            return state[4], state[5]
                        self.assertEqual(frame, p.getBodyInfo(body, physicsClientId=client)[0].decode())
                        com = p.getBasePositionAndOrientation(body, physicsClientId=client)
                        dynamics = p.getDynamicsInfo(body, -1, physicsClientId=client)
                        return p.multiplyTransforms(*com, *p.invertTransform(dynamics[3], dynamics[4]))
                    random_q = spec.bounds[:, 0] + rng.uniform(.1, .9, spec.dof) * np.diff(spec.bounds, axis=1)[:, 0]
                    for q in (spec.q_reference, random_q):
                        for joint, value in spec.expand(q).items():
                            p.resetJointState(body, joints[joint], value, physicsClientId=client)
                        inverse = p.invertTransform(*pose(spec.root_link))
                        expected = np.array([p.multiplyTransforms(*inverse, *pose(tip))[0] for tip in spec.tip_frames])
                        actual = fk.positions(torch.tensor(q[None], dtype=torch.float64), spec.tip_frames).detach().numpy()[0]
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
                    p.removeBody(body, physicsClientId=client)

    def test_autograd_includes_mimic_chains(self):
        for name in ("l21", "h1_2"):
            spec = load_hand_spec(name, "right")
            fk = DifferentiableHandModel(spec)
            q = torch.tensor(spec.bounds.mean(1)[None], dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda values: fk.positions(values, spec.tip_frames),
                                                    (q,), fast_mode=True, atol=1e-5, rtol=1e-3))

    def test_shared_targets_remove_human_scale_and_use_each_robot_reference_length(self):
        for name in ("l21", "g1"):
            for side in ("left", "right"):
                spec = load_hand_spec(name, side)
                one = build_hand_targets(normalize_hand_window(metric_window()), spec)
                two = build_hand_targets(normalize_hand_window(metric_window((1.3,) * 3)), spec)
                np.testing.assert_allclose(one.vectors, two.vectors, atol=1e-6)
                np.testing.assert_allclose(one.positions_native_m / spec.reference_palm_length_m, one.vectors, atol=1e-12)
                back = one.vectors @ spec.native_to_canonical
                np.testing.assert_allclose(back[0], [.5, .25, .75], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
