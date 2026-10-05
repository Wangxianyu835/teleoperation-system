import math
import unittest

import torch
import torch.nn as nn

from model.losses import thumb_loss2, tip_distance_loss
from retargeting.config import ROBOT_JOINTS, SOURCE_JOINTS


class TipDistanceLossTests(unittest.TestCase):
    def _positions(self):
        tips = torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [0.03, 0.01, 0.00],
                [0.02, 0.02, 0.01],
                [0.01, 0.03, 0.01],
                [-0.01, 0.02, 0.00],
            ],
            dtype=torch.float64,
        )
        robot = torch.zeros((1, 23, 3), dtype=tips.dtype)
        source = torch.zeros((1, 25, 3), dtype=tips.dtype)
        robot[:, ROBOT_JOINTS["TIP_dic"]] = tips
        source[:, SOURCE_JOINTS["TIP_dic"]] = tips * 1.2
        return robot, source

    def _loss(self, robot, source):
        return tip_distance_loss(
            robot, source, nn.MSELoss(), ROBOT_JOINTS, SOURCE_JOINTS
        )

    def test_repeating_one_sample_four_times_preserves_loss(self):
        robot, source = self._positions()
        single = self._loss(robot, source)
        repeated = self._loss(robot.repeat(4, 1, 1), source.repeat(4, 1, 1))

        self.assertGreater(single.item(), 0.0)
        torch.testing.assert_close(repeated, single)

    def test_repeating_a_mixed_sample_distribution_preserves_loss(self):
        robot, source = self._positions()
        robot = torch.cat((robot, robot * 1.1))
        source = torch.cat((source, source * 1.4))
        original = self._loss(robot, source)
        repeated = self._loss(robot.repeat(4, 1, 1), source.repeat(4, 1, 1))

        torch.testing.assert_close(repeated, original)

    def test_repeated_samples_have_finite_and_equivalent_accumulated_gradients(self):
        gradients = []
        for copies in (1, 4):
            robot, source = self._positions()
            robot.requires_grad_()
            source.requires_grad_()
            loss = self._loss(
                robot.repeat(copies, 1, 1), source.repeat(copies, 1, 1)
            )
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()

            for points in (robot, source):
                self.assertIsNotNone(points.grad)
                self.assertTrue(torch.isfinite(points.grad).all().item())
                self.assertGreater(points.grad.abs().sum().item(), 0.0)
            gradients.append((robot.grad, source.grad))

        for single, repeated in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(repeated, single)


class ThumbAngleLossTests(unittest.TestCase):
    def _points(self, angle, robot, dtype=torch.float64, length=1.0):
        points = torch.zeros((1, 23 if robot else 25, 3), dtype=dtype)
        first = 15 if robot else 2
        points[0, first + 1, 0] = length
        # Make the two endpoint cases exactly collinear, including sin(pi)=0.
        sine = 0.0 if angle in (0.0, math.pi) else math.sin(angle)
        points[0, first + 2] = torch.tensor(
            [length * (1.0 + math.cos(angle)), length * sine, 0.0],
            dtype=dtype,
        )
        return points

    def _loss(self, robot, source):
        return thumb_loss2(robot, source, nn.MSELoss(), ROBOT_JOINTS, SOURCE_JOINTS)

    def _assert_finite_backward(self, robot, source, expected):
        robot.requires_grad_()
        source.requires_grad_()
        loss = self._loss(robot, source)
        self.assertTrue(torch.isfinite(loss).item())
        torch.testing.assert_close(loss, loss.new_tensor(expected))
        loss.backward()
        for points in (robot, source):
            self.assertIsNotNone(points.grad)
            self.assertTrue(torch.isfinite(points.grad).all().item())
        return loss

    def _check_collinear(self, angle):
        for dtype in (torch.float32, torch.float64):
            for boundary_side in ("robot", "source"):
                with self.subTest(dtype=dtype, boundary_side=boundary_side):
                    robot_angle = angle if boundary_side == "robot" else 0.6
                    source_angle = angle if boundary_side == "source" else 0.6
                    self._assert_finite_backward(
                        self._points(robot_angle, True, dtype),
                        self._points(source_angle, False, dtype),
                        (robot_angle - source_angle) ** 2,
                    )

    def test_parallel_segments_have_finite_gradients(self):
        self._check_collinear(0.0)

    def test_antiparallel_segments_have_finite_gradients(self):
        self._check_collinear(math.pi)

    def test_nearly_parallel_and_antiparallel_segments(self):
        for dtype, delta in ((torch.float32, 1e-3), (torch.float64, 1e-7)):
            for angle in (delta, math.pi - delta):
                with self.subTest(dtype=dtype, angle=angle):
                    robot = self._points(angle, True, dtype)
                    a = robot[:, 17] - robot[:, 16]
                    b = robot[:, 16] - robot[:, 15]
                    cosine = (nn.functional.normalize(a, dim=-1)
                              * nn.functional.normalize(b, dim=-1)).sum()
                    self.assertLess(abs(cosine.item()), 1.0)
                    self._assert_finite_backward(
                        robot, self._points(0.6, False, dtype), (angle - 0.6) ** 2
                    )

    def test_tiny_nonzero_segments_above_normalization_threshold(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self._assert_finite_backward(
                    self._points(0.7, True, dtype, length=1e-10),
                    self._points(1.1, False, dtype, length=1e-10),
                    (0.7 - 1.1) ** 2,
                )

    def test_segments_below_normalization_threshold_are_masked(self):
        robot = self._points(0.7, True, length=1e-14)
        source = self._points(1.1, False)
        self._assert_finite_backward(robot, source, 0.0)
        self.assertEqual(robot.grad.abs().sum().item(), 0.0)
        self.assertEqual(source.grad.abs().sum().item(), 0.0)

    def test_zero_length_segments_are_masked_on_either_side(self):
        for side in ("robot", "source"):
            for segment in (0, 1):
                with self.subTest(side=side, segment=segment):
                    robot = self._points(0.7, True)
                    source = self._points(1.1, False)
                    points = robot if side == "robot" else source
                    first = (15 if side == "robot" else 2) + segment
                    points[:, first + 1] = points[:, first]
                    self._assert_finite_backward(robot, source, 0.0)
                    self.assertEqual(robot.grad.abs().sum().item(), 0.0)
                    self.assertEqual(source.grad.abs().sum().item(), 0.0)

    def test_masked_samples_do_not_dilute_valid_sample_loss(self):
        valid_robot = self._points(0.7, True)
        valid_source = self._points(1.1, False)
        robot = torch.cat((valid_robot, torch.zeros_like(valid_robot)))
        source = torch.cat((valid_source, valid_source))
        self._assert_finite_backward(robot, source, (0.7 - 1.1) ** 2)
        self.assertEqual(robot.grad[1].abs().sum().item(), 0.0)
        self.assertEqual(source.grad[1].abs().sum().item(), 0.0)

    def test_regular_geometry_preserves_acos_loss_and_gradients(self):
        robot = self._points(0.7, True).requires_grad_()
        source = self._points(1.1, False).requires_grad_()
        reference_angles = []
        for points, first in ((robot, 15), (source, 2)):
            a = nn.functional.normalize(points[:, first + 2] - points[:, first + 1], dim=-1)
            b = nn.functional.normalize(points[:, first + 1] - points[:, first], dim=-1)
            reference_angles.append(torch.acos((a * b).sum(dim=-1)))
        reference = nn.MSELoss()(*reference_angles)
        expected_gradients = torch.autograd.grad(reference, (robot, source))
        loss = self._loss(robot, source)
        actual_gradients = torch.autograd.grad(loss, (robot, source))
        torch.testing.assert_close(loss, loss.new_tensor((0.7 - 1.1) ** 2))
        torch.testing.assert_close(loss, reference)
        for actual, expected in zip(actual_gradients, expected_gradients):
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(actual, expected)

    def test_nonfinite_geometry_is_rejected_instead_of_masked(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for side in ("robot", "source"):
                with self.subTest(value=value, side=side):
                    robot = self._points(0.7, True)
                    source = self._points(1.1, False)
                    if side == "robot":
                        robot[0, 17, 0] = value
                    else:
                        source[0, 4, 0] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self._loss(robot, source)


if __name__ == "__main__":
    unittest.main()
