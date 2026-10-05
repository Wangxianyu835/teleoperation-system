import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from retargeting.training import LOSS_NAMES, _run_epoch


class TrainingSafetyTests(unittest.TestCase):
    def _run(self, loss_factory, *, gradient_value=None, training=True):
        model = nn.Sequential(nn.Flatten(), nn.Linear(225, 18, bias=False))
        nn.init.constant_(model[1].weight, 0.01)
        self.parameters = list(model.parameters())
        self.before = [parameter.detach().clone() for parameter in self.parameters]
        self.backward_calls = []

        def gradient_hook(gradient):
            self.backward_calls.append(True)
            if gradient_value is not None:
                return torch.full_like(gradient, gradient_value)
            return gradient

        model[1].weight.register_hook(gradient_hook)
        self.real_optimizer = torch.optim.SGD(self.parameters, lr=0.01, momentum=0.9)
        self.optimizer = Mock(wraps=self.real_optimizer)
        batch = {}
        for side in ("left", "right"):
            batch[f"{side}_input"] = np.ones((1, 3, 25, 3), dtype=np.float32)
            batch[f"{side}_target"] = np.ones((1, 1, 25, 3), dtype=np.float32)
            batch[f"{side}_valid"] = np.array([side == "left"])
        generator = Mock()
        generator.next_epoch.return_value = iter([batch])

        def fake_hand_loss(predicted, *args, **kwargs):
            loss = loss_factory(predicted)
            return (loss,) * len(LOSS_NAMES)

        with patch("retargeting.training._masked_hand_loss", side_effect=fake_hand_loss):
            return _run_epoch(
                model=model, generator=generator, device="cpu",
                pos_loss=nn.MSELoss(), vec_loss=None,
                collision_losses={"left": None, "right": None},
                reg_criterion=None, hand_fks={"left": None, "right": None},
                optimizer=self.optimizer if training else None,
                model_parameters=self.parameters, writer=Mock(),
                global_step=7, logger=None, training=training,
            )

    def _assert_no_update(self):
        self.optimizer.step.assert_not_called()
        self.assertEqual(len(self.real_optimizer.state), 0)
        for parameter, before in zip(self.parameters, self.before):
            torch.testing.assert_close(parameter.detach(), before, rtol=0, atol=0)

    def test_nonfinite_total_loss_stops_before_backward_and_update(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(FloatingPointError, r"loss.*batch=1.*global_step=7"):
                    self._run(lambda predicted: predicted.mean() * value)
                self.assertEqual(self.backward_calls, [])
                self._assert_no_update()

    def test_nonfinite_gradients_stop_before_update(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, r"batch=1.*global_step=7"):
                    self._run(lambda predicted: predicted.square().mean(), gradient_value=value)
                self.assertTrue(self.backward_calls)
                self._assert_no_update()

    def test_overflowing_gradient_norm_stops_before_update(self):
        with self.assertRaisesRegex(RuntimeError, r"batch=1.*global_step=7"):
            self._run(lambda predicted: predicted.square().mean(), gradient_value=1e38)
        self._assert_no_update()

    def test_finite_loss_and_gradients_allow_update(self):
        stats, step = self._run(lambda predicted: predicted.square().mean())
        self.optimizer.step.assert_called_once()
        self.assertEqual(step, 8)
        self.assertTrue(np.isfinite(stats["total"]))
        self.assertGreater(stats["gradient_norm"], 0.0)
        self.assertTrue(any(
            not torch.equal(parameter.detach(), before)
            for parameter, before in zip(self.parameters, self.before)
        ))

    def test_nonfinite_validation_loss_is_also_rejected(self):
        with self.assertRaisesRegex(FloatingPointError, r"loss.*batch=1.*global_step=7"):
            self._run(lambda predicted: predicted.mean() * float("nan"), training=False)
        self._assert_no_update()


if __name__ == "__main__":
    unittest.main()
