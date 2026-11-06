import unittest
from unittest.mock import Mock,patch


class SceneSetupTests(unittest.TestCase):
    def test_initialization_retries_and_keeps_the_selected_client(self):
        from teleoperation.simulation.urdf_loader import prepare_physics
        call=Mock(side_effect=[RuntimeError('Not connected to physics server'),None])
        with patch('time.perf_counter',side_effect=[0.,.1]),patch('time.sleep') as pause:
            prepare_physics(call,(0,0,-9.81),client_id=7)
        self.assertEqual(call.call_count,2)
        for invocation in call.call_args_list:self.assertEqual(invocation.kwargs,{'physicsClientId':7})
        pause.assert_called_once_with(.1)

    def test_initialization_timeout_reports_the_original_error(self):
        from teleoperation.simulation.urdf_loader import prepare_physics
        call=Mock(side_effect=RuntimeError('Not connected to physics server'))
        with patch('time.perf_counter',side_effect=[0.,2.]),patch('time.sleep') as pause:
            with self.assertRaisesRegex(RuntimeError,'PyBullet did not answer within 1 s') as error:
                prepare_physics(call,(),client_id=3,timeout=1.)
        self.assertIsInstance(error.exception.__cause__,RuntimeError)
        pause.assert_not_called()
