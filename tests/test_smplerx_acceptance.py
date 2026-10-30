"""Temporal initialization and source identity acceptance, without GPU imports."""
import unittest
import numpy as np


class OperatorInitializationTests(unittest.TestCase):
    def test_stale_startup_frame_cannot_initialize_operator_identity(self):
        from unittest.mock import Mock, patch
        from external.smplerx_worker.backend import SmplerXBackend
        from teleoperation.contracts.body import BodyFrameId
        backend = SmplerXBackend.__new__(SmplerXBackend)
        backend._session, backend.model_hash, backend._operator = 's', 'hash', Mock()
        with patch('external.smplerx_worker.backend.time.monotonic', return_value=10.):
            result = backend.infer(np.zeros((4,5,3),dtype=np.uint8), BodyFrameId('s',1,1.))
        self.assertFalse(result.valid['left'])
        self.assertIn('expired',result.reason)
        backend._operator.person.assert_not_called()

    def test_camera_session_change_resets_neutral_and_wrist_alignment(self):
        from unittest.mock import Mock
        from teleoperation.applications.camera_scene_arm import apply_camera_arm_frame
        from tests.smplerx_fixtures import observation
        bridge, driver, wrist = Mock(), Mock(), Mock()
        driver.sides = ()
        bridge.update.return_value = {}
        body = observation(0)[0].upper_body
        apply_camera_arm_frame(bridge, driver, wrist, body, {}, .03)
        bridge.reset.assert_not_called()
        body.metadata['session_id'] = 'changed'
        apply_camera_arm_frame(bridge, driver, wrist, body, {}, .03)
        bridge.reset.assert_called_once_with(recalibrate=True)
        wrist.reset.assert_called_once()

    def test_shape_requires_five_stable_samples_then_freezes(self):
        from external.smplerx_worker.operator_state import OperatorState
        state = OperatorState()
        for index in range(4):
            self.assertIsNone(state.shape_sample(np.ones(10) * (index * .01)))
        fixed = state.shape_sample(np.ones(10) * .02)
        np.testing.assert_allclose(fixed, .02)
        np.testing.assert_array_equal(state.shape_sample(np.ones(10) * 9), fixed)

    def test_unstable_shape_restarts_window(self):
        from external.smplerx_worker.operator_state import OperatorState
        state = OperatorState()
        state.shape_sample(np.zeros(10))
        state.shape_sample(np.ones(10) * 2)
        for _ in range(3):
            self.assertIsNone(state.shape_sample(np.ones(10) * 2))
        np.testing.assert_allclose(state.shape_sample(np.ones(10) * 2), 2)

    def test_ambiguous_or_long_absence_requires_explicit_new_session(self):
        from external.smplerx_worker.operator_state import OperatorState
        state = OperatorState()
        box = np.array([0.,0.,100.,200.])
        self.assertEqual(state.person(box, 1.), '')
        self.assertEqual(state.person(None, 1.2), 'no person detected')
        self.assertEqual(state.person(box, 1.3), '')
        state.person(None, 4.)
        self.assertIn('new session', state.person(box, 4.1))
        state = OperatorState()
        state.person(box, 1.)
        state.person(None, 1.1, ambiguous=True)
        self.assertIn('new session', state.person(box, 1.2))
