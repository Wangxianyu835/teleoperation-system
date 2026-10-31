"""Operator identity must follow continuity instead of detector box count."""
import unittest
import numpy as np

from external.smplerx_worker.operator_state import OperatorState


class OperatorAssociationTests(unittest.TestCase):
    def test_recorded_local_false_positive_does_not_lock_operator(self):
        state = OperatorState()
        previous = [122.934, .697, 527.605, 476.749, .9979]
        operator = [84.003, 0., 484.354, 478.224, .9974]
        false_positive = [430.259, .001, 606.958, 469.929, .5347]
        box, reason = state.select_person(np.array([previous]), 1.)
        self.assertEqual(reason, '')
        box, reason = state.select_person(np.array([false_positive, operator]), 1.2)
        self.assertEqual(reason, '')
        np.testing.assert_allclose(box, operator[:4])
        self.assertFalse(state.locked)

    def test_initial_multiple_people_cannot_choose_largest_or_highest_score(self):
        state = OperatorState()
        box, reason = state.select_person(np.array([[0, 0, 100, 200, .6],
                                                    [200, 0, 500, 400, .99]]), 1.)
        self.assertIsNone(box)
        self.assertTrue(reason)
        self.assertFalse(state.locked)

    def test_short_ambiguity_requires_three_consistent_frames_to_recover(self):
        state = OperatorState()
        original = np.array([[0., 0., 100., 200., .99]])
        state.select_person(original, 1.)
        box, reason = state.select_person(np.vstack([original, [2, 0, 102, 200, .98]]), 1.1)
        self.assertIsNone(box)
        self.assertIn('ambiguous', reason)
        for stamp in (1.2, 1.3):
            box, reason = state.select_person(original, stamp)
            self.assertIsNone(box)
            self.assertIn('recover', reason)
        box, reason = state.select_person(original, 1.4)
        np.testing.assert_array_equal(box, original[0, :4])
        self.assertEqual(reason, '')

    def test_recovery_cannot_adopt_unrelated_person(self):
        state = OperatorState()
        state.select_person(np.array([[0., 0., 100., 200., .99]]), 1.)
        state.person(None, 1.1, ambiguous=True)
        for stamp in (1.2, 1.3, 1.4):
            box, reason = state.select_person(np.array([[300., 0., 400., 200., .99]]), stamp)
            self.assertIsNone(box)
        box, reason = state.select_person(np.array([[300., 0., 400., 200., .99]]), 3.1)
        self.assertIsNone(box)
        self.assertTrue(state.locked)

    def test_ambiguity_breaks_recovery_streak(self):
        state = OperatorState()
        box = np.array([0., 0., 100., 200.])
        state.person(box, 1.)
        state.person(None, 1.1, ambiguous=True)
        state.person(box, 1.2)
        state.person(box, 1.3)
        state.person(None, 1.4, ambiguous=True)
        self.assertTrue(state.person(box, 1.5))
        self.assertTrue(state.person(box, 1.6))
        self.assertEqual(state.person(box, 1.7), '')

    def test_persistent_ambiguity_at_start_requires_new_session(self):
        state = OperatorState()
        state.person(None, 1., ambiguous=True)
        self.assertIn('new session', state.person(None, 3.1, ambiguous=True))


if __name__ == '__main__':
    unittest.main()
