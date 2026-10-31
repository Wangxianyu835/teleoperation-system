"""Operator identity must follow continuity instead of detector box count."""
import unittest
import numpy as np

from external.smplerx_worker.operator_state import OperatorState, filter_person_detections


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

    def test_initial_multiple_people_choose_highest_score_immediately(self):
        state = OperatorState()
        box, reason = state.select_person(np.array([[0, 0, 100, 200, .6],
                                                    [200, 0, 500, 400, .99]]), 1.)
        np.testing.assert_array_equal(box, [200, 0, 500, 400])
        self.assertEqual(reason, '')
        self.assertFalse(state.locked)

    def test_close_competing_matches_do_not_hold_or_require_recovery(self):
        state = OperatorState()
        original = np.array([[0., 0., 100., 200., .99]])
        state.select_person(original, 1.)
        box, reason = state.select_person(np.vstack([original, [2, 0, 102, 200, .98]]), 1.1)
        np.testing.assert_array_equal(box, original[0, :4])
        self.assertEqual(reason, '')
        for stamp in (1.2, 1.3):
            box, reason = state.select_person(original, stamp)
            np.testing.assert_array_equal(box, original[0, :4])
            self.assertEqual(reason, '')

    def test_unrelated_jump_still_rejects_the_result(self):
        state = OperatorState()
        state.select_person(np.array([[0., 0., 100., 200., .99]]), 1.)
        box, reason = state.select_person(np.array([[300., 0., 400., 200., .99]]), 1.2)
        self.assertIsNone(box)
        self.assertTrue(state.locked)

    def test_moderate_iou_and_small_winning_margin_are_accepted(self):
        state = OperatorState()
        state.select_person(np.array([[0., 0., 100., 200., .99]]), 1.)
        box, reason = state.select_person(np.array([[50., 0., 150., 200., .4],
                                                    [60., 0., 160., 200., .99]]), 1.2)
        np.testing.assert_array_equal(box, [50., 0., 150., 200.])
        self.assertEqual(reason, '')

    def test_equal_iou_uses_confidence_to_break_tie(self):
        state = OperatorState()
        state.select_person(np.array([[0., 0., 100., 200., .99]]), 1.)
        box, reason = state.select_person(np.array([[0., 0., 100., 200., .4],
                                                    [0., 0., 100., 200., .9]]), 1.2)
        self.assertEqual(reason, '')
        self.assertEqual(state.selection['selected_index'], 1)

    def test_repeated_competing_boxes_do_not_accumulate_a_timeout(self):
        state = OperatorState()
        boxes = np.array([[0., 0., 100., 200., .8], [2., 0., 102., 200., .9]])
        for sequence in range(20):
            box, reason = state.select_person(boxes, 1. + sequence * .2)
            self.assertIsNotNone(box)
            self.assertEqual(reason, '')
            self.assertFalse(state.locked)

    def test_person_filter_accepts_lower_confidence_but_rejects_small_boxes(self):
        boxes = [[0, 0, 100, 200, .29], [0, 0, 100, 200, .30],
                 [0, 0, 100, 200, .49], [0, 0, 49, 200, .99], [0, 0, 100, 149, .99]]
        accepted = filter_person_detections(boxes)
        np.testing.assert_allclose(accepted, [[0, 0, 100, 200, .30], [0, 0, 100, 200, .49]])


if __name__ == '__main__':
    unittest.main()
