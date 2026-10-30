import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.contracts.body import BodyFrameId, SmplerXResult
from teleoperation.inputs.body_tracking import BodyTracker


def make_result(sequence, capture_time, *, offset=0.0, left=True, right=True, rotation=None):
    joints = np.zeros((6, 3), dtype=np.float64)
    joints[0:3] = [[0.0, 0.2, 0.0], [0.0, 0.1, 0.0], [offset, 0.0, 0.0]]
    joints[3:6] = [[0.0, -0.2, 0.0], [0.0, -0.1, 0.0], [-offset, 0.0, 0.0]]
    rotations = np.tile(np.eye(3), (6, 1, 1))
    if rotation is not None:
        rotations[2] = rotation
    return SmplerXResult(
        BodyFrameId("session", sequence, capture_time), capture_time + .01,
        "s32", "c" * 64,
        {"joints_root_m": joints, "global_rotations": rotations, "image_joints": joints},
        {"left": left, "right": right},
        metadata={"joint_indices": {
            "left": {"shoulder": 0, "elbow": 1, "wrist": 2},
            "right": {"shoulder": 3, "elbow": 4, "wrist": 5},
        }},
    )


class BodyTrackerTests(unittest.TestCase):
    def test_duplicate_tracking_sample_cannot_count_for_calibration(self):
        tracker = BodyTracker()
        result = make_result(1, 1.)
        tracker.update(result, now_s=1.)
        repeated = tracker.update(result, now_s=1.01)
        self.assertFalse(repeated.body.valid['left'])
        self.assertFalse(repeated.fresh['left'])

    def test_waiting_and_held_samples_keep_source_contract(self):
        tracker = BodyTracker()
        waiting = tracker.update(None, now_s=1.)
        self.assertEqual(waiting.body.metadata['source_coordinate_frame'], 'smplx_camera_root_relative_m')
        tracker.update(make_result(1, 1.), now_s=1.)
        held = tracker.update(None, now_s=1.1)
        self.assertEqual(held.body.timestamp, 1.)
        self.assertEqual(held.body.metadata['sample_sequence'], 1)
        self.assertFalse(held.body.valid['left'])

    def test_expiry_invalidates_body_and_wrist_without_discarding_hold_position(self):
        tracker = BodyTracker()
        tracker.update(make_result(1, 1.), now_s=1.)
        held = tracker.update(None, now_s=1.501)
        self.assertIsNotNone(held.body.world_arms['left'])
        self.assertFalse(held.body.valid['left'])
        self.assertFalse(held.wrists['left'].orientation_valid)

    def test_expired_source_stays_held(self):
        tracker = BodyTracker(max_age_s=.5)
        tracker.update(make_result(1, 0.0), now_s=0.0)
        tracked = tracker.update(make_result(2, 0.0), now_s=.501)
        self.assertFalse(tracked.fresh["left"])
        self.assertEqual(tracked.state_by_side["left"], "lost_hold")

    def test_jump_is_held_until_three_distinct_samples_are_consistent(self):
        tracker = BodyTracker(position_jump_m=.15, recovery_samples=3)
        tracker.update(make_result(1, 0.0), now_s=0.0)
        for sequence in (2, 3):
            tracked = tracker.update(make_result(sequence, sequence / 30, offset=.4), now_s=sequence / 30)
            self.assertEqual(tracked.state_by_side["left"], "reacquiring")
            self.assertAlmostEqual(tracked.body.world_arms["left"][2, 0], 0.0)
        tracked = tracker.update(make_result(4, 4 / 30, offset=.4), now_s=4 / 30)
        self.assertEqual(tracked.state_by_side["left"], "tracking")
        self.assertAlmostEqual(tracked.body.world_arms["left"][2, 0], .4)

    def test_duplicate_sample_does_not_count_as_recovery(self):
        tracker = BodyTracker(position_jump_m=.15, recovery_samples=3)
        tracker.update(make_result(1, 0.0), now_s=0.0)
        tracker.update(make_result(2, 2 / 30, offset=.4), now_s=2 / 30)
        tracked = tracker.update(make_result(2, 2 / 30, offset=.4), now_s=2 / 30)
        self.assertEqual(tracked.state_by_side["left"], "reacquiring")

    def test_missing_right_side_does_not_stop_left_side(self):
        tracker = BodyTracker()
        tracked = tracker.update(make_result(1, 0.0, right=False), now_s=0.0)
        self.assertEqual(tracked.state_by_side["left"], "tracking")
        self.assertIn(tracked.state_by_side["right"], ("hold", "lost_hold", "calibrating"))

    def test_session_change_resets_previous_body(self):
        tracker = BodyTracker()
        tracker.update(make_result(1, 0.0), now_s=0.0)
        changed = make_result(1, 1.0)
        changed = SmplerXResult(
            BodyFrameId("new-session", 1, 1.0), changed.completed_time_s,
            changed.model_id, changed.model_hash, changed.arrays, changed.valid,
            metadata=changed.metadata,
        )
        tracked = tracker.update(changed, now_s=1.0)
        self.assertEqual(tracked.state_by_side["left"], "tracking")


if __name__ == "__main__":
    unittest.main()
