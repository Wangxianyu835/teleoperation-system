import unittest

import numpy as np

from teleoperation.contracts.body import BodyFrameId, SmplerXResult
from teleoperation.contracts.observations import CameraFrame, RawHandFrame
from teleoperation.applications.camera_teleop_source import CameraTeleopSource


def body_result(sequence=1, timestamp=1.0):
    joints = np.zeros((6, 3), dtype=np.float64)
    joints[0:3] = [[0., .2, 0.], [0., .1, 0.], [.1, 0., 0.]]
    joints[3:6] = [[0., -.2, 0.], [0., -.1, 0.], [-.1, 0., 0.]]
    return SmplerXResult(
        BodyFrameId("camera", sequence, timestamp), timestamp + .01, "s32", "d" * 64,
        {"joints_root_m": joints, "global_rotations": np.tile(np.eye(3), (6, 1, 1)),
         "image_joints": joints}, {"left": True, "right": True},
        metadata={"joint_indices": {
            "left": {"shoulder": 0, "elbow": 1, "wrist": 2},
            "right": {"shoulder": 3, "elbow": 4, "wrist": 5},
        }},
    )


class FakeCamera:
    def __init__(self):
        self.calls = 0

    def capture_sample(self):
        self.calls += 1
        return CameraFrame(np.zeros((4, 5, 3), dtype=np.uint8), 1.0, self.calls)


class FakeHands:
    def __init__(self):
        self.frames = []

    def process_sample(self, frame):
        self.frames.append(frame)
        return RawHandFrame({"left": None, "right": None}, frame.timestamp, "hands")


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.submissions = []

    def submit(self, frame_id, rgb):
        self.submissions.append((frame_id, rgb))

    def poll(self):
        result, self.result = self.result, None
        return result


class SmplerXCameraSourceTests(unittest.TestCase):
    def test_worker_result_counts_do_not_count_waiting_polls(self):
        result = SmplerXResult(BodyFrameId('camera', 1, 1.), 1.1, 's32', 'd' * 64,
                              {}, {'left': False, 'right': False}, reason='no person detected')
        source = CameraTeleopSource(FakeCamera(), FakeHands(), FakeClient(result))
        source.next_teleop_observation()
        source.next_teleop_observation()
        summary = source.worker_summary()
        self.assertEqual(summary['results_received'], 1)
        self.assertEqual(summary['result_reasons'], {'no person detected': 1})
        self.assertEqual(summary['valid_results'], {'left': 0, 'right': 0})
        self.assertEqual(summary['last_result']['sequence'], 1)

    def test_worker_fault_is_retained_in_summary(self):
        class FailedClient(FakeClient):
            def poll(self):
                raise RuntimeError('worker exited: missing asset')
        source = CameraTeleopSource(FakeCamera(), FakeHands(), FailedClient(None))
        source.next_teleop_observation()
        self.assertEqual(source.worker_summary()['fault'], 'worker exited: missing asset')
        self.assertFalse(source.last_tracked.body.valid['left'])

    def test_one_capture_is_shared_by_hands_and_body_worker(self):
        camera, hands = FakeCamera(), FakeHands()
        client = FakeClient(body_result())
        source = CameraTeleopSource(camera, hands, client, session_id="camera")
        observation = source.next_teleop_observation()
        self.assertEqual(camera.calls, 1)
        self.assertIs(hands.frames[0].rgb, client.submissions[0][1])
        self.assertTrue(observation.upper_body.valid["left"])
        self.assertEqual(observation.wrists["left"].source, "smplerx")

    def test_no_arms_does_not_start_or_call_worker(self):
        camera, hands = FakeCamera(), FakeHands()
        source = CameraTeleopSource(camera, hands, None, arms_enabled=False)
        observation = source.next_teleop_observation()
        self.assertFalse(observation.upper_body.valid["left"])
        self.assertIsNone(observation.wrists)

    def test_hands_and_body_validity_are_independent(self):
        camera, hands = FakeCamera(), FakeHands()
        client = FakeClient(None)
        source = CameraTeleopSource(camera, hands, client, session_id="camera")
        observation = source.next_teleop_observation()
        self.assertIsNone(observation.hands.hands["left"])
        self.assertFalse(observation.upper_body.valid["left"])


if __name__ == "__main__":
    unittest.main()
