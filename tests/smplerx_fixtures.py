"""Explicit synthetic SMPL-X capture fixtures. No model inference is claimed."""
from types import SimpleNamespace
import numpy as np
from teleoperation.contracts.observations import CameraFrame, RawHandFrame, RawTeleopObservation
from teleoperation.inputs.body_tracking import BodyTracker
from teleoperation.data.body_recording import capture_sample, save_body_recording
from tests.test_smplerx_camera_source import body_result

METADATA = dict(source='smplerx', model_id='s32', model_hash='synthetic-test-only',
                time_unit='monotonic_s', capture_kind='synthetic_validation_only')


def observation(index):
    from tests.test_world_hand_pipeline import world_frame, observation as hand_observation
    hands = hand_observation(world_frame(index))
    stamp = 10. + index / 30.
    hands = RawHandFrame(hands.hands, stamp, hands.source, hands.metadata,
                         hands.geometry_encoding, hands.world_hands, hands.wrists)
    result = body_result(index + 1, stamp)
    tracked = BodyTracker().update(result, now_s=stamp)
    return RawTeleopObservation(hands, tracked.body, tracked.wrists), result


def row(index):
    raw, result = observation(index)
    source = SimpleNamespace(session_id='camera', last_frame=CameraFrame(
        np.zeros((4, 5, 3), dtype=np.uint8), raw.hands.timestamp, index), last_result=result)
    return capture_sample(source, raw)


def write(path, count=3):
    return save_body_recording(path, [row(i) for i in range(count)], METADATA)
