"""Build wrist observations before hand-local alignment or handedness reflection."""
import numpy as np
from scipy.spatial.transform import Rotation
from teleoperation.contracts.wrist import WristObservation
from teleoperation.retargeting.hand.coordinates import build_palm_basis, MEDIAPIPE_PALM_INDICES

SIDES = ("left", "right")
CAMERA_REFERENCE = "mediapipe_hand_world_axes"
CAMERA_LOCAL = "palm_wrist_index_middle_v1"
VR_REFERENCE = "visionpro_sdk_zup"
VR_LOCAL = "visionpro_wrist"


def camera_wrists(world_hands, timestamp=None):
    result = {}
    for side in SIDES:
        try:
            points = np.asarray((world_hands or {}).get(side), dtype=float)
            if points.shape != (21, 3):
                raise ValueError("missing world hand landmarks")
            basis = build_palm_basis(points, MEDIAPIPE_PALM_INDICES)
            result[side] = WristObservation(
                Rotation.from_matrix(basis).as_quat(), CAMERA_REFERENCE,
                CAMERA_LOCAL, "camera_palm", timestamp, orientation_valid=True)
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            result[side] = WristObservation(None, CAMERA_REFERENCE, CAMERA_LOCAL,
                                           "camera_palm", timestamp, reason=str(error))
    return result


def vr_wrist(matrix, timestamp=None, *, tracked=True):
    try:
        if not tracked:
            raise ValueError("VR wrist not tracked")
        t = np.asarray(matrix, dtype=float)
        if t.shape == (1, 4, 4):
            t = t[0]
        if t.shape != (4, 4) or not np.isfinite(t).all():
            raise ValueError("missing or malformed VR wrist transform")
        r = t[:3, :3]
        if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("invalid homogeneous wrist transform")
        if not np.allclose(r.T @ r, np.eye(3), atol=2e-3) or not np.isclose(np.linalg.det(r), 1., atol=2e-3):
            raise ValueError("VR wrist rotation is not in SO(3)")
        if np.allclose(t, np.eye(4), atol=1e-8):
            raise ValueError("placeholder identity wrist transform")
        sdk_placeholder = np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                    [0, 1, 0, 0], [0, 0, 0, 1]])
        if np.allclose(t, sdk_placeholder, atol=1e-8):
            raise ValueError("placeholder wrist after SDK YUP2ZUP conversion")
        return WristObservation(Rotation.from_matrix(r).as_quat(), VR_REFERENCE,
                                VR_LOCAL, "visionpro", timestamp, t[:3, 3].copy(), True, True)
    except (ValueError, TypeError) as error:
        return WristObservation(None, VR_REFERENCE, VR_LOCAL, "visionpro",
                                timestamp, reason=str(error))


def recorded_camera_wrist(quaternion, timestamp, reference, local, valid=True):
    if isinstance(reference, bytes):
        reference = reference.decode("utf-8")
    if isinstance(local, bytes):
        local = local.decode("utf-8")
    if reference != CAMERA_REFERENCE or local != CAMERA_LOCAL:
        raise ValueError("Recorded camera wrist coordinate convention differs")
    try:
        if not valid:
            raise ValueError("recorded wrist not tracked")
        return WristObservation(np.asarray(quaternion, dtype=float), reference, local,
                                "camera_palm", timestamp, orientation_valid=True)
    except ValueError as error:
        return WristObservation(None, reference, local, "camera_palm", timestamp, reason=str(error))


def write_wrist_datasets(handle, rows, timestamps):
    """Add versioned optional observations to a camera H5 recording."""
    handle.attrs["wrist_observation_version"] = 1
    handle.attrs["wrist_reference_frame"] = CAMERA_REFERENCE
    handle.attrs["wrist_local_frame"] = CAMERA_LOCAL
    for side in SIDES:
        observations = [camera_wrists({side: row.get("world_" + side, row.get(side + "_world_landmarks"))}, stamp)[side]
                        for row, stamp in zip(rows, timestamps)]
        handle.create_dataset(side + "_wrist_orientation_xyzw", data=np.asarray([
            o.orientation_xyzw if o.orientation_valid else np.full(4, np.nan)
            for o in observations], dtype=float).reshape(-1, 4))
        handle.create_dataset(side + "_wrist_orientation_valid",
                              data=np.asarray([o.orientation_valid for o in observations], dtype=bool))
