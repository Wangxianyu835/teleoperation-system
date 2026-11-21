"""Parse Vision Pro wrist observations without changing hand geometry."""
import numpy as np
from scipy.spatial.transform import Rotation
from teleoperation.contracts.wrist import WristObservation

VR_REFERENCE = "visionpro_sdk_zup"
VR_LOCAL = "visionpro_wrist"


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
