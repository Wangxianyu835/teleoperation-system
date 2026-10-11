import numpy as np
from .coordinates import SOURCE_TO_L21_MATRIX

def transform_diagnostics(matrix=None) -> dict:
    matrix = np.asarray(SOURCE_TO_L21_MATRIX if matrix is None else matrix)
    determinant = float(np.linalg.det(matrix))
    orthogonal = bool(np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6))
    return {
        "matrix": matrix.tolist(),
        "row_vector_rule": "p_aligned = p_source @ matrix.T",
        "axis_rule": "x'=-y, y'=z, z'=-x",
        "orthogonal": orthogonal,
        "determinant": determinant,
        "reflection_detected": determinant < 0.0,
        "proper_rotation": orthogonal and bool(np.isclose(determinant, 1.0)),
    }


def signed_hand_volume(hand25: np.ndarray) -> float:
    """Triple product of wrist->thumb/index/pinky MCP, in coordinate units^3."""
    wrist = hand25[0]
    a, b, c = hand25[[1, 6, 21]] - wrist
    return float(np.dot(a, np.cross(b, c)))

