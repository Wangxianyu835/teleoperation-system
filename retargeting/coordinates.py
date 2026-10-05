"""Legacy fixed alignment and independent per-frame palm-local alignment."""

from __future__ import annotations

import numpy as np


COORDINATE_FRAME = "l21"
COORDINATE_ALIGNMENT = "source_to_l21_xyz"

# For row-vector points, p_aligned = p_source @ MATRIX.T.
# This encodes x'=-y, y'=z, z'=-x.
SOURCE_TO_L21_MATRIX = np.asarray(
    (
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    ),
    dtype=np.float32,
)


def align_source_hand_coordinates(points: np.ndarray) -> np.ndarray:
    """Map source-hand coordinates into the fixed L21 frame.

    Supports individual hands, temporal windows, and batches. The input is
    never modified in place.
    """
    values = np.asarray(points, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(
            "Hand coordinates must have at least two dimensions and shape "
            f"(..., 3), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Hand coordinates must be finite")
    aligned = np.matmul(values, SOURCE_TO_L21_MATRIX.T)
    if not np.isfinite(aligned).all():
        raise ValueError("Aligned hand coordinates contain NaN or Inf")
    return aligned.astype(np.float32, copy=False)


# Legacy constants/functions above retain their historical contract.
PALM_LOCAL_COORDINATE_ALIGNMENT = "palm_local_to_l21_v1"
SUPPORTED_COORDINATE_ALIGNMENTS = frozenset({
    COORDINATE_ALIGNMENT, PALM_LOCAL_COORDINATE_ALIGNMENT,
})


def validate_coordinate_alignment(value: object, context: str) -> str:
    """Decode a declared identifier; never infer a mode from absent metadata."""
    if value is None:
        raise ValueError(
            f"{context} does not declare coordinate_alignment. "
            "Refusing to infer legacy or palm-local coordinates."
        )
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{context} coordinate_alignment must be UTF-8") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} coordinate_alignment must be a nonempty scalar string")
    if value not in SUPPORTED_COORDINATE_ALIGNMENTS:
        raise ValueError(
            f"{context}: Unsupported coordinate_alignment={value!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_COORDINATE_ALIGNMENTS))}"
        )
    return value


PALM_BASIS_EPS = 1e-8
SOURCE_PALM_MCP_INDICES = (6, 11, 16, 21)
ROBOT_PALM_MCP_INDICES = (1, 4, 7, 10)
PALM_LOCAL_METADATA = {
    "coordinate_frame": COORDINATE_FRAME,
    "coordinate_alignment": PALM_LOCAL_COORDINATE_ALIGNMENT,
    "source_landmark_space": "mediapipe_normalized",
    "palm_basis_version": "wrist_four_mcp_pinky_to_index_v1",
    "palm_longitudinal": "wrist_to_four_mcp_mean",
    "palm_lateral": "pinky_mcp_to_index_mcp",
    "palm_normal": "lateral_cross_longitudinal",
}


class PalmBasisError(ValueError):
    """Geometry cannot define a finite, proper palm frame."""


def _normalize_palm_axis(vector: np.ndarray, step: str) -> np.ndarray:
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= PALM_BASIS_EPS:
        raise PalmBasisError(f"{step}: axis length {length!r} <= EPS or nonfinite")
    return vector / length


def _validate_palm_basis(basis: np.ndarray) -> None:
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise PalmBasisError("final basis: expected finite (3,3) matrix")
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5, rtol=0):
        raise PalmBasisError("final basis: orthogonality check failed")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-5, rtol=0):
        raise PalmBasisError("final basis: determinant must be +1")


def build_palm_basis(
    points: np.ndarray,
    mcp_indices: tuple[int, int, int, int] = SOURCE_PALM_MCP_INDICES,
    wrist_index: int = 0,
) -> np.ndarray:
    """Columns are X=pinky→index, Y=wrist→four MCP mean, Z=X×Y.

    No history, handedness flips, or scale normalization is used. Degenerate
    geometry raises PalmBasisError; callers decide whether it is a missing
    source frame or a fatal robot-reference error. Float64 intermediates keep
    normalization accurate; the public matrix is float32.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Palm points must have shape (joints,3), got {values.shape}")
    if len(mcp_indices) != 4 or min(wrist_index, *mcp_indices) < 0 or max(wrist_index, *mcp_indices) >= len(values):
        raise ValueError("Palm indices must name the wrist and four existing MCP nodes")
    if not np.isfinite(values).all():
        raise PalmBasisError("input points: nonfinite coordinates")
    y_axis = _normalize_palm_axis(
        values[list(mcp_indices)].mean(axis=0) - values[wrist_index], "longitudinal",
    )
    lateral_raw = values[mcp_indices[0]] - values[mcp_indices[-1]]
    x_axis = _normalize_palm_axis(
        lateral_raw - np.dot(lateral_raw, y_axis) * y_axis, "projected lateral",
    )
    z_axis = _normalize_palm_axis(np.cross(x_axis, y_axis), "cross product normal")
    x_axis = _normalize_palm_axis(np.cross(y_axis, z_axis), "recomputed lateral")
    basis = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)
    _validate_palm_basis(basis)
    return basis


def build_l21_reference_basis(side: str) -> np.ndarray:
    """Compute one side's zero-pose reference using unchanged 23-node L21 FK."""
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side: {side!r}")
    # Lazy imports keep the legacy NumPy-only coordinate API lightweight.
    import torch
    from model.kinematics import create_hand_kinematics
    from retargeting.config import L21

    fk = create_hand_kinematics(
        getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(), device="cpu",
        scale_factor=L21.training.robot_scale,
    )
    with torch.no_grad():
        _, _, global_positions = fk.forward(torch.zeros((1, 23), dtype=torch.float32))
    points = global_positions[0].detach().cpu().numpy()
    try:
        return build_palm_basis(points, ROBOT_PALM_MCP_INDICES)
    except PalmBasisError as error:
        nodes = (0, *ROBOT_PALM_MCP_INDICES)
        positions = {index: points[index].tolist() for index in nodes}
        raise PalmBasisError(
            f"L21 reference side={side}; robot node indices={nodes}; "
            f"positions={positions}; failed step: {error}"
        ) from error


def align_palm_local_coordinates(points25: np.ndarray, robot_basis: np.ndarray) -> np.ndarray:
    """Align an already wrist-relative (25,3) frame, without recentering it.

    Row vectors: local = points25 @ B_source;
    aligned = local @ B_robot.T. Invalid source geometry returns a zero frame;
    invalid robot geometry is a configuration error and must raise.
    """
    values = np.asarray(points25, dtype=np.float32)
    if values.shape != (25, 3):
        raise ValueError(f"Expected wrist-relative (25,3) source, got {values.shape}")
    reference = np.asarray(robot_basis, dtype=np.float32)
    _validate_palm_basis(reference)
    zero = np.zeros((25, 3), dtype=np.float32)
    if not np.isfinite(values).all() or not np.any(values):
        return zero
    try:
        source_basis = build_palm_basis(values)
    except PalmBasisError:
        return zero
    with np.errstate(over="ignore", invalid="ignore"):
        aligned = values @ source_basis @ reference.T
    return aligned.astype(np.float32, copy=False) if np.isfinite(aligned).all() else zero
