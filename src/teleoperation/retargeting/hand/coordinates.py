"""Three-point palm frames and explicit row-vector handedness normalization."""

from __future__ import annotations

import numpy as np


from teleoperation.contracts.coordinates import *

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
PALM_BASIS_EPS = 1e-8
PALM_COLLINEAR_RATIO = 1e-3
SOURCE_PALM_INDICES = (0, 6, 11)  # wrist, index MCP, middle MCP in hand25
ROBOT_PALM_INDICES = (0, 1, 4)
MEDIAPIPE_PALM_INDICES = (0, 5, 9)
# dex matrices act on rows: [a,b,c] -> [-b,c,-a] / [b,-c,-a].
OPERATOR_TO_DEX_RIGHT = np.asarray(((0, 0, -1), (-1, 0, 0), (0, 1, 0)), dtype=np.float32)
OPERATOR_TO_DEX_LEFT = np.asarray(((0, 0, -1), (1, 0, 0), (0, -1, 0)), dtype=np.float32)
LEFT_DEX_TO_CANONICAL = np.diag([1, -1, 1]).astype(np.float32)
OPERATOR_TO_CANONICAL = {
    "right": OPERATOR_TO_DEX_RIGHT,
    "left": OPERATOR_TO_DEX_LEFT @ LEFT_DEX_TO_CANONICAL,
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
    reference_indices: tuple[int, int, int] = SOURCE_PALM_INDICES,
) -> np.ndarray:
    """dex intermediate frame: X=middle->wrist, Y=normal, Z=toward index.

    Columns are axes in the input frame; row points become local via p @ R.
    Only wrist/index/middle define orientation. R is always a proper rotation;
    left-hand reflection belongs to OPERATOR_TO_CANONICAL, never to R.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Palm points must have shape (joints,3), got {values.shape}")
    if len(reference_indices) != 3 or len(set(reference_indices)) != 3 or min(reference_indices) < 0 or max(reference_indices) >= len(values):
        raise ValueError("Palm indices must name three distinct wrist/index/middle nodes")
    if not np.isfinite(values).all():
        raise PalmBasisError("input points: nonfinite coordinates")
    wrist, index, middle = values[list(reference_indices)]
    longitudinal = _normalize_palm_axis(wrist - middle, "middle-to-wrist")
    centered = np.stack((wrist, index, middle))
    centered -= centered.mean(axis=0, keepdims=True)
    try:
        _, singular_values, vh = np.linalg.svd(centered)
    except np.linalg.LinAlgError as error:
        raise PalmBasisError("palm SVD did not converge") from error
    if singular_values[0] <= PALM_BASIS_EPS or singular_values[1] <= max(PALM_BASIS_EPS, PALM_COLLINEAR_RATIO * singular_values[0]):
        raise PalmBasisError("wrist/index/middle are coincident or nearly collinear")
    y_axis = vh[-1]
    x_axis = _normalize_palm_axis(longitudinal - np.dot(longitudinal, y_axis) * y_axis, "projected longitudinal")
    z_axis = _normalize_palm_axis(np.cross(x_axis, y_axis), "lateral")
    if np.dot(z_axis, index - middle) < 0:
        y_axis = -y_axis
        z_axis = -z_axis
    basis = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)
    _validate_palm_basis(basis)
    return basis


def build_l21_reference_basis(side: str) -> np.ndarray:
    """Compute one side's zero-pose reference using unchanged 23-node L21 FK."""
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side: {side!r}")
    # Lazy imports keep the legacy NumPy-only coordinate API lightweight.
    import torch
    from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
    from teleoperation.retargeting.hand.config import L21

    fk = create_hand_kinematics(
        getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(), device="cpu",
        scale_factor=L21.training.robot_scale,
    )
    with torch.no_grad():
        _, _, global_positions = fk.forward(torch.zeros((1, 23), dtype=torch.float32))
    points = global_positions[0].detach().cpu().numpy()
    try:
        return build_palm_basis(points, ROBOT_PALM_INDICES)
    except PalmBasisError as error:
        nodes = ROBOT_PALM_INDICES
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


def align_mediapipe_hand(points21, robot_basis: np.ndarray | None = None, *, side: str | None = None) -> np.ndarray:
    """Raw21/25 -> wrist-relative -> three-point local -> declared output frame.

    side selects canonical right-hand coordinates; robot_basis selects native
    root coordinates for normalized compatibility callers. Never select both.
    """
    from .topology import expand_hand21_to25, ensure_hand25

    if (side is None) == (robot_basis is None):
        raise ValueError("Select canonical side or native robot_basis, exclusively")
    if side is not None and side not in OPERATOR_TO_CANONICAL:
        raise ValueError(f"Unknown hand side: {side!r}")
    if robot_basis is not None:
        _validate_palm_basis(np.asarray(robot_basis))

    if points21 is None:
        return np.zeros((25, 3), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.asarray(points21, dtype=np.float32)
        if points.shape not in ((21, 3), (25, 3)):
            raise ValueError(f"Hand shape must be (21,3) or (25,3), got {points.shape}")
        if not np.isfinite(points).all() or not np.any(points):
            return np.zeros((25, 3), dtype=np.float32)
        wrist_relative = points - points[0:1]
        if side is None:
            return align_palm_local_coordinates(ensure_hand25(points), robot_basis)
        try:
            palm_basis = build_palm_basis(wrist_relative, MEDIAPIPE_PALM_INDICES if len(points) == 21 else SOURCE_PALM_INDICES)
        except PalmBasisError:
            return np.zeros((25, 3), dtype=np.float32)
        canonical = wrist_relative @ palm_basis @ OPERATOR_TO_CANONICAL[side]
        if not np.isfinite(canonical).all():
            return np.zeros((25, 3), dtype=np.float32)
        return expand_hand21_to25(canonical) if len(canonical) == 21 else canonical.astype(np.float32, copy=False)
