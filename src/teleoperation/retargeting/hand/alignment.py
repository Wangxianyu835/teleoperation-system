"""Palm-local recording alignment math, independent of storage."""
import numpy as np
from .coordinates import build_l21_reference_basis, align_mediapipe_hand, OPERATOR_TO_CANONICAL


def align_recording(raw_hands, landmark_space="normalized"):
    if landmark_space not in ("world", "normalized"):
        raise ValueError(f"Unknown landmark space: {landmark_space!r}")
    references = (OPERATOR_TO_CANONICAL if landmark_space == "world" else
                  {side: build_l21_reference_basis(side) for side in ("left", "right")})
    aligned, statistics = {}, {}
    for side, raw in raw_hands.items():
        result = np.zeros((len(raw), 25, 3), dtype=np.float32)
        missing, degenerate = 0, 0
        for index, frame in enumerate(raw):
            result[index] = (align_mediapipe_hand(frame, side=side) if landmark_space == "world"
                             else align_mediapipe_hand(frame, references[side]))
            if not np.any(frame):
                missing += 1
            elif not np.any(result[index]):
                degenerate += 1
        aligned[side] = result
        basis = references[side]
        statistics[side] = {
            "nonzero_frames": int(np.any(result != 0, axis=(1, 2)).sum()),
            "zero_source_frames": missing,
            "degenerate_frames": degenerate,
            "nonfinite_values": int((~np.isfinite(result)).sum()),
            "output_matrix": basis.tolist(),
            "determinant": float(np.linalg.det(basis.astype(np.float64))),
            "orthogonality_error": float(np.max(np.abs(basis.astype(np.float64).T @ basis - np.eye(3)))),
        }
    return aligned, statistics
