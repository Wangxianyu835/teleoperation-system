"""Palm-local recording alignment math, independent of storage."""
import numpy as np
from .coordinates import build_l21_reference_basis, align_palm_local_coordinates
from .topology import ensure_hand25


def align_recording(raw_hands):
    references = {side: build_l21_reference_basis(side) for side in ("left", "right")}
    aligned, statistics = {}, {}
    for side, raw in raw_hands.items():
        result = np.zeros((len(raw), 25, 3), dtype=np.float32)
        missing, degenerate = 0, 0
        for index, frame in enumerate(raw):
            # ensure_hand25 already performs wrist-relative conversion.
            with np.errstate(invalid="ignore", over="ignore"):
                points25 = ensure_hand25(frame)
            result[index] = align_palm_local_coordinates(points25, references[side])
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
            "robot_basis": basis.tolist(),
            "determinant": float(np.linalg.det(basis.astype(np.float64))),
            "orthogonality_error": float(np.max(np.abs(basis.astype(np.float64).T @ basis - np.eye(3)))),
        }
    return aligned, statistics
