# Coordinate Systems and Units

This document records the coordinate and unit evidence for the current
Hand-only offline path. It is an audit report, not a change to the numerical
pipeline.

## Status vocabulary

- `CODE_DEFINED`: directly implemented or declared by repository code.
- `DOCUMENTED`: stated by an upstream or official document.
- `VERIFIED`: code, documentation, and a repeatable experiment agree.
- `UNRESOLVED`: the available evidence is insufficient.

No item in this report is marked `VERIFIED` unless an experiment is described
and has actually been run.

The official MediaPipe URLs are included below as the normative sources. Live
fetching of `ai.google.dev` was blocked by the current sandbox, so these
upstream statements still require a human browser check before they can be
promoted from `DOCUMENTED` to `VERIFIED` for this project.

## MediaPipe source contract

The locally audited collector (`hand capture media.py`, in the separate capture project) uses
the MediaPipe Tasks Hand Landmarker in `VIDEO` mode. For each detection it
stores `[lm.x, lm.y, lm.z]` from `result.hand_landmarks`; it does not read
`result.hand_world_landmarks`. The body-capture script follows the same path.

| Claim | Status | Evidence / consequence |
|---|---|---|
| `x` is horizontal image position normalized by image width | `DOCUMENTED` | MediaPipe Hand Landmarker documentation: [Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python). The collector confirms pixel drawing with `int(lm.x * width)`. |
| `y` is vertical image position normalized by image height | `DOCUMENTED` | Same official guide; collector uses `int(lm.y * height)`. |
| `z` is relative depth, with the wrist as the origin and a magnitude scale roughly comparable to normalized image coordinates | `DOCUMENTED` | MediaPipe Hand Landmarker result/landmark contract. This is not a metric distance field. |
| `hand_landmarks` are meters or another metric unit | `UNRESOLVED` | The official contract describes normalized image coordinates and relative depth, not a calibrated length unit. The repository performs no calibration. |
| `hand_world_landmarks` are real-world 3D coordinates in meters | `DOCUMENTED` | Official Hand Landmarker output documentation describes world landmarks in meters and gives a hand-centered origin. This output is not used by this repository. |
| Wrist is landmark index `0` and is the reference used by the source contract | `DOCUMENTED` | Official 21-landmark topology; repository `wrist_relative()` subtracts point `0`. |
| Current code uses world landmarks | `CODE_DEFINED` | False. Both collectors read only `result.hand_landmarks`; `hand_world_landmarks` has no runtime reference in the capture path. |
| Handedness labels are mirror-safe for the current unmirrored capture | `UNRESOLVED` | Official handedness documentation assumes a horizontally mirrored/selfie input and says labels must be swapped for a non-mirrored input. The collector explicitly does not mirror the frame (comment in the capture script), but no left/right physical-camera experiment is recorded. |

The capture code branches on `result.handedness[i][0].category_name`, placing
`"Left"` in `left_hand` and every other label in `right_hand`. The comment
`不镜像！左右手正确` is local code intent, not a physical validation.

## Canonicalization and offline alignment

The offline loader requires an H5 root attribute `coordinate_frame="l21"`.
The explicit preprocessing CLI `scripts/align_h5_coordinates.py` performs:

1. `ensure_hand25()` (21 points to the project 25-point topology, then wrist-relative subtraction).
2. `align_source_hand_coordinates()` on both hands.
3. Writes `coordinate_frame="l21"` and `coordinate_alignment="source_to_l21_xyz"`.

The loader validates `coordinate_frame`, but does not validate the
`coordinate_alignment` attribute. This is `CODE_DEFINED`.

`retargeting/coordinates.py` defines, for row vectors,

```text
p_aligned = p_source @ M.T
M = [[ 0, -1,  0],
     [ 0,  0,  1],
     [-1,  0,  0]]

x' = -y
y' =  z
z' = -x
```

The transform and its name are `CODE_DEFINED`. Git history shows that commit
`319e9f5` (2026-09-29, “加入坐标系对齐”) introduced this hard-coded matrix;
the commit message and repository search contain no paper, upstream SDK, or
calibration reference. Therefore its physical correctness is `UNRESOLVED`.
The preceding refactor commit `767c2df` also does not provide an external
derivation. It must not be treated as a verified coordinate proof.

## L21 base frame evidence

The active URDFs are:

- `dataset/robot/l21_left/linkerhand_l21_left.urdf`
- `dataset/robot/l21_right/linkerhand_l21_right.urdf`

Both define `hand_base_link` visual and collision origins as `xyz="0 0 0"`
and `rpy="0 0 0"`, using `meshes/hand_base_link.STL`. This is
`CODE_DEFINED`. The URDF has no textual declaration that +X, +Y, or +Z means
“fingers”, “palm”, “back”, or another human anatomical direction. Mesh axes
and joint axes alone do not uniquely establish those names, so the physical
interpretation of +X/+Y/+Z is `UNRESOLVED` pending mesh visualization and a
known-pose experiment.

The complete joint-level evidence is in
[`L21_JOINT_CONTRACT.md`](L21_JOINT_CONTRACT.md).

## Scale and units

| Stage | What code does | Status |
|---|---|---|
| MediaPipe input | Copies normalized `x,y,z` values from `hand_landmarks` | `CODE_DEFINED`; absolute unit `UNRESOLVED` |
| `wrist_relative` | `(points - points[0:1]) * scale_factor`; default `scale_factor=1.0` | `CODE_DEFINED` |
| H5 dataset | `CanonicalHandProcessor` converts 21→25, wrist-relativizes, and keeps the configured scale | `CODE_DEFINED` |
| Training source scale | `L21.training.source_scale = 1.0` | `CODE_DEFINED`; physical reasonableness `UNRESOLVED` |
| FK geometry | URDF origins are approximately `0.018` to `0.141` in the file's length units; no unit declaration is present in the repository | `CODE_DEFINED`; meters `UNRESOLVED` |
| Training robot scale | `L21.training.robot_scale = 1.0`, passed to FK | `CODE_DEFINED`; scale matching `UNRESOLVED` |

There is no documented calibration step that maps a human hand span in
normalized MediaPipe coordinates to the URDF link lengths. Consequently a
default factor of `1.0` is only a code default, not a demonstrated metric
match (`UNRESOLVED`).

### Loss sensitivity

The current losses in `model/losses.py` have different scale behavior:

| Loss | Implementation | Absolute-scale sensitivity |
|---|---|---|
| `vec_inter_loss` | Differences between MCP/DIP vectors followed by `F.normalize` | Direction-only after normalization; largely scale-insensitive, except degenerate/near-zero vectors (`CODE_DEFINED`). |
| `tip_pos_loss` | Tip-minus-MCP vectors followed by `F.normalize` | Direction-only after normalization; largely scale-insensitive (`CODE_DEFINED`). |
| `thumb_loss` | Point-to-plane distances; target is multiplied by `0.9` | Sensitive to relative length scale (`CODE_DEFINED`). |
| `tip_distance_loss` | Pairwise tip distances, each multiplied by `1000.0` | Sensitive to absolute relative scale; the multiplication is `CODE_DEFINED`, while its physical unit is `UNRESOLVED`. |
| `thumb_loss2` | Angles between normalized thumb segment vectors | Direction/angle-only, scale-insensitive except degenerate vectors (`CODE_DEFINED`). |
| `CollisionLoss` | Raw FK point distances compared with `threshold=0.010` | Sensitive to FK length units and scale (`CODE_DEFINED`). |

No loss normalizes all Cartesian distances to a shared hand-size measure.
Whether the chosen source and robot scales are compatible remains
`UNRESOLVED`.

## Minimum evidence needed for `VERIFIED`

### Automatically tested software invariants

`test_coordinate_modes.py`, `test_coordinate_contracts.py`, and
`test_coordinate_diagnostics.py` verify basis mapping (+X -> -Z, +Y -> -X,
+Z -> +Y), orthogonality, determinant +1 and detection of an injected
determinant -1 reflection. Non-coplanar synthetic bilateral geometries preserve
signed volume through conversion, wrist subtraction, rotation and labeled windows.
Signed volume is an orientation probe, not a physical handedness classifier.

Wrist subtraction and rotation preserve lengths. `source_scale` reaches model
windows and newest-frame targets; both FK sides apply `robot_scale`. A common
position scale multiplies fingertip MSE by its square; the collision threshold
remains 0.010 in scaled FK coordinate units. Multiplying distances by 1000 does
not establish millimeters. Zero/numerically unresolved thumb segments are masked
under the contract in [the verification report](P0_VERIFICATION_REPORT.md).

The read-only diagnostic reports declarations, missing metadata, representative
distances, signed volume and FK geometry without modifying the input:

```text
python scripts/diagnose_hand_coordinates.py --input input/aligned_visual_hand_data_20260912_153542.h5
```

The smallest useful experiments are listed in
[`UNRESOLVED_VERIFICATION_PLAN.md`](UNRESOLVED_VERIFICATION_PLAN.md). Until
they are run, this document intentionally does not claim that the current
matrix, handedness, or scale is physically correct.
