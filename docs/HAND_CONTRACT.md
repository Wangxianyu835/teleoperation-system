# Hand-only L21 Data Contract

This is the contract declared by the current implementation. Physical coordinate interpretation, source units, and hardware-specific joint semantics are explicitly separated where the repository does not prove them.

## Status vocabulary

- **CODE-DEFINED**: enforced or directly constructed by current code.
- **UNRESOLVED**: cannot be established from this repository alone and requires human/source-device confirmation.

## Layered shapes and types

| Layer | Shape / dtype | Units | Coordinate frame | Semantics |
|---|---|---|---|---|
| MediaPipe detection | `(21,3)`, `float32` | **UNRESOLVED**; MediaPipe fields are copied as normalized `x,y,z` | **UNRESOLVED** source camera/MediaPipe frame | MP wrist, thumb 4, index 4, middle 4, ring 4, pinky 4 |
| Vision Pro converted points | `(25,3)`, `float32` | **UNRESOLVED**; translation units are not declared | Adapter-specific signed extraction; physical frame **UNRESOLVED** | One point per streamed finger transform entry; expected 25 entries is not checked here |
| Canonical single frame | `(25,3)`, `float32` | `scale_factor` times source units; default `1.0`; absolute unit **UNRESOLVED** | Wrist-relative; source frame unless offline alignment was pre-applied | Point 0 is zero; four non-thumb palm-root midpoints are synthetic |
| Canonical window | `(3,25,3)`, `float32` | same as frame | same as frame | Chronological deque, oldest to newest |
| Model batch input | `(B,3,25,3)`, `torch.float32` | same as canonical input | same as canonical input | One shared model is called separately for left and right |
| PoseTransformer output | `(B,18)`, `torch.float32` | configured/limited as radians by `ANGLE_LIMITS`; angle-unit declaration is code/config, hardware meaning **UNRESOLVED** | Joint scalar outputs, no spatial frame | Flat L21 angle vector |
| Training padded FK input | `(B,23)`, `torch.float32` | radians for model outputs; FK link lengths use URDF units, physical unit **UNRESOLVED** | L21 URDF graph | 18 model values + five fixed zero tip-node values |
| FK positions | `(B,23,3)`, `torch.float32` | URDF length units, physical unit **UNRESOLVED** | FK/URDF frame; relation to source frame **UNRESOLVED** | Positions for 18 angle/root nodes plus 5 tips |
| Exported angles H5 | `(N,18)`, `float32` | documented as rad; dim 0 fixed zero | L21 alignment attribute | Invalid rows contain held values but `*_valid` remains false |
| Simulation hand DOFs | `(17,)`, `float32` | radians by contract/config | Hardware mapping **UNRESOLVED** | `angle18[1:]` |
| Simulation nodes | `(23,)`, `float32` | same as angle vector | Hardware/simulator mapping **UNRESOLVED** | 18 values plus five zero placeholders |

## Input H5 schema

`retargeting.data.load_twohand_h5` accepts one of:

```text
root/frame_ids                 (N,) optional, default arange(N)
root/timestamps                (N,) optional, default arange(N)
root/left_hand_keypoints       (N,21,3) or (N,25,3)
root/right_hand_keypoints      (N,21,3) or (N,25,3)
```

or exactly one group containing `l_glove_pos` and `r_glove_pos` with the same shapes, plus optional group vectors `frame_ids` and `timestamps`.

Required root attribute when `require_aligned=True` (the default):

```text
coordinate_frame = "l21"
```

The README also documents `coordinate_alignment="source_to_l21_xyz"`, and the alignment script writes it, but `load_twohand_h5` does not validate that attribute. Input numeric dtype is converted to `float32`; finite values are checked per frame later, not by `_validate_hand_array` at load time.

## Training H5 schema

There is no separate target-angle training H5 format in this implementation. Training consumes the same aligned input H5 schema. `TwoHandH5Dataset` creates an in-memory target for each valid frame: `(1,25,3)`, equal to the newest converted frame in the three-frame window. It also creates zero-filled placeholders and boolean masks for invalid sides. The learning target is therefore keypoint geometry, not recorded L21 angles.

## Output angle H5 schema

Written by `retargeting.inference.run`:

```text
frame_ids                 (N,)
timestamps                (N,)
left_angles               (N,18), float32
right_angles              (N,18), float32
left_valid                (N,), bool
right_valid               (N,), bool
```

Attributes include `input_file`, `checkpoint`, `output_shape=(18,)`, `coordinate_alignment`, identity-tracking settings, and `invalid_angle_policy="hold_previous"`. `retargeting.simulation.iter_angle_h5` requires all six datasets, verifies shapes/finiteness, and yields held previous angles for invalid rows.

## 18-angle order

| Dim | Configured name | Configured range | Meaning status |
|---:|---|---:|---|
| 0 | `hand_base_link` | `[0,0]` | Fixed FK-root placeholder; not a wrist free DOF |
| 1 | `index_mcp_roll` | `[-0.18,0.18]` | Code name; hardware axis/sign **UNRESOLVED** |
| 2 | `index_mcp_pitch` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 3 | `index_pip` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 4 | `middle_mcp_roll` | `[-0.18,0.18]` | Code name; hardware axis/sign **UNRESOLVED** |
| 5 | `middle_mcp_pitch` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 6 | `middle_pip` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 7 | `ring_mcp_roll` | `[-0.18,0.18]` | Code name; hardware axis/sign **UNRESOLVED** |
| 8 | `ring_mcp_pitch` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 9 | `ring_pip` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 10 | `pinky_mcp_roll` | `[-0.18,0.18]` | Code name; hardware axis/sign **UNRESOLVED** |
| 11 | `pinky_mcp_pitch` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 12 | `pinky_pip` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 13 | `thumb_cmc_roll` | `[-0.6,0.6]` | Code name; hardware axis/sign **UNRESOLVED** |
| 14 | `thumb_cmc_yaw` | `[0,1.6]` | Code name; hardware axis/sign **UNRESOLVED** |
| 15 | `thumb_cmc_pitch` | `[0,1.0]` | Code name; hardware axis/sign **UNRESOLVED** |
| 16 | `thumb_mcp` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |
| 17 | `thumb_ip` | `[0,1.57]` | Code name; hardware axis/sign **UNRESOLVED** |

The 17 movable DOFs are exactly dims `1..17` in this order. `angle18_to_dofs` implements that drop. Whether these names match the L21 hardware command protocol is **UNRESOLVED**; the repository only proves the internal config/FK order.

## Coordinate and mirror contract

The offline alignment matrix is explicitly `[[0,-1,0],[0,0,1],[-1,0,0]]` applied as `p_source @ M.T`, equivalent to `x'=-y, y'=z, z'=-x`. This is CODE-DEFINED. The physical meaning of source axes, length units, and whether this is the correct L21 frame are **UNRESOLVED**.

Left and right are not proven to be a single mirrored convention: offline alignment is identical for both, while Vision Pro extraction uses opposite signs. The intended left/right mirror rule is **UNRESOLVED**.

