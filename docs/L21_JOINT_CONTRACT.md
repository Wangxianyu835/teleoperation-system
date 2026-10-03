# LinkerHand L21 Joint Contract

This report is derived from the two L21 URDFs in this repository and from the
old SDK sources preserved in Git history. It does not change FK, limits, or
hardware code.

## Status vocabulary

`CODE_DEFINED`, `DOCUMENTED`, `VERIFIED`, and `UNRESOLVED` have the meanings
defined in [`COORDINATE_SYSTEMS.md`](COORDINATE_SYSTEMS.md). Historical SDK
facts are marked `CODE_DEFINED`; they are not treated as an official current
hardware specification.

## `hand_base_link`

Both URDFs use `hand_base_link` as the logical root. Its visual and collision
geometry has `origin xyz=(0,0,0), rpy=(0,0,0)` and mesh
`meshes/hand_base_link.STL` (`CODE_DEFINED`). The URDF does not name the
physical anatomical meaning of its axes. Therefore the physical directions of
+X, +Y, and +Z are `UNRESOLVED`; a mesh visualization with a known pose is
required.

## 17 movable URDF joints

The parent/child graph and limits below are read directly from both URDFs.
Limits are identical left/right. Small origin and rpy differences are CAD
mirror offsets and are included in the difference column.

| Joint | Parent -> child | Left axis | Right axis | Lower, upper (rad in URDF) | Left/right difference |
|---|---|---:|---:|---:|---|
| `index_mcp_roll` | `hand_base_link` -> `index_metacarpals` | `(1,0,0)` | `(1,0,0)` | `-0.18, 0.18` | Mirrored Y origin; rpy X sign flips |
| `index_mcp_pitch` | `index_metacarpals` -> `index_proximal` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Mirrored/small CAD origin and right rpy X offset |
| `index_pip` | `index_proximal` -> `index_middle` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Small origin difference |
| `middle_mcp_roll` | `hand_base_link` -> `middle_metacarpals` | `(1,0,0)` | `(1,0,0)` | `-0.18, 0.18` | Mirrored Y origin; rpy X sign flips |
| `middle_mcp_pitch` | `middle_metacarpals` -> `middle_proximal` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Small origin/rpy difference |
| `middle_pip` | `middle_proximal` -> `middle_middle` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Negligible CAD difference |
| `ring_mcp_roll` | `hand_base_link` -> `ring_metacarpals` | `(1,0,0)` | `(1,0,0)` | `-0.18, 0.18` | Mirrored Y origin; rpy X sign flips |
| `ring_mcp_pitch` | `ring_metacarpals` -> `ring_proximal` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Small origin/rpy difference |
| `ring_pip` | `ring_proximal` -> `ring_middle` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Negligible CAD difference |
| `pinky_mcp_roll` | `hand_base_link` -> `pinky_metacarpals` | `(1,0,0)` | `(1,0,0)` | `-0.18, 0.18` | Mirrored Y origin; rpy X sign flips |
| `pinky_mcp_pitch` | `pinky_metacarpals` -> `pinky_proximal` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Small origin/rpy difference |
| `pinky_pip` | `pinky_proximal` -> `pinky_middle` | `(0,1,0)` | `(0,1,0)` | `0, 1.57` | Negligible CAD difference |
| `thumb_cmc_roll` | `hand_base_link` -> `thumb_metacarpals_base1` | `(-1,0,0)` | `(1,0,0)` | `-0.6, 0.6` | Axis sign flips; mirrored origin/rpy |
| `thumb_cmc_yaw` | `thumb_metacarpals_base1` -> `thumb_metacarpals_base2` | `(0,0,1)` | `(0,0,-1)` | `0, 1.6` | Axis sign flips; origin/rpy differ |
| `thumb_cmc_pitch` | `thumb_metacarpals_base2` -> `thumb_metacarpals` | `(0,1,0)` | `(0,-1,0)` | `0, 1.0` | Axis sign flips; large mirrored rpy change |
| `thumb_mcp` | `thumb_metacarpals` -> `thumb_proximal` | `(0,1,0)` | `(0,-1,0)` | `0, 1.57` | Axis sign flips; origin/rpy differ |
| `thumb_ip` | `thumb_proximal` -> `thumb_distal` | `(0,1,0)` | `(0,-1,0)` | `0, 1.57` | Axis sign flips; origin differs |

The parent, child, axis, and numeric limit values are `CODE_DEFINED`. The
labels “roll”, “pitch”, and “yaw” are repository names; their anatomical
meaning and positive physical direction are `UNRESOLVED` until a visualized
known-pose test is performed.

### Exact joint origins

The following values preserve the URDF `origin xyz` and `origin rpy` records;
they are included so that the mirror differences are auditable rather than
inferred from joint names.

| Joint | Left xyz / rpy | Right xyz / rpy |
|---|---|---|
| `index_mcp_roll` | `0.0059834 -0.032908 0.13338` / `-0.0056645 0 0` | `0.0114834254522634 0.032907936497646 0.133378233122808` / `0.00566453071771697 0 0` |
| `index_mcp_pitch` | `0.002358 -0.00022359 0.018075` / `0 0 0` | `-0.00314196475147223 0.000199976188793055 0.0180749331977774` / `0 -0.0785815229629661 0` |
| `index_pip` | `-0.000158 0 0.044` / `0 0 0` | `-0.000158000000628931 0 0.0440000000952079` / `0 0 0` |
| `middle_mcp_roll` | `0.0059834 -0.010766 0.14075` / `-0.0056645 0 0` | `0.0114840098523855 0.0107658073090937 0.140752927340313` / `0.00566453071771697 0 0` |
| `middle_mcp_pitch` | `0.0023586 -0.0002236 0.018075` / `0 0 0` | `-0.00314196644862088 0.000423561853376581 0.0180749332117821` / `0 -0.078581522962966 0` |
| `middle_pip` | `0.00044403 0 0.043998` / `0 0 0` | `-0.000157999999317671 0 0.0440000000810654` / `0 0 0` |
| `ring_mcp_roll` | `0.0059834 0.010791 0.13313` / `-0.0056645 0 0` | `0.0114828065489108 -0.01079120337128 0.133130689240496` / `0.00566449433579413 0 0` |
| `ring_mcp_pitch` | `0.0023592 -0.0002236 0.018075` / `0 0 0` | `-0.00314018253610867 0.00042356120929779 0.0180749332271929` / `0 -0.078581522962966 0` |
| `ring_pip` | `-0.000158 0 0.044` / `0 0 0` | `-0.00015799999931737 0 0.0440000000810648` / `0 0 0` |
| `pinky_mcp_roll` | `0.0059834 0.032806 0.11807` / `-0.0056645 0 0` | `0.011482977852884 -0.0328063925464449 0.118067803706636` / `0.00566453071771697 0 0` |
| `pinky_mcp_pitch` | `0.0023598 -0.0002236 0.018075` / `0 0 0` | `-0.00313976943770327 0.000423561870027511 0.018075` / `0 -0.0785815229629661 0` |
| `pinky_pip` | `-0.000158 0 0.044` / `0 0 0` | `-0.000158 0 0.044` / `0 0 0` |
| `thumb_cmc_roll` | `0.002384 -0.002009 0.071277` / `-0.0056645 0 0` | `-0.0038161 0.002009 0.071277` / `0.0056645 0 0` |
| `thumb_cmc_yaw` | `0.02725 -0.008383 -0.0024` / `0 0 0.00011487` | `0.03345 0.008383 0.0026` / `0 0 0.023144` |
| `thumb_cmc_pitch` | `0.0063626 -0.01451 0.0040125` / `3.1416 -1.5438 -1.6` | `0.0090013 0.014318 -0.00098753` / `0 0 1.5477` |
| `thumb_mcp` | `-0.0039464 0 0.034242` / `0 0.026973 0` | `0.033763 0.0030005 -0.0066361` / `0 0.078877 0.023144` |
| `thumb_ip` | `-0.0056449 0 0.045952` / `0 -0.12449 0` | `0.04616 0 -0.0035646` / `0 0.079326 0` |

The two base meshes are also unrotated at the link level (`xyz=0 0 0`,
`rpy=0 0 0`, same mesh filename). This is `CODE_DEFINED`; mesh vertex axes
and their anatomical interpretation remain `UNRESOLVED`.

## Internal model correspondence

`angle18[0]` is the fixed root placeholder. `angle18[1:]` follows the 17-joint table above exactly; training appends five fixed tip nodes for FK. This is a code-defined internal correspondence, not a hardware command mapping. Shapes, topology and H5 formats are maintained in [HAND_CONTRACT.md](HAND_CONTRACT.md).

## Hardware SDK evidence and limits

The following evidence is a read-only inspection of historical sources at
Git tag `hand-v1-pre-cleanup`; these adapters are not used by the current hand CLI:

- `legacy/LinkerHand/core/can/linker_hand_l21_can.py`
- `legacy/LinkerHand/core/rml485/linker_hand_l21_485.py`
- `legacy/LinkerHand/utils/mapping.py`
- `legacy/LinkerHand/config/L21_positions.yaml`

The historical CAN class accepts a 25-value high-level pose. Its `joint_map`
creates a 30-value CAN payload, leaving nine payload slots absent/zero and
reordering the 21 values it actually maps. It sends five six-byte groups.
The topic path instead splits the 25 values into five groups of five. The
historical RS-485 class writes ten 8-bit angle registers. These are mutually
inconsistent transport shapes, and neither path consumes the current internal
17D vector.

| Hardware question | Finding | Status |
|---|---|---|
| Real L21 command vector length | Historical code exposes 25 input values for CAN, 30 packed CAN slots, and 10 RS-485 angle registers; no single current contract | `UNRESOLVED` |
| Position order | Historical `joint_map` is the only repository mapping; it is a 25-to-30 transport reorder, not an internal17D mapping. A current 17D mapping is `MAPPING_REQUIRED`. | `UNRESOLVED` |
| Reserved positions | Historical CAN payload contains nine unfilled slots; historical YAML also skips positions in its legacy range helpers | `CODE_DEFINED`; current hardware meaning `UNRESOLVED` |
| Left/right limit differences | Historical YAML gives separate arrays, but the active URDF limits are identical left/right | `CODE_DEFINED`; authoritative current SDK limits `UNRESOLVED` |
| Sign/direction | Historical YAML has side-specific `*_derict` arrays; no link to current internal axes is proven | `CODE_DEFINED`; semantic correspondence `UNRESOLVED` |
| Unit | Active URDF limits are numerically radians by convention; historical RS-485/CAN commands are byte values (0..255), not radians | `CODE_DEFINED`; official current protocol unit `UNRESOLVED` |
| Internal 17D -> hardware command | No mapping exists in current formal code or in the inspected historical adapter; `MAPPING_REQUIRED`. | `UNRESOLVED` |

Do not pass `angle18[1:]` directly to hardware. A protocol-specific mapping,
including side, order, sign, limits, unit conversion, and reserved fields,
must be established from the current LinkerHand SDK/manual and a device test.
