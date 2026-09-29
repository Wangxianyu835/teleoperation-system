# Hand-only L21 Pipeline (as implemented)

This document follows one hand frame from source input to an L21 angle vector. It describes the current code; it is not a proposed redesign.

## Executive path

```text
MediaPipe HandLandmarker result.hand_landmarks (21,3)
  -> retargeting.mediapipe.MediaPipeCameraAdapter.next_input
  -> HandIdentityTracker.update_detections
  -> HandWindowBuffer.update
  -> ensure_hand25 / mediapipe21_to_hand25 / wrist_relative
  -> (3,25,3) canonical payload
  -> TwoHandRetargeter.predict
  -> PoseTransformer.forward
  -> (18,) angle vector
  -> offline H5 validation/hold policy OR realtime simulator adapter
```

There are two implemented ingestion branches. They share the 25-point conversion and windowing code, but they do not currently share all coordinate and identity stages.

## Realtime MediaPipe branch

1. `retargeting/mediapipe.py:106-137` reads one BGR camera frame, converts it to RGB, calls MediaPipe `HandLandmarker.detect_for_video`, and copies each detected landmark's normalized `x,y,z` into a `float32` array of shape `(21,3)`. The MediaPipe handedness category is used as the initial `left`/`right` label.
2. `retargeting/mediapipe.py:139` calls `HandIdentityTracker.update_detections`. A missing or rejected side is returned as `None`; `:141-143` then clears only that side's `HandWindowBuffer`.
3. `retargeting/tracking.py:389-436` appends each present hand to a side-specific deque. `ensure_hand25` (`:240-270`) dispatches 21-point data to `mediapipe21_to_hand25` (`:273-333`) and 25-point data directly to `wrist_relative`.
4. The resulting side window is stacked in chronological order as `(3,25,3)` (`tracking.py:418-436`) and wrapped by `build_retarget_input` (`contracts.py:104-135`). A side can remain `None` while the other side has a complete window.
5. A caller passes that payload to `TwoHandRetargeter.predict` (`retargeting/model.py:57-77`). Each non-`None` side is converted to a tensor `(1,3,25,3)` (`:100-101`) and sent through the shared model.

MediaPipe coordinates are not passed through `retargeting.coordinates.align_source_hand_coordinates` in this realtime path. Their physical frame and units are therefore not established by this code.

## Vision Pro branch

`retargeting/visionpro.py:26-48` reads one side from `VisionProStreamer`, converts each 4x4 transform translation to 25 points with `visionpro_fingers_to_points` (`:51-69`), and sends the points to the same `HandWindowBuffer`. The function uses different signed axis formulas for left and right hands. It does not use `HandIdentityTracker` because the adapter is configured for one side. Window conversion and wrist-relative processing then follow the MediaPipe branch.

## Offline H5 branch (training and export)

1. `retargeting/data.py:241-324` loads either root datasets `left_hand_keypoints`/`right_hand_keypoints` or one legacy group containing `l_glove_pos`/`r_glove_pos`. It requires root attribute `coordinate_frame == "l21"` by default (`:250-260`). `coordinate_alignment` is not required by the loader.
2. `TwoHandH5Dataset._build_samples` (`data.py:92-182`) optionally runs `HandIdentityTracker.update` (`:120-126`), clears a side buffer for invalid/zero frames (`:128-145`), converts each valid frame with `ensure_hand25`, and emits a sample once three consecutive frames are available. Invalid sides receive zero input/target arrays and `*_valid=False` (`:147-180`).
3. For each batch, `TwoHandH5ChunkedGenerator.next_epoch` returns `*_input` with shape `(B,3,25,3)`, `*_target` with shape `(B,1,25,3)`, and boolean side masks (`data.py:189-238`).
4. Training (`retargeting/training.py:404-495`) calls the same `PoseTransformer`, pads its 18 outputs with five zeros (`:417-422`) to the 23-node FK graph, and computes losses only for valid side rows. Inference (`retargeting/inference.py:62-84`) calls the model on valid windows, checks finite values and `ANGLE_LIMITS`, and writes accepted predictions at the target frame index.
5. `retargeting/inference.py:86-104` applies `_hold_last_valid_angles` (`:133-140`) independently per side. Leading invalid frames stay the initialized all-zero vector; later invalid frames copy the last valid 18-vector. The validity bit remains false.

`scripts/align_h5_coordinates.py` is a preprocessing step, not part of `TwoHandH5Dataset`: it first calls `ensure_hand25`, then applies `align_source_hand_coordinates` to both sides and writes `coordinate_frame=l21` and `coordinate_alignment=source_to_l21_xyz`.

## 21 -> 25 topology

`tracking.py:273-333` keeps MediaPipe indices 0-4 unchanged and inserts four midpoint points before the four non-thumb fingers:

| Canonical index | Meaning | Construction |
|---:|---|---|
| 0 | wrist | MediaPipe 0 |
| 1-4 | thumb MCP/PIP/DIP/TIP | MediaPipe 1-4 |
| 5 | index palm-root | `(MP0 + MP5) / 2` |
| 6-9 | index MCP/PIP/DIP/TIP | MediaPipe 5-8 |
| 10 | middle palm-root | `(MP0 + MP9) / 2` |
| 11-14 | middle MCP/PIP/DIP/TIP | MediaPipe 9-12 |
| 15 | ring palm-root | `(MP0 + MP13) / 2` |
| 16-19 | ring MCP/PIP/DIP/TIP | MediaPipe 13-16 |
| 20 | pinky palm-root | `(MP0 + MP17) / 2` |
| 21-24 | pinky MCP/PIP/DIP/TIP | MediaPipe 17-20 |

`wrist_relative` (`tracking.py:218-237`) subtracts point 0 from every point and multiplies by `scale_factor`. For 21-point input this happens after insertion; for 25-point input it happens directly. The wrist row therefore becomes exactly zero.

## Coordinate transformations and symmetry

- Offline preprocessing uses the row-vector matrix in `retargeting/coordinates.py:9-18`: `x'=-y`, `y'=z`, `z'=-x`. It is applied identically to both sides.
- MediaPipe realtime performs no explicit source-to-L21 matrix transform.
- Vision Pro performs side-dependent extraction in `visionpro.py:56-66`: left `[m[1,3], -m[2,3], m[0,3]]`; right `[-m[1,3], m[2,3], -m[0,3]]`.
- The code therefore does not establish a completely symmetric left/right path. Whether the Vision Pro signs are the intended mirror rule is `UNRESOLVED`.

## Identity continuity and reset

`HandIdentityTracker.assign` (`tracking.py:81-173`) evaluates every candidate against each previous side using:

- palm-center displacement (`_palm_center`, `:208-215`) with indices `(0,5,9,13,17)` for 21 points or `(0,6,11,16,21)` for 25 points;
- wrist-relative landmark RMSE (`_continuity_metrics`, `:193-205`).

Both must be within `0.08` and `0.05` by default. It enumerates one-to-one assignments, maximizes accepted matches, then label matches, then minimizes normalized cost (`:124-167`). MediaPipe labels are only a tie-break/initialization aid; temporal continuity is authoritative.

After assignment, `_previous` is replaced by the current assignment (`:74-79`), including `None`. A rejected jump therefore clears the tracker state for that side. The MediaPipe adapter additionally clears that side's three-frame deque (`mediapipe.py:141-143`); the offline dataset clears the side deque when its assigned frame is invalid (`data.py:128-145`). Recovery requires three new consecutive valid frames.

## Model and FK

`PoseTransformer` is configured by `L21.model_kwargs()` (`retargeting/config.py:118-136`) with `num_frame=3`, `in_num_joints=25`, `in_chans=3`, and `out_num_joint=18`. Its actual public input is `(B,3,25,3)` (`model/pose_transformer.py:256-288`), internally permuted to `(B,3,3,25)`, spatially encoded per frame, temporally encoded, reduced by a learned `Conv1d(num_frame -> 1)`, and returned as `(B,18)`.

The three-frame window is required by the configured temporal receptive field and temporal positional embedding; the learned weighted mean combines all three frame features. The code comments call the result a center-frame feature, but the convolution is learned and is not hard-coded to the center frame.

During training only, `hand_loss` calls `hand_fk_model.forward(predicted_angle)` (`model/losses.py:19-31`). Training pads 18 angles to 23 FK nodes before this call. FK converts angles and URDF offsets/axes into `(B,23,3)` positions and rotations (`model/kinematics.py:47-158`), which are consumed by the geometric and collision losses. Inference does not run FK; it only applies finite/range checks. Simulation only drops the fixed angle or appends five zero nodes (`retargeting/simulation.py:25-33`).

The six configured loss terms (`model/losses.py:23-30`) are:

- `vec_inter_loss`: normalized source PIP-to-MCP directions versus robot DIP-to-MCP directions for the four non-thumb fingers;
- `tip_pos_loss`: normalized source MCP-to-tip directions versus robot MCP-to-tip directions;
- `CollisionLoss`: penalizes robot point pairs closer than `collision_threshold`, excluding the root and configured adjacent-link pairs;
- `thumb_loss`: matches thumb-tip distance from a source palm plane to the robot thumb-tip distance (with a `0.9` target scale in code);
- `tip_distance_loss`: matches seven pairwise fingertip distances, multiplied by `1000` inside the loss;
- `thumb_loss2`: matches the angle between two adjacent thumb segment vectors.

`vec_loss_function` is deleted/unused by `hand_loss` (the vector terms use `pos_loss_function`), and `reg_loss_function` is accepted but deleted/unused. Thus the six returned values are not six independently wired generic objectives despite the legacy parameter names.

## L21 angle output

The model outputs 18 values. `ANGLE_LIMITS[0] == (0,0)` makes dim 0 fixed zero; it is the FK root placeholder, not a wrist DOF. `retargeting/simulation.angle18_to_dofs` drops dim 0 and returns 17 values. The configured order is in `retargeting/config.py:38-58` and is reproduced in `HAND_CONTRACT.md`.
