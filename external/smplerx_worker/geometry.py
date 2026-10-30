"""Derive one consistent body skeleton from the upstream SMPL-X forward pass."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def derive_body_arrays(joints, full_pose, parents, cam_trans, focal, princpt,
                       input_body_shape, input_img_shape, bb2img, image_shape):
    joints = np.asarray(joints, dtype=np.float64)
    pose = np.asarray(full_pose, dtype=np.float64).reshape(-1, 3)
    if joints.shape != (len(parents), 3) or len(pose) != len(parents):
        raise ValueError('SMPL-X joint/pose/parent counts do not agree')
    local = Rotation.from_rotvec(pose).as_matrix()
    global_rotations = np.empty_like(local)
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= index:
            raise ValueError('invalid SMPL-X parent topology')
        global_rotations[index] = local[index] if parent == -1 else global_rotations[parent] @ local[index]
    camera_points = joints + np.asarray(cam_trans).reshape(1, 3)
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError('SMPL-X skeleton projects behind the camera')
    # Matches SMPLer_X.Model.get_coord; then scale body pixels to the input
    # crop and invert preprocessing.generate_patch_image's affine transform.
    xy = camera_points[:, :2] / (camera_points[:, 2:] + 1e-4)
    xy = xy * np.asarray(focal) + np.asarray(princpt)
    xy *= np.array([input_img_shape[1] / input_body_shape[1],
                    input_img_shape[0] / input_body_shape[0]])
    original_xy = np.c_[xy, np.ones(len(xy))] @ np.asarray(bb2img).T
    projected = np.zeros_like(joints)
    projected[:, :2] = original_xy / np.array([image_shape[1], image_shape[0]])
    return {'joints_root_m': (joints - joints[0]).astype(np.float32),
            'global_rotations': global_rotations.astype(np.float32),
            'image_joints': projected.astype(np.float32)}
