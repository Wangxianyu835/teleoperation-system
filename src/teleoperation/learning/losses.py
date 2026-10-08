"""Losses used by the two-hand LinkerHand L21 training pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def hand_loss(predicted_angle, source_3D, rb_dic, source_dic,
              pos_loss_function, vec_loss_function, col_loss_function,
              reg_loss_function=None, visualizer=None, hand_fk_model=None,
              logger=None, loss_weight=None, hand_side="right"):
    """Compute the six losses used by the current L21 training loop."""
    del vec_loss_function, reg_loss_function, visualizer, logger
    if hand_fk_model is None:
        raise ValueError("hand_fk_model is required")
    if loss_weight is None or len(loss_weight) != 6:
        raise ValueError("loss_weight must contain six values")
    source_3D = source_3D.to(predicted_angle.device)
    batch, _, joints, coords = source_3D.shape
    source_3D = source_3D.view(batch, joints, coords)
    _, _, robot = hand_fk_model.forward(predicted_angle)
    values = [
        vec_inter_loss(robot, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[0],
        tip_pos_loss(robot, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[1],
        col_loss_function(robot) * loss_weight[2] if loss_weight[2] > 0 else predicted_angle.new_tensor(0.0),
        thumb_loss(robot, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[3] if loss_weight[3] > 0 else predicted_angle.new_tensor(0.0),
        tip_distance_loss(robot, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[4],
        thumb_loss2(robot, source_3D, pos_loss_function, rb_dic, source_dic) * loss_weight[5],
    ]
    return (sum(values), *values)


def vec_inter_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    target = target_3D[:, rb_dic["DIP_dic"][1:], :] - target_3D[:, rb_dic["MCP_dic"][1:], :]
    source = source_3D[:, source_dic["PIP_dic"][1:], :] - source_3D[:, source_dic["MCP_dic"][1:], :]
    return loss_function(F.normalize(source, dim=-1), F.normalize(target, dim=-1))


def thumb_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    source_normal = torch.cross(source_3D[:, 21] - source_3D[:, 0], source_3D[:, 6] - source_3D[:, 0], dim=-1)
    target_normal = torch.cross(target_3D[:, 10] - target_3D[:, 0], target_3D[:, 1] - target_3D[:, 0], dim=-1)
    source_distance = point_plane_distance_batch(source_3D[:, source_dic["TIP_dic"][0]], source_3D[:, 0], source_normal)
    target_distance = point_plane_distance_batch(target_3D[:, rb_dic["TIP_dic"][0]], target_3D[:, 0], target_normal)
    return loss_function(source_distance, target_distance * 0.9)


def point_plane_distance_batch(points_outside, points_on_plane, normals):
    normals = F.normalize(normals, dim=-1)
    return torch.abs(torch.sum((points_outside - points_on_plane) * normals, dim=-1))


def thumb_loss2(target_3D, source_3D, loss_function, rb_dic, source_dic):
    """Compare thumb angles only where all four segments are resolvable.

    The existing normalization epsilon (1e-12 in coordinate units) is a
    numerical cutoff, not a physical length limit. Zero or shorter segments
    have no reliable direction and are excluded from this term's reduction.
    An entirely masked batch contributes a differentiable zero, not an angle.
    """
    del rb_dic, source_dic
    target_segments = torch.stack((target_3D[:, 17] - target_3D[:, 16],
                                   target_3D[:, 16] - target_3D[:, 15]), dim=1)
    source_segments = torch.stack((source_3D[:, 4] - source_3D[:, 3],
                                   source_3D[:, 3] - source_3D[:, 2]), dim=1)
    if not (torch.isfinite(target_segments).all() and torch.isfinite(source_segments).all()):
        raise ValueError("Thumb-angle segments must be finite")
    norm_eps = 1e-12
    valid = ((torch.linalg.vector_norm(target_segments, dim=-1) > norm_eps).all(dim=1)
             & (torch.linalg.vector_norm(source_segments, dim=-1) > norm_eps).all(dim=1))
    if not valid.any():
        return (target_segments * 0.0).sum() + (source_segments * 0.0).sum()

    angles = []
    for segments in (target_segments, source_segments):
        a, b = F.normalize(segments[valid], dim=-1, eps=norm_eps).unbind(dim=1)
        # atan2 preserves 0/pi without acos's singular derivative at +/-1.
        sine = torch.linalg.vector_norm(torch.cross(a, b, dim=-1), dim=-1)
        cosine = torch.sum(a * b, dim=-1)
        angles.append(torch.atan2(sine, cosine))
    return loss_function(angles[1], angles[0])


def tip_pos_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    target = target_3D[:, rb_dic["TIP_dic"][1:], :] - target_3D[:, rb_dic["MCP_dic"][1:], :]
    source = source_3D[:, source_dic["TIP_dic"][1:], :] - source_3D[:, source_dic["MCP_dic"][1:], :]
    return loss_function(F.normalize(source, dim=-1), F.normalize(target, dim=-1))


class CollisionLoss(nn.Module):
    def __init__(self, threshold, rb_dic, excluded_points=None,
                 excluded_pairs=None, mode="sphere-sphere", hand_type="right"):
        super().__init__()
        self.threshold = threshold
        self.rb_dic = rb_dic
        self.excluded_points = excluded_points or [0]
        self.excluded_pairs = excluded_pairs or []
        self.mode = mode
        self.hand_type = hand_type

    def forward(self, positions):
        count = positions.shape[1]
        distances = torch.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=-1)
        mask = (~torch.eye(count, dtype=torch.bool, device=positions.device)).unsqueeze(0)
        mask = mask.expand(positions.shape[0], -1, -1).clone()
        for point in self.excluded_points:
            if point < count:
                mask[:, point, :] = False
                mask[:, :, point] = False
        for first, second in self.excluded_pairs:
            if first < count and second < count:
                mask[:, first, second] = False
                mask[:, second, first] = False
        collisions = (distances < self.threshold) & mask
        if collisions.any():
            normalized = distances[collisions] / self.threshold
            loss = torch.mean(torch.exp(-(normalized ** 2)))
        else:
            loss = positions.new_tensor(0.0)
        return loss + 1e-6


def tip_distance_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    source_tip = source_dic["TIP_dic"]
    target_tip = rb_dic["TIP_dic"]
    pairs = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4))
    total = target_3D.new_tensor(0.0)
    for first, second in pairs:
        target_distance = torch.norm(target_3D[:, target_tip[first]] - target_3D[:, target_tip[second]], dim=-1) * 1000.0
        source_distance = torch.norm(source_3D[:, source_tip[first]] - source_3D[:, source_tip[second]], dim=-1) * 1000.0
        total = total + loss_function(source_distance, target_distance)
    # MSELoss already averages over the batch; only average the finger pairs.
    return total / len(pairs)


class RegLoss(nn.Module):
    def forward(self, value):
        batch_size = value.shape[0]
        return torch.mean(torch.norm(value.view(batch_size, -1), dim=1).pow(2))
