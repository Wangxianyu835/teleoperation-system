"""Six-term losses for target-native hand training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_NAMES = ("vec", "pos", "collision", "thumb", "tip_distance", "thumb2")
LOSS_WEIGHTS = (500, 500, 10, 500, 0.05, 500)
COLLISION_THRESHOLD_M = 0.010

# Chains run from the palm toward the tip; tips come from the target HandSpec.
_HAND_LAYOUTS = {
    "l21": {
        "prefix": {"left": "", "right": ""},
        "thumb": ("thumb_metacarpals_base1", "thumb_metacarpals_base2",
                  "thumb_metacarpals", "thumb_proximal", "thumb_distal"),
        "finger": ("{finger}_metacarpals", "{finger}_proximal", "{finger}_middle"),
        "thumb_angle": (2, 3, 4),
        "thumb_exclusions": ((1, 2),),
        "finger_exclusions": ((0, 1),),
    },
    "h1_2": {
        "prefix": {"left": "L_", "right": "R_"},
        "thumb": ("thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal"),
        "finger": ("{finger}_proximal", "{finger}_intermediate"),
        "thumb_angle": (1, 2, 3),
        "thumb_exclusions": ((0, 1), (1, 2), (2, 3), (3, 4)),
        "finger_exclusions": ((0, 1), (1, 2)),
    },
    "gr1_t2": {
        "prefix": {"left": "L_", "right": "R_"},
        "thumb": ("thumb_proximal_base", "thumb_proximal", "thumb_distal"),
        "finger": ("{finger}_proximal", "{finger}_intermediate"),
        "thumb_angle": (1, 2, 3),
        "thumb_exclusions": ((0, 1), (1, 2), (2, 3)),
        "finger_exclusions": ((0, 1), (1, 2)),
    },
    "g1": {
        "prefix": {"left": "left_hand_", "right": "right_hand_"},
        "thumb": ("thumb_0_link", "thumb_1_link", "thumb_2_link"),
        "finger": ("{finger}_0_link", "{finger}_1_link"),
        "thumb_angle": (0, 1, 2),
        "thumb_exclusions": ((0, 1), (1, 2), (2, 3)),
        "finger_exclusions": ((0, 1), (1, 2)),
    },
}


def hand_loss_geometry(fk):
    """Resolve loss points once, before creating a training run directory."""
    spec = fk.spec
    layout = _HAND_LAYOUTS[spec.hand_model]
    count = len(spec.tip_frames)
    if tuple(spec.human_tip_indices) != (4, 9, 14, 19, 24)[:count] or count not in (3, 5):
        raise ValueError("Six-term losses require ordered thumb/index/middle task points")
    prefix = layout["prefix"][spec.side]
    frames, chains, excluded = [spec.root_link], [], []
    for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")[:count]):
        templates = layout["thumb"] if i == 0 else layout["finger"]
        names = [prefix + name.format(finger=finger) for name in templates]
        names.append(spec.tip_frames[i])
        chain = list(range(len(frames), len(frames) + len(names)))
        frames.extend(names)
        chains.append(chain)
        pairs = layout["thumb_exclusions"] if i == 0 else layout["finger_exclusions"]
        excluded.extend([[chain[a], chain[b]] for a, b in pairs])
    available = {spec.root_link, *(joint.child for joint in fk.tree)}
    if len(set(frames)) != len(frames) or set(frames) - available:
        raise ValueError(f"Invalid six-term geometry for {spec.hand_model}/{spec.side}")
    return {
        "frames": frames,
        "robot": {
            "MCP_dic": [chains[0][1], *(chain[0] for chain in chains[1:])],
            "DIP_dic": [chain[-2] for chain in chains],
            "TIP_dic": [chain[-1] for chain in chains],
            "palm": [0, chains[1][0], chains[2][0]],
            "thumb_angle": [chains[0][i] for i in layout["thumb_angle"]],
        },
        "source": {
            "MCP_dic": [1, 6, 11, 16, 21][:count],
            "PIP_dic": [2, 7, 12, 17, 22][:count],
            "TIP_dic": list(spec.human_tip_indices),
            "palm": [0, 6, 11],
            "thumb_angle": [2, 3, 4],
        },
        "collision_excluded_pairs": excluded,
    }


def hand_loss(predicted_angle, source_3D, fk, geometry):
    """Compare normalized human hand25 points with native robot geometry in metres."""
    source = torch.as_tensor(source_3D, dtype=predicted_angle.dtype, device=predicted_angle.device)
    if source.shape != (predicted_angle.shape[0], 25, 3) or not torch.isfinite(source).all():
        raise ValueError("Six-term losses require finite batched hand25 targets")
    basis = torch.as_tensor(fk.spec.native_to_canonical, dtype=source.dtype, device=source.device)
    source = ((source - source[:, :1]) @ basis.T) * fk.spec.reference_palm_length_m
    robot = fk.positions(predicted_angle, tuple(geometry["frames"]))
    rb_dic, source_dic = geometry["robot"], geometry["source"]
    criterion = nn.MSELoss()
    collision = CollisionLoss(COLLISION_THRESHOLD_M, rb_dic, excluded_points=[0],
                              excluded_pairs=geometry["collision_excluded_pairs"])
    values = (
        vec_inter_loss(robot, source, criterion, rb_dic, source_dic),
        tip_pos_loss(robot, source, criterion, rb_dic, source_dic),
        collision(robot),
        thumb_loss(robot, source, criterion, rb_dic, source_dic),
        tip_distance_loss(robot, source, criterion, rb_dic, source_dic),
        thumb_loss2(robot, source, criterion, rb_dic, source_dic),
    )
    weighted = {name: value * weight for name, value, weight in zip(LOSS_NAMES, values, LOSS_WEIGHTS)}
    return sum(weighted.values()), weighted


def vec_inter_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    target = target_3D[:, rb_dic["DIP_dic"][1:], :] - target_3D[:, rb_dic["MCP_dic"][1:], :]
    source = source_3D[:, source_dic["PIP_dic"][1:], :] - source_3D[:, source_dic["MCP_dic"][1:], :]
    return loss_function(F.normalize(source, dim=-1), F.normalize(target, dim=-1))


def thumb_loss(target_3D, source_3D, loss_function, rb_dic, source_dic):
    source_root, source_index, source_middle = source_dic["palm"]
    target_root, target_index, target_middle = rb_dic["palm"]
    source_normal = torch.cross(source_3D[:, source_middle] - source_3D[:, source_root], source_3D[:, source_index] - source_3D[:, source_root], dim=-1)
    target_normal = torch.cross(target_3D[:, target_middle] - target_3D[:, target_root], target_3D[:, target_index] - target_3D[:, target_root], dim=-1)
    source_distance = point_plane_distance_batch(source_3D[:, source_dic["TIP_dic"][0]], source_3D[:, source_root], source_normal)
    target_distance = point_plane_distance_batch(target_3D[:, rb_dic["TIP_dic"][0]], target_3D[:, target_root], target_normal)
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
    target_a, target_b, target_c = rb_dic["thumb_angle"]
    source_a, source_b, source_c = source_dic["thumb_angle"]
    target_segments = torch.stack((target_3D[:, target_c] - target_3D[:, target_b],
                                   target_3D[:, target_b] - target_3D[:, target_a]), dim=1)
    source_segments = torch.stack((source_3D[:, source_c] - source_3D[:, source_b],
                                   source_3D[:, source_b] - source_3D[:, source_a]), dim=1)
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
    count = len(source_tip)
    if count != len(target_tip) or count < 2:
        raise ValueError("Tip-distance loss requires matching finger sets")
    pairs = [(0, i) for i in range(1, count)] + [(i, i + 1) for i in range(1, count - 1)]
    total = target_3D.new_tensor(0.0)
    for first, second in pairs:
        target_distance = torch.norm(target_3D[:, target_tip[first]] - target_3D[:, target_tip[second]], dim=-1) * 1000.0
        source_distance = torch.norm(source_3D[:, source_tip[first]] - source_3D[:, source_tip[second]], dim=-1) * 1000.0
        total = total + loss_function(source_distance, target_distance)
    # MSELoss already averages over the batch; only average the finger pairs.
    return total / len(pairs)
