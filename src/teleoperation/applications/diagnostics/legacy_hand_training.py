"""Read-only reference geometry for auditing historical L21 checkpoint losses.

No training entry point lives here. New target-specific training uses HandSpec
and DifferentiableHandModel; these reference axes retain old model semantics.
"""
import torch

from teleoperation.retargeting.hand.config import L21
from teleoperation.retargeting.hand.kinematics import create_hand_kinematics

def create_legacy_hand_fks(device, hand_sides=('left','right'), *, canonical=False):
    if canonical and tuple(hand_sides) != ('right',):
        raise ValueError('Canonical historical checkpoints use right robot FK only')
    fks = {side:create_hand_kinematics(getattr(L21,f'{side}_urdf'),L21.hand_kinematics_config(),device,
                                      scale_factor=L21.training.robot_scale,axis_correction_matrix=None)
           for side in hand_sides}
    if canonical:
        from teleoperation.retargeting.hand.coordinates import build_palm_basis, ROBOT_PALM_INDICES, OPERATOR_TO_CANONICAL
        fk = fks['right']
        with torch.no_grad():
            points = fk.forward(torch.zeros((1,23),device=device))[2][0].cpu().numpy()
        matrix = build_palm_basis(points,ROBOT_PALM_INDICES) @ OPERATOR_TO_CANONICAL['right']
        fk.axis_correction_matrix = torch.as_tensor(matrix.T,device=device)
    return fks
