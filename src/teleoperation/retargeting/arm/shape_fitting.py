"""Fit body shape against explicitly aligned robot reference points."""
from dataclasses import dataclass
import numpy as np
from .body_model import reference_pose

@dataclass(frozen=True)
class ShapeFitResult:
    betas: np.ndarray
    initial_rms: float
    final_rms: float
    per_point_error: np.ndarray
    iterations: int

    def to_dict(self):
        return dict(betas=self.betas.tolist(),initial_rms=self.initial_rms,final_rms=self.final_rms,
                    per_point_error=self.per_point_error.tolist(),iterations=self.iterations)


def fit_shape(body_model,target_points,joint_indices,steps=1000,learning_rate=.05,pose_params=None,anchor_indices=None):
    import torch
    layer=body_model.layer
    target=np.asarray(target_points,dtype=float)
    if target.shape!=(len(joint_indices),3) or not len(target) or not np.isfinite(target).all():
        raise ValueError('target_points must be finite (K,3) matching joint_indices')
    if steps<1 or learning_rate<=0: raise ValueError('steps and learning_rate must be positive')
    if any(i<0 or i>=len(layer.parents) for i in joint_indices): raise ValueError('joint index out of bounds')
    if anchor_indices is not None and (not len(anchor_indices) or any(i<0 or i>=len(layer.parents) for i in anchor_indices)):
        raise ValueError('anchor_indices must contain native joint indices')
    target=torch.as_tensor(target,dtype=layer.shapedirs.dtype,device=body_model.device)
    beta=torch.zeros((1,body_model.spec.num_betas),dtype=target.dtype,device=target.device,requires_grad=True)
    optimizer=torch.optim.Adam([beta],lr=learning_rate)
    pose=reference_pose() if pose_params is None else pose_params
    def errors():
        joints=body_model.joints_tensor(beta,pose)[0]
        if anchor_indices is not None: joints=joints-joints[list(anchor_indices)].mean(0)
        return (joints[list(joint_indices)]-target).square().sum(-1)
    with torch.no_grad(): initial=float(errors().mean().sqrt()); best_loss=initial**2;best=beta.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad();loss=errors().mean();loss.backward();optimizer.step()
        with torch.no_grad():
            current=float(errors().mean())
            if current<best_loss: best_loss=current;best=beta.detach().clone()
    with torch.no_grad():
        beta.copy_(best);per_point=errors().sqrt().cpu().numpy()
    return ShapeFitResult(best[0].cpu().numpy(),initial,float(np.sqrt(np.mean(per_point**2))),per_point,int(steps))
