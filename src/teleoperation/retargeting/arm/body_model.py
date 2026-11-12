"""Explicit SMPL-family asset identity and differentiable native joint geometry."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import importlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation


def reference_pose():
    """Upright reference used for fitting; runtime poses must supply their own root."""
    return {'global_orient': Rotation.from_euler('xyz', [np.pi/2, 0, np.pi/2]).as_rotvec()}


@dataclass(frozen=True)
class BodyModelSpec:
    model_path: str
    model_type: str = 'smplx'
    gender: str = 'neutral'
    num_betas: int = 20
    module_dir: str | None = None

    def __post_init__(self):
        if self.model_type not in ('smpl', 'smplh', 'smplx'):
            raise ValueError('model_type must be smpl, smplh or smplx')
        if self.gender not in ('neutral', 'male', 'female'):
            raise ValueError('unsupported gender')
        if not isinstance(self.num_betas, int) or isinstance(self.num_betas, bool) or self.num_betas < 1:
            raise ValueError('num_betas must be a positive integer')

    @property
    def asset_path(self):
        path = Path(self.model_path).expanduser().resolve()
        if path.is_file():
            return path
        folder = path / self.model_type if (path / self.model_type).is_dir() else path
        candidates = [folder / f'{self.model_type.upper()}_{self.gender.upper()}.{ext}' for ext in ('npz','pkl')]
        if self.model_type == 'smpl':
            name = {'neutral':'neutral','male':'m','female':'f'}[self.gender]
            candidates.append(folder / f'basicmodel_{name}_lbs_10_207_0_v1.1.0.pkl')
        for candidate in candidates:
            if candidate.is_file(): return candidate
        raise FileNotFoundError(f'No {self.model_type} {self.gender} asset in {path}')

    def to_dict(self):
        values = asdict(self)
        values['model_path'] = str(self.asset_path)
        return values

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    @property
    def hash(self):
        digest = hashlib.sha256()
        with self.asset_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
        # Installation locations do not change the anatomical identity.
        digest.update(json.dumps({'model_type':self.model_type,'gender':self.gender,'num_betas':self.num_betas},sort_keys=True).encode())
        return digest.hexdigest()


class BodyModel:
    def __init__(self, spec: BodyModelSpec, device='cpu'):
        self.spec, self.device, self._layer = spec, device, None

    @property
    def layer(self):
        if self._layer is None:
            import torch
            if self.spec.module_dir:
                module_dir = str(Path(self.spec.module_dir).resolve())
                if module_dir not in sys.path: sys.path.insert(0, module_dir)
            smplx = importlib.import_module('smplx')
            path = self.spec.asset_path
            # Check raw capacity before the vendor's constructor can silently clip it.
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=True) as data: capacity=data['shapedirs'].shape[-1]
            else:
                import pickle
                with path.open('rb') as stream: data=pickle.load(stream,encoding='latin1')
                capacity=data['shapedirs'].shape[-1]
            capacity=min(capacity,300)
            if self.spec.num_betas > capacity:
                raise ValueError(f'Requested {self.spec.num_betas} betas but asset supports {capacity}')
            cls={'smpl':'SMPL','smplh':'SMPLH','smplx':'SMPLX'}[self.spec.model_type]
            layer=getattr(smplx,cls)(str(path),gender=self.spec.gender,num_betas=self.spec.num_betas,
                                    use_pca=False,flat_hand_mean=True,ext=path.suffix[1:]).to(self.device)
            if layer.shapedirs.shape[-1] != self.spec.num_betas:
                raise ValueError(f'Body model silently changed beta dimension to {layer.shapedirs.shape[-1]}; requested {self.spec.num_betas}')
            layer.eval()
            for parameter in layer.parameters(): parameter.requires_grad_(False)
            # The model is fixed after construction. Project dense mesh shape
            # directions to native joints once, retaining beta gradients in FK.
            with torch.no_grad():
                self._joint_template=torch.einsum('jv,vc->jc',layer.J_regressor,layer.v_template)
                self._joint_directions=torch.einsum('jv,vcb->jcb',layer.J_regressor,layer.shapedirs)
            self._layer=layer
        return self._layer

    def _local_pose(self, pose_params, batch, dtype, device):
        import torch
        layer=self.layer
        count=len(layer.parents)
        pose=torch.zeros((batch,count,3),dtype=dtype,device=device)
        aliases={'global_orient':('smplx_root_pose','root_pose'), 'body_pose':('smplx_body_pose',),
                 'left_hand_pose':('smplx_lhand_pose',), 'right_hand_pose':('smplx_rhand_pose',),
                 'jaw_pose':('smplx_jaw_pose',), 'leye_pose':(), 'reye_pose':()}
        body_count=23 if self.spec.model_type=='smpl' else 21
        sections={'global_orient':(0,1),'body_pose':(1,body_count)}
        if self.spec.model_type=='smplx': sections.update(jaw_pose=(22,1),leye_pose=(23,1),reye_pose=(24,1),left_hand_pose=(25,15),right_hand_pose=(40,15))
        elif self.spec.model_type=='smplh': sections.update(left_hand_pose=(22,15),right_hand_pose=(37,15))
        for name,(offset,size) in sections.items():
            key=next((key for key in (name,*aliases.get(name,())) if key in pose_params),None)
            if key is None: continue
            value=torch.as_tensor(pose_params[key],dtype=dtype,device=device)
            if value.numel()==size*3: value=value.reshape(1,size,3).expand(batch,-1,-1)
            elif value.numel()==batch*size*3: value=value.reshape(batch,size,3)
            else: raise ValueError(f'{key} must contain {size*3} values per batch')
            if not torch.isfinite(value).all(): raise ValueError(f'{key} must be finite')
            pose[:,offset:offset+size]=value
        return pose

    def _geometry_tensor(self, betas, pose_params):
        import torch
        from smplx.lbs import batch_rodrigues
        layer=self.layer
        if betas.ndim!=2 or betas.shape[1]!=self.spec.num_betas or not torch.isfinite(betas).all():
            raise ValueError(f'betas must be finite (batch,{self.spec.num_betas}) without truncation or padding')
        # Native SMPL joints depend linearly on shape before articulated FK.
        # Keep each morphology's reduction order independent of how many robots
        # share this update. Tiny batch-dependent float32 differences can grow
        # into different branches of a redundant arm's iterative IK trajectory.
        rest=self._joint_template[None]+torch.stack([
            torch.einsum('b,jcb->jc',shape,self._joint_directions) for shape in betas])
        local=batch_rodrigues(self._local_pose(pose_params,len(betas),betas.dtype,betas.device).reshape(-1,3)).reshape(len(betas),-1,3,3)
        positions,rotations=[],[]
        for index,parent in enumerate(layer.parents.tolist()):
            if parent<0: positions.append(torch.zeros_like(rest[:,0]));rotations.append(local[:,index])
            else:
                rotations.append(rotations[parent]@local[:,index])
                positions.append(positions[parent]+(rotations[parent]@(rest[:,index]-rest[:,parent])[...,None])[...,0])
        return torch.stack(positions,1),torch.stack(rotations,1)

    def joints_tensor(self, betas, pose_params=None):
        self.layer
        return self._geometry_tensor(betas,pose_params or {})[0]

    def reconstruct(self, pose_params, betas):
        import torch
        layer=self.layer
        values=torch.as_tensor(betas,dtype=layer.shapedirs.dtype,device=self.device)
        with torch.no_grad(): joints,rotations=self._geometry_tensor(values,pose_params)
        return joints.cpu().numpy(),rotations.cpu().numpy()
