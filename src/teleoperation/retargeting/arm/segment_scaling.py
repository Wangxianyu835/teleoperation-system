"""Scale articulated segments independently without changing their directions."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np

PARENTS={'shoulder_center':'pelvis','left_shoulder':'shoulder_center','right_shoulder':'shoulder_center',
         'left_elbow':'left_shoulder','right_elbow':'right_shoulder','left_wrist':'left_elbow','right_wrist':'right_elbow','head':'shoulder_center'}


def _points(points,anchor='pelvis'):
    result={k:np.asarray(v,dtype=float).copy() for k,v in points.items()}
    required={'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'}
    if anchor=='pelvis': required.add('pelvis')
    if not required.issubset(result): raise ValueError(f'missing points: {sorted(required-set(result))}')
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in result.values()): raise ValueError('points must be finite three-vectors')
    result['shoulder_center']=(result['left_shoulder']+result['right_shoulder'])/2
    return result


@dataclass(frozen=True)
class SegmentScaling:
    scales: Mapping[str,float]
    anchor: str = 'pelvis'

    def __post_init__(self):
        if self.anchor not in ('pelvis','shoulder_center'): raise ValueError('unsupported scaling anchor')
        if any(k not in PARENTS or not np.isfinite(v) or v<=0 for k,v in self.scales.items()):
            raise ValueError('segment scales must be positive finite values for named chain segments')
        object.__setattr__(self,'scales',dict(self.scales))

    @classmethod
    def from_tpose(cls,human_points,robot_points,anchor='pelvis'):
        human,robot=_points(human_points,anchor),_points(robot_points,anchor)
        scales={}
        for child,parent in PARENTS.items():
            if child==anchor: continue
            if child not in human or child not in robot: continue
            source=np.linalg.norm(human[child]-human[parent]);target=np.linalg.norm(robot[child]-robot[parent])
            if source<1e-10 or target<1e-10: raise ValueError(f'degenerate {parent} to {child} reference segment')
            scales[child]=float(target/source)
        return cls(scales,anchor)

    def apply(self,points):
        source=_points(points,self.anchor)
        output={self.anchor:source[self.anchor].copy()}
        if 'pelvis' in source: output['pelvis']=source['pelvis'].copy()
        for child,parent in PARENTS.items():
            if child==self.anchor: continue
            if child in source:
                if child not in self.scales: raise ValueError(f'missing scale for {child}')
                output[child]=output[parent]+self.scales[child]*(source[child]-source[parent])
        return {k:output.get(k,v.copy()) for k,v in points.items()}

    def to_dict(self): return {'scales':dict(self.scales),'anchor':self.anchor}

    @classmethod
    def from_dict(cls,value): return cls(value['scales'],value.get('anchor','pelvis'))
