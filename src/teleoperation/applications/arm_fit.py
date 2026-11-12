"""Fit the same SMPL morphology pipeline to declarative robot geometry."""
from pathlib import Path
import json
import numpy as np
from teleoperation.retargeting.arm.body_model import BodyModel,BodyModelSpec
from teleoperation.retargeting.arm.shape_fitting import fit_shape
from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
from teleoperation.robots.arm_spec import load_arm_spec
from teleoperation.data.arm_fitting import save_arm_fitting

JOINT_INDICES={'pelvis':0,'left_shoulder':16,'right_shoulder':17,'left_elbow':18,
               'right_elbow':19,'left_wrist':20,'right_wrist':21,'head':15}


def _torso_basis(points,up_axis=None):
    left=np.asarray(points['left_shoulder'])-points['right_shoulder']
    up=(np.asarray(points['left_shoulder'])+points['right_shoulder'])/2-points['pelvis'] if up_axis is None else np.asarray(up_axis,dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(up).all() or np.linalg.norm(left)<1e-9:
        raise ValueError('shoulder axis must be finite and nondegenerate')
    left=left/np.linalg.norm(left)
    up=up-left*np.dot(left,up)
    if np.linalg.norm(up)<1e-9: raise ValueError('pelvis to shoulder center must define a torso up axis')
    up=up/np.linalg.norm(up)
    forward=np.cross(left,up)
    return np.column_stack((forward,left,up))


def torso_alignment(human_points,robot_points,robot_up_axis=None):
    """Map orthonormal anatomical torso axes by a single proper rotation."""
    return _torso_basis(robot_points,robot_up_axis)@_torso_basis(human_points).T


def _named(joints,include_head=False):
    return {name:joints[index].copy() for name,index in JOINT_INDICES.items() if name!='head' or include_head}


def run(args):
    robots=list(args.robots)
    if not robots or len(set(robots))!=len(robots): raise ValueError('robots must be a nonempty distinct list')
    output_dir=Path(args.output_dir)
    paths={robot:output_dir/f'{robot}.json' for robot in robots}
    report_path=getattr(args,'report',None)
    existing=[path for path in paths.values() if path.exists()]
    if report_path is not None and Path(report_path).exists(): existing.append(Path(report_path))
    if existing: raise FileExistsError(f'Refusing to overwrite existing fitting artifacts: {existing}')
    spec=BodyModelSpec(str(args.body_model_path),getattr(args,'body_model_type','smplx'),
                       getattr(args,'gender','neutral'),getattr(args,'num_betas',20),
                       str(args.smplx_module_dir) if getattr(args,'smplx_module_dir',None) is not None else None)
    model=BodyModel(spec,device=getattr(args,'device','cpu'))
    specs={robot:load_arm_spec(robot,sides=('left','right')) for robot in robots}
    reference_joints,_=model.reconstruct({},np.zeros((1,spec.num_betas)))
    summary={'schema_version':1,'body_spec':spec.to_dict(),'body_spec_hash':spec.hash,'robots':{}}
    # Fit all profiles before publishing any: bad geometry cannot leave a partial set.
    prepared=[]
    for robot,robot_spec in specs.items():
        robot_points=robot_spec.landmark_positions(tpose=True)
        include_head='head' in robot_points
        reference=_named(reference_joints[0],include_head)
        rotation=torso_alignment(reference,robot_points,robot_up_axis=robot_spec.reference_up_axis)
        anchor=robot_spec.anatomical_anchor
        if anchor not in ('pelvis','shoulder_center'): raise ValueError(f'unsupported anatomical anchor: {anchor}')
        robot_anchor=robot_points['pelvis'] if anchor=='pelvis' else (robot_points['left_shoulder']+robot_points['right_shoulder'])/2
        names=['left_shoulder','right_shoulder','left_wrist','right_wrist']
        if anchor=='pelvis': names.insert(0,'pelvis')
        if include_head: names.append('head')
        target=np.stack([rotation.T@(robot_points[name]-robot_anchor) for name in names])
        fitted=fit_shape(model,target,tuple(JOINT_INDICES[name] for name in names),
                         steps=getattr(args,'fit_steps',1000),learning_rate=getattr(args,'fit_learning_rate',.05),pose_params={},
                         anchor_indices=(16,17) if anchor=='shoulder_center' else None)
        joints,global_rotations=model.reconstruct({},fitted.betas[None])
        human_anchor=(joints[0,16]+joints[0,17])/2 if anchor=='shoulder_center' else joints[0,0]
        human_points={name:rotation@(point-human_anchor) for name,point in _named(joints[0],include_head).items()}
        centered_robot={name:point-robot_anchor for name,point in robot_points.items()}
        if anchor=='shoulder_center':
            human_points.pop('pelvis');centered_robot.pop('pelvis')
        scaling=SegmentScaling.from_tpose(human_points,centered_robot,anchor=anchor)
        wrist_rotations=robot_spec.wrist_tpose_rotations()
        offsets={side:(rotation@global_rotations[0,JOINT_INDICES[f'{side}_wrist']]).T@wrist_rotations[side] for side in ('left','right')}
        scaled=scaling.apply(human_points)
        scaled_error={name:float(np.linalg.norm(scaled[name]-centered_robot[name])) for name in centered_robot if name in scaled}
        summary['robots'][robot]={'path':str(paths[robot].resolve()),'robot_spec_hash':robot_spec.spec_hash,
            **fitted.to_dict(),'fit_point_names':names,'segment_scaling':scaling.to_dict(),
            'beta_l2_norm':float(np.linalg.norm(fitted.betas)),
            'beta_abs_max':float(np.abs(fitted.betas).max()),
            'beta_count_abs_above_3':int(np.count_nonzero(np.abs(fitted.betas)>3)),
            'scaled_reference_point_error_m':scaled_error,'body_to_robot':rotation.tolist(),
            'shape_rms_within_20mm':bool(fitted.final_rms<=.02),'anchor':anchor,'robot_anchor':robot_anchor.tolist()}
        prepared.append((robot,robot_spec,fitted,scaling,rotation,offsets,anchor,robot_anchor))
    for robot,robot_spec,fitted,scaling,rotation,offsets,anchor,robot_anchor in prepared:
        save_arm_fitting(paths[robot],body_spec=spec,robot_spec_hash=robot_spec.spec_hash,fit_result=fitted,
                         segment_scaling=scaling,body_to_robot=rotation,pose_params={},wrist_rotation_offsets=offsets,
                         anchor=anchor,robot_anchor=robot_anchor)
    if report_path is not None:
        report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
        with report_path.open('x',encoding='utf-8') as stream: json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def run_cli(args):
    print(json.dumps(run(args),indent=2,allow_nan=False))
    return 0
