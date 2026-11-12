"""Real-model acceptance for the shared PINK solver and output-only PyBullet.

Run with the pinned native Python and PINK dependency path, for example:
  python devtools/verify_unified_pink.py --robots-root assets/robots
The JSON distinguishes mathematical IK residuals from physically read link poses.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from teleoperation.robots.arm_spec import load_arm_spec
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.retargeting.arm.pink_ik import PinkArmIK
from teleoperation.simulation.robot_backends import SimulationGroup
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.simulation.robot_output import RobotOutput
from teleoperation.applications.robot_commands import RobotCommandAssembler
from teleoperation.contracts.robot_control import RobotPartCommand

PATHS={'h1_2':'h1_2/h1_2.urdf','gr1_t2':'gr1/urdf/robot.urdf',
       'g1':'g1/g1_29dof_with_hand_lock_waist.urdf','tron2a_l21':'tron2a/robot.urdf'}
DT=.02

def pose_errors(actual,targets):
    import pinocchio as pin
    errors={s:{'position_mm':float(np.linalg.norm(actual[s][:3,3]-t[:3,3])*1000),
               'orientation_deg':float(np.rad2deg(np.linalg.norm(pin.log3(actual[s][:3,:3].T@t[:3,:3]))))}
            for s,t in targets.items()}
    return {'sides':errors,'max_position_mm':max(v['position_mm'] for v in errors.values()),
            'max_orientation_deg':max(v['orientation_deg'] for v in errors.values())}

def target_for(ik):
    goal=ik.q.copy()
    for names in ik.spec.arm_joint_names.values():
        for name in names[:4]:
            j=ik.model.joints[ik.model.getJointId(name)]
            goal[j.idx_q]+=.09
    return ik.measure(goal)

def trajectory(spec,initial,steps):
    ik=PinkArmIK(spec,initial)
    targets=target_for(ik)
    trace=[]; max_ratio=0.; invariant=True
    for _ in range(steps):
        before=ik.q; proposal=ik.prepare(targets,DT)
        invariant &= np.array_equal(before,ik.q)
        max_ratio=max(max_ratio,float(np.max(np.abs(proposal.q-before)/DT/ik.model.velocityLimit)))
        ik.commit(proposal); trace.append(ik.q)
    right=ik.joint_values()['right']
    candidate=ik.prepare({'left':targets['left']},DT)
    held=right==dict(candidate.joint_values['right'])
    missing=np.array_equal(ik.q,ik.prepare({},DT).q)
    return targets,trace,{'solver_errors':pose_errors(ik.measure(),targets),
                          'max_velocity_bound_ratio':max_ratio,
                          'prepare_preserves_committed_state':bool(invariant),
                          'single_side_other_arm_exactly_held':bool(held),
                          'missing_targets_exactly_held':bool(missing)}

def bullet_poses(backend,spec):
    p=backend.scene.p; cid=backend.scene.cid; body=backend.robot['body']
    links={p.getJointInfo(body,i,physicsClientId=cid)[12].decode():i
           for i in range(p.getNumJoints(body,physicsClientId=cid))}
    def matrix(position,quaternion):
        t=np.eye(4); t[:3,:3]=np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)
        t[:3,3]=position; return t
    com=matrix(*p.getBasePositionAndOrientation(body,physicsClientId=cid))
    dynamics=p.getDynamicsInfo(body,-1,physicsClientId=cid)
    world_root=com@np.linalg.inv(matrix(dynamics[3],dynamics[4]))
    result={}
    for side,frame in spec.wrist_frames.items():
        state=p.getLinkState(body,links[frame],computeForwardKinematics=True,physicsClientId=cid)
        result[side]=np.linalg.inv(world_root)@matrix(state[4],state[5])@spec.wrist_offsets[side]
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robots-root',type=Path,default=ROOT/'assets/robots')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/acceptance/unified-smpl-pink/pink-acceptance.json')
    parser.add_argument('--steps',type=int,default=101)
    parser.add_argument('--tron-geometry',choices=('full','simplified'),default='full')
    args=parser.parse_args(argv)
    if args.steps<1: parser.error('--steps must be positive')
    started=time.monotonic()
    group=SimulationGroup(gui=False)
    report={'schema_version':1,'hardware_verified':False,'python':platform.python_version(),
            'control_dt_s':DT,'steps':args.steps,'physics_dt_s':group.time_step,
            'physics_model':'original full URDF with original meshes, gravity, fixed root; position motor outputs',
            'robots':{},'scene_load_errors':{},'tron_geometry':args.tron_geometry}
    output=None
    try:
        backends={}; specs={}
        for index,(name,path) in enumerate(PATHS.items()):
            urdf=args.robots_root/path
            specs[name]=load_arm_spec(name,urdf)
            control=load_robot_control_spec(name,resource_overrides={'parts':'both','urdf_path':urdf})
            try:
                backends[name]=RobotPartSceneBuilder().build(control,group=group,solve_arms=False,
                    base_pose=(index*2.2,0.,1.6),geometry=args.tron_geometry if name=='tron2a_l21' else 'full')
            except FileNotFoundError as error:
                report['scene_load_errors'][name]=str(error)
                continue
            assert backends[name].driver is None
        output=RobotOutput(backends)
        state=output.read_state(); initial={}
        for name,backend in backends.items():
            frame=backend.spec.reference_frame(0,0.)
            initial[name]=replace(frame,parts={k:replace(v,positions=state[name][k]) for k,v in frame.parts.items()})
        assembler=RobotCommandAssembler({n:b.spec for n,b in backends.items()},initial)
        solvers={};targets={};traces={}
        for name,spec in specs.items():
            if name in backends:
                backend=backends[name]
                named={n:float(q) for key,part in backend.spec.parts.items() if key.endswith('_arm')
                       for n,q in zip(part.joint_names,state[name][key])}
            else:
                named={n:spec.reference[n] for names in spec.arm_joint_names.values() for n in names}
            targets[name],traces[name],record=trajectory(specs[name],named,args.steps)
            solvers[name]=PinkArmIK(specs[name],named)
            report['robots'][name]={**record,'anatomical_anchor':specs[name].anatomical_anchor,
                'arm_spec_hash':specs[name].spec_hash,'urdf':str(specs[name].urdf_path),
                'urdf_sha256':hashlib.sha256(specs[name].urdf_path.read_bytes()).hexdigest(),
                'single_vs_multi_max_joint_difference_rad':0.,
                'physical_available':name in backends, 'physical_pass':False,
                'actual_errors_after_2s_settle':None,
                'full_mesh_geometry':name!='tron2a_l21' or args.tron_geometry=='full'}
        for step in range(args.steps):
            pending={name:ik.prepare(targets[name],DT) for name,ik in solvers.items()}
            batch={}
            for name,proposal in pending.items():
                if name not in backends:continue
                updates={}
                for side,values in proposal.joint_values.items():
                    key=side+'_arm'; part=backends[name].spec.parts[key]
                    updates[key]=RobotPartCommand(part.joint_names,[values[n] for n in part.joint_names],
                        0.,'synthetic_static_fk_s',True,'tracking',0,1e6,step==0)
                batch[name]=assembler.prepare(name,step,(step+1)*DT,updates)
            receipt=output.submit(batch,steps=5)
            if not all(s=='submitted' for s in receipt.statuses.values()):raise RuntimeError('Output rejected proposal')
            for name,proposal in pending.items():
                if name in backends:assembler.commit(batch[name])
                solvers[name].commit(proposal)
                difference=float(np.max(np.abs(proposal.q-traces[name][step])))
                record=report['robots'][name]
                record['single_vs_multi_max_joint_difference_rad']=max(record['single_vs_multi_max_joint_difference_rad'],difference)
        # Hold the final accepted command for two physical seconds, through the same output boundary.
        final={name:assembler.prepare(name,args.steps,(args.steps+1)*DT,{}) for name in backends}
        output.submit(final,steps=480)
        for name,ik in solvers.items():
            record=report['robots'][name]
            if name in backends:
                measured=bullet_poses(backends[name],specs[name])
                record['actual_errors_after_2s_settle']=pose_errors(measured,targets[name])
                record['actual_vs_solver']=pose_errors(measured,ik.measure())
            record['accepted_frames']=args.steps+1
            record['solver_pass']=bool(record['solver_errors']['max_position_mm']<5 and
                record['solver_errors']['max_orientation_deg']<3 and record['max_velocity_bound_ratio']<=1+1e-7
                and record['prepare_preserves_committed_state'] and record['single_side_other_arm_exactly_held']
                and record['missing_targets_exactly_held'] and record['single_vs_multi_max_joint_difference_rad']<1e-12)
            record['physical_pass']=bool(record['physical_available'] and record['full_mesh_geometry'] and
                record['actual_errors_after_2s_settle']['max_position_mm']<5 and
                record['actual_errors_after_2s_settle']['max_orientation_deg']<3)
        report['pass']=all(r['solver_pass'] and r['physical_pass'] for r in report['robots'].values())
    finally:
        if output is not None:output.close()
        else:group.close()
    report['elapsed_s']=time.monotonic()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'pass':report['pass'],
                      'robots':{n:{k:r[k] for k in ('solver_pass','physical_pass','actual_errors_after_2s_settle')}
                                for n,r in report['robots'].items()}},indent=2))
    return 0 if report['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
