"""Real SMPL geometry -> shared CLI session -> PyBullet, with synthetic raw poses.

This verifies software integration and single/multi command equivalence. It does
not run a camera or neural estimator and does not certify physical tracking.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
ROBOTS=('h1_2','gr1_t2','g1','tron2a_l21')


def generate_body(path,profile_dir,frames):
    from teleoperation.data.arm_fitting import load_arm_fitting
    from teleoperation.retargeting.arm.body_model import BodyModel
    from teleoperation.contracts.body import BodyFrameId,SmplerXResult
    from teleoperation.contracts.observations import RawHandFrame,RawTeleopObservation
    from teleoperation.inputs.smplerx_geometry import smplerx_observations
    from teleoperation.data.body_recording import capture_sample,save_body_recording
    profile=load_arm_fitting(profile_dir/'g1.json')
    model=BodyModel(profile.body_spec)
    rows=[]
    for frame in range(frames):
        stamp=100.+frame/30
        body=np.zeros((1,63),dtype=np.float32)
        body[0,15*3+2]=.1*np.sin(frame/30*2*np.pi)
        body[0,16*3+2]=-.1*np.sin(frame/30*2*np.pi)
        # Use the estimator's actual camera convention: an upright operator
        # facing the camera has Rx(pi), with a small changing native yaw.
        yaw=.15*np.sin(frame/30*2*np.pi)
        root=Rotation.from_matrix(np.diag([1.,-1.,-1.])@Rotation.from_euler('y',yaw).as_matrix()).as_rotvec()
        params={'smplx_root_pose':root[None].astype(np.float32),'smplx_body_pose':body,
                'smplx_lhand_pose':np.zeros((1,45),dtype=np.float32),
                'smplx_rhand_pose':np.zeros((1,45),dtype=np.float32)}
        joints,rotations=model.reconstruct(params,np.zeros((1,profile.body_spec.num_betas)))
        result=SmplerXResult(BodyFrameId('synthetic-raw-real-smpl',frame,stamp),stamp+.001,
            'synthetic_pose_real_smplx',profile.body_spec_hash,
            {**params,'joints_root_m':joints[0],'global_rotations':rotations[0],'image_joints':joints[0]},
            {'left':True,'right':True},metadata={'joint_indices':{
                'left':{'shoulder':16,'elbow':18,'wrist':20},'right':{'shoulder':17,'elbow':19,'wrist':21}},
                'capture_kind':'synthetic_raw_pose_real_smpl_geometry'})
        upper,wrists=smplerx_observations(result)
        raw=RawTeleopObservation(RawHandFrame({'left':None,'right':None},stamp,'smplerx',
                    metadata={'timestamp_unit':'monotonic_s'}),upper,wrists,result)
        source=SimpleNamespace(session_id=result.frame_id.session_id,last_result=result,
            last_frame=SimpleNamespace(frame_id=frame,timestamp=stamp,rgb=np.zeros((8,8,3),dtype=np.uint8)))
        rows.append(capture_sample(source,raw))
    save_body_recording(path,rows,{'source':'smplerx','model_id':'synthetic_pose_real_smplx',
        'model_hash':profile.body_spec_hash,'time_unit':'monotonic_s',
        'capture_kind':'synthetic_raw_pose_real_smpl_geometry','real_camera':False,
        'source_hz':30.,'runtime_root_pose':'camera Rx(pi) reference with changing native yaw; observed root retained'})


def trace(path,robot):
    from teleoperation.data.robot_command_recording import load_robot_recording
    record=load_robot_recording(path,(robot,))
    times=[];positions=[];newframes=set();statuses={};maxstep=0.
    previous=None
    for batch,receipt in record.iter_frames():
        command=batch[robot];times.append(command.timestamp)
        values=np.concatenate([command.parts[key].positions for key in sorted(command.parts)])
        if previous is not None:maxstep=max(maxstep,float(np.abs(values-previous).max()))
        previous=values;positions.append(values)
        for part in command.parts.values():
            statuses[part.status]=statuses.get(part.status,0)+1
            if part.source_is_new:newframes.add(part.target_sequence)
    return np.asarray(times),np.asarray(positions),{'frames':len(times),'new_target_sequences':len(newframes),
        'statuses':statuses,'max_command_step_rad':maxstep}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/acceptance/unified-smpl-pink/pipeline')
    parser.add_argument('--fitting-dir',type=Path,default=ROOT/'outputs/acceptance/unified-smpl-pink/fittings-v2')
    parser.add_argument('--source-frames',type=int,default=30)
    args=parser.parse_args(argv)
    directory=args.output_dir.resolve()
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(directory)
    directory.mkdir(parents=True,exist_ok=True)
    body_path=directory/'synthetic-raw-real-smpl.h5'
    generate_body(body_path,args.fitting_dir,args.source_frames)
    report={'capture_kind':'synthetic_raw_pose_real_smpl_geometry','real_camera':False,
        'hardware_verified':False,'physical_tracking_verified':False,
        'resource_loaders_patched':False,'body_recording':str(body_path),'source_frames':args.source_frames,
        'source_hz':30.,'control_hz':100.,'runs':{},'comparisons':{},
        'geometry':{'h1_2':'original full mesh','gr1_t2':'original full mesh','g1':'original full mesh',
                    'tron2a_l21':'explicit simplified proxy; full-mesh acceptance excluded'}}
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    for name,robots in [('all',ROBOTS),*((robot,(robot,)) for robot in ROBOTS)]:
        recording=directory/f'{name}.h5';summary=directory/f'{name}-report.json'
        selector=['--robot',robots[0]] if len(robots)==1 else ['--robots',*robots]
        command=[sys.executable,'-m','teleoperation','robot','control',*selector,
                 '--parts','arms','--side','both','--source','camera','--backend','dex','--headless',
                 '--geometry','simplified','--input-body',str(body_path),
                 '--arm-fitting-dir',str(args.fitting_dir),'--control-hz','100','--target-max-age','.3',
                 '--record-results',str(recording),'--report',str(summary)]
        started=time.perf_counter()
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (directory/f'{name}.log').write_text(completed.stdout,encoding='utf-8')
        if completed.returncode:raise RuntimeError(f'{name} CLI failed ({completed.returncode}); see {directory/name}.log')
        report['runs'][name]={**json.loads(summary.read_text()),'elapsed_wall_s':time.perf_counter()-started,
                             'recording':str(recording),'command':command}
        print(name,report['runs'][name]['frames'],report['runs'][name]['body_reconstructions'],flush=True)
    for robot in ROBOTS:
        single_time,single,info=trace(directory/f'{robot}.h5',robot)
        multi_time,multi,_=trace(directory/'all.h5',robot)
        same_time=np.array_equal(single_time,multi_time)
        difference=float(np.abs(single-multi).max()) if same_time and single.shape==multi.shape else None
        report['comparisons'][robot]={**info,'control_timestamps_equal':same_time,
            'max_single_vs_multi_joint_difference_rad':difference,
            'command_equivalence_pass':bool(difference is not None and difference<=1e-5)}
    report['integration_pass']=all(r['command_equivalence_pass'] for r in report['comparisons'].values()) and all(
        r['body_reconstructions']==args.source_frames for r in report['runs'].values())
    (directory/'pipeline-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'integration_pass':report['integration_pass'],'comparisons':report['comparisons']},indent=2))
    return 0 if report['integration_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
