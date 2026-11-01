"""Run the unified world-H5 -> target NN -> native command replay workflow.

Input must be a real or explicitly synthetic world-metre recording. This tool
never relabels image-normalized coordinates as metres or fabricates training data.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from teleoperation.contracts.hand_models import HAND_MODELS
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
from teleoperation.paths import PROJECT_ROOT, training_run_directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True,help='raw world-metre human H5')
    parser.add_argument('--work-dir',type=Path,default=PROJECT_ROOT/'outputs/tmp_hand_pipeline')
    parser.add_argument('--hand-model',choices=HAND_MODELS,default='l21')
    parser.add_argument('--hand',choices=('left','right'),default='right')
    parser.add_argument('--run-name',default='pipeline')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--no-replay',action='store_true')
    args=parser.parse_args()
    if not args.raw.is_file(): parser.error(f'Missing recording: {args.raw}')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    aligned=args.work_dir/'aligned.h5'
    result=args.work_dir/'joint_results.h5'
    models=args.work_dir/'models'
    checkpoint=training_run_directory(models,args.hand_model,args.hand,PALM_LENGTH_PROFILE,args.run_name)/'model_best.pth'
    commands=[
        ['hand','align','--input',args.raw,'--output',aligned,'--landmark-space','world'],
        ['hand','train','--input',aligned,'--hand-model',args.hand_model,'--hand-side',args.hand,
         '--run-name',args.run_name,'--checkpoint-root',models,'--epochs',args.epochs,
         '--batch-size',args.batch_size,'--device',args.device],
        ['hand','export','--input',aligned,'--output',result,'--hand-model',args.hand_model,
         '--hand',args.hand,'--backend','nn','--checkpoint',checkpoint,'--device',args.device],
        ['hand','inspect','--angle-h5',result],
    ]
    if not args.no_replay:
        commands.append(['replay','hand-joints','--input',result,'--hand-model',args.hand_model,'--hand',args.hand])
    for command in commands:
        invocation=[sys.executable,'-B','-m','teleoperation',*map(str,command)]
        print('$ '+' '.join(invocation),flush=True)
        code=subprocess.run(invocation,cwd=PROJECT_ROOT).returncode
        if code: return code
    print(f'Saved native results: {result}; short training is a smoke run, not accuracy acceptance.')
    return 0


if __name__=='__main__': raise SystemExit(main())
