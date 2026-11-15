"""Render controlled arm diagnosis artifacts without rerunning either controller."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--replay', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    with (args.matrix/'matrix.csv').open(encoding='utf-8') as handle: rows = list(csv.DictReader(handle))
    # Exclude reference and opposite-arm perturbations: 28 nontrivial target-side cases.
    active = [r for r in rows if r['case'].startswith(r['side']+'_') and int(r['step']) == 59]
    groups = sorted({r['group'] for r in active})
    matrix_summary = {}
    for group in groups:
        subset = [r for r in active if r['group'] == group]
        matrix_summary[group] = {'cases': len(subset),
            'position_median_mm': float(np.median([float(r['position_mm']) for r in subset])),
            'position_p95_mm': float(np.percentile([float(r['position_mm']) for r in subset],95)),
            'orientation_median_deg': float(np.median([float(r['orientation_deg']) for r in subset])),
            'integrated_seconds': 60*float(subset[0]['dt'])}
    all_summary = {'active_matrix': matrix_summary, 'replays': {}}
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels = [g.replace('_bullet',' / Bullet').replace('_pink',' / Pink').replace('_',' ') for g in groups]
    axes[0].bar(range(len(groups)),[matrix_summary[g]['position_p95_mm'] for g in groups],color='#277b9a')
    axes[1].bar(range(len(groups)),[matrix_summary[g]['orientation_median_deg'] for g in groups],color='#cf713f')
    for ax in axes:
        ax.set_xticks(range(len(groups)),labels,rotation=40,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set(title='Position P95 on 28 actual arm perturbations',ylabel='mm')
    axes[1].set(title='Orientation median (position-only groups do not constrain it)',ylabel='degrees')
    fig.suptitle('Same model + same end-effector + reachable FK targets\n60 updates: 30 Hz groups integrate 2 s; 5 ms clock contrasts integrate 0.3 s')
    fig.savefig(args.output/'controlled-matrix.png',dpi=150);plt.close(fig)
    witness = load_json(args.probe/'witness.json')
    history = witness['diagnostic_position_progress']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels=['Current hold','Pink pose','Diagnostic position progress']
    position=[witness['after_60']['bullet']['error'][0],witness['after_60']['pink']['error'][0],history[-1]['position_mm']]
    orientation=[witness['after_60']['bullet']['error'][1],witness['after_60']['pink']['error'][1],history[-1]['orientation_deg']]
    for ax,values,title,unit in zip(axes,(position,orientation),('Position error','Orientation error'),('mm','degrees')):
        ax.bar(labels,values,color=['#9b6363','#277b9a','#cf713f']);ax.set(title=title,ylabel=unit)
        ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Known reachable pose: moving toward position helps position, but does NOT solve orientation\nDiagnostic counterfactual only; no production behavior changed')
    fig.savefig(args.output/'reachable-witness.png',dpi=150);plt.close(fig)
    for folder in args.replay:
        datasets={mode:np.load(folder/(mode+'.npz')) for mode in ('baseline','position')}
        summaries={mode:load_json(folder/(mode+'-summary.json')) for mode in datasets}
        a,b=datasets.values()
        n=min(summaries['baseline']['frames_processed'],summaries['position']['frames_processed'])
        if n == 0: raise ValueError('No common executed frames')
        finite=np.isfinite(a['target'][:n])&np.isfinite(b['target'][:n])
        target_difference=float(np.max(np.abs(a['target'][:n][finite]-b['target'][:n][finite])))
        identical_targets = target_difference <= 1e-7
        offsets = {}
        for j,side in enumerate(('left','right')):
            delta = a['target'][:n,j]-b['target'][:n,j]
            delta = delta[np.isfinite(delta).all(axis=1)]
            if not np.allclose(delta, delta[:1], atol=1e-9, rtol=0):
                raise ValueError('Ablation position targets differ by more than a constant calibration offset')
            offsets[side] = delta[0].tolist()
        t=a['timestamps'][:n]-a['timestamps'][0]
        info={'common_frames':n,'common_duration_s':float(t[-1]),
              'maximum_target_component_difference_m':target_difference,
              'identical_targets': identical_targets, 'calibration_target_offset_m': offsets,
              'modes':summaries}
        all_summary['replays'][folder.name]=info
        recording = Path(load_json(folder/'manifest.json')['input'])
        with h5py.File(recording, 'r') as handle:
            head = handle['head_transform'][:, :3, 3]
            wrist = handle['left_wrist_transform'][:, :3, 3]
            input_t = handle['timestamps'][:]-handle['timestamps'][0]
        basis = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        from_start = (wrist-wrist[:1]) @ basis.T
        from_head = ((wrist-head)-(wrist[:1]-head[:1])) @ basis.T
        fig,ref_axes = plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
        for k,ax in enumerate(ref_axes):
            ax.plot(input_t,from_start[:,k]*1000,label='initial wrist reference',lw=.8)
            ax.plot(input_t,from_head[:,k]*1000,label='head-relative change',lw=.8)
            ax.set(ylabel='XYZ'[k]+' motion (mm)');ax.grid(alpha=.2)
        ref_axes[0].legend();ref_axes[-1].set_xlabel('Recording time (s)')
        fig.suptitle('Isolating reference choice: same left-wrist input, unit scale, no robot offsets\nThese are relative input motions, not full official/current robot targets.')
        fig.savefig(args.output/(folder.name+'-reference-motion.png'),dpi=145);plt.close(fig)
        info['reference_choice_max_motion_difference_mm'] = float(1000*np.max(np.linalg.norm(from_start-from_head,axis=1)))
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for j,side in enumerate(('left','right')):
            for mode,z in datasets.items():
                color='#277b9a' if mode=='baseline' else '#cf713f'
                error=np.linalg.norm(z['command_fk'][:n,j]-z['target'][:n,j],axis=1)*1000
                axes[0,j].plot(t,error,label=mode,color=color,lw=.9)
                axes[1,j].plot(t,z['orientation_error_deg'][:n,j],label=mode,color=color,lw=.9)
            axes[0,j].set(title=side.title()+' position',ylabel='mm')
            axes[1,j].set(title=side.title()+' orientation',ylabel='degrees',xlabel='Recording time (s)')
        for ax in axes.flat:ax.legend();ax.grid(alpha=.2)
        target_label = ('IDENTICAL position targets' if identical_targets else
                        f'constant calibration offset: max {target_difference*1000:.4f} mm; NOT subtracted')
        fig.suptitle(f'{folder.name}: same native model; {target_label}\nCommon executed prefix: {n} frames / {t[-1]:.2f} s. Full-run failures are recorded separately.')
        fig.savefig(args.output/(folder.name+'.png'),dpi=145);plt.close(fig)
        if folder==args.replay[0]:
            fig,axes=plt.subplots(1,2,figsize=(10,5),subplot_kw={'projection':'3d'})
            lines=[]
            for ax,name in zip(axes,('Current baseline','Position-only diagnostic')):
                ax.set(title=name,xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',xlim=(-.15,.75),ylim=(-.65,.65),zlim=(-.25,.85))
                ax.set_xticks([0,.2,.4,.6]);ax.set_yticks([-.6,-.3,0,.3,.6]);ax.set_zticks([0,.2,.4,.6,.8])
                ax.view_init(elev=15,azim=-45);ax.set_box_aspect((.9,1.3,1.1))
                panel=[]
                for color in ('#277b9a','#cf713f'):
                    line,=ax.plot([],[],[],color=color,marker='o',lw=3)
                    target,=ax.plot([],[],[],color=color,marker='x',linestyle='None',ms=8)
                    panel.append((line,target))
                lines.append(panel)
            title=fig.suptitle('')
            indices=np.linspace(0,n-1,110,dtype=int)
            def update(frame):
                i=indices[frame]
                title.set_text(f'Actual PyBullet arm pose | recorded t={t[i]:.2f} s | x=position target\nPosition-only can lose wrist orientation. No objects or contact task.')
                for k,z in enumerate(datasets.values()):
                    for j in range(2):
                        points=np.stack([z['upper'][i,j],z['elbow'][i,j],z['actual'][i,j]])
                        line,target=lines[k][j]
                        line.set_data_3d(points[:,0],points[:,1],points[:,2])
                        point=z['target'][i,j];target.set_data_3d([point[0]],[point[1]],[point[2]])
                return []
            animation=FuncAnimation(fig,update,frames=len(indices),interval=100,blit=False)
            update(50);fig.savefig(args.output/'native-preview.png',dpi=130)
            animation.save(args.output/'native-preview.gif',writer=PillowWriter(fps=10),dpi=85)
            plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(all_summary,indent=2),encoding='utf-8')
    sources=[Path(__file__),ROOT/'devtools/gr1_arm_diagnosis.py']
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (args.output/'plot-sources.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(all_summary,indent=2))


if __name__=='__main__':main()
