"""Train target-native joint networks with six weighted hand losses."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE, palm_length_metadata
from teleoperation.data.transaction import atomic_output
from teleoperation.data.hand_h5 import load_twohand_h5
from teleoperation.learning.dataset import TwoHandH5Dataset
from teleoperation.learning.losses import (
    COLLISION_THRESHOLD_M, LOSS_NAMES, LOSS_WEIGHTS, hand_loss, hand_loss_geometry,
)
from teleoperation.paths import training_run_directory
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.predictor import build_joint_model
from teleoperation.retargeting.hand.hand_model import DifferentiableHandModel
from teleoperation.retargeting.hand.checkpoint import (
    load_shared_joint_checkpoint,
    save_joint_checkpoint,
    shared_joint_checkpoint_payload,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def _epoch(model, fk, samples, sides, batch_size, device, optimizer=None, seed=0,
           loss_geometry=None):
    if isinstance(sides, str):
        sides = (sides,)
    entries = [
        (i, side)
        for side in sides
        for i, row in enumerate(samples)
        if row[side + '_valid']
    ]
    if not entries:
        raise ValueError(f'No valid windows for sides {tuple(sides)!r}')
    training = optimizer is not None
    model.train(training)
    if training:
        np.random.RandomState(seed).shuffle(entries)
    totals = dict.fromkeys(('loss', *LOSS_NAMES), 0.)
    if loss_geometry is None:
        loss_geometry = hand_loss_geometry(fk)
    with torch.set_grad_enabled(training):
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start+batch_size]
            rows = [samples[i] for i, _ in batch]
            inputs = torch.as_tensor(
                np.stack([row[side + '_input'] for row, (_, side) in zip(rows, batch)]),
                device=device,
            )
            human = np.stack([
                row[side + '_target'][0]
                for row, (_, side) in zip(rows, batch)
            ])
            q = model(inputs)
            loss, components = hand_loss(q, human, fk, loss_geometry)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint training loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
            for name, value in {'loss':loss, **components}.items():
                totals[name] += float(value.detach())*len(rows)
    return {**{name:value/len(entries) for name,value in totals.items()}, 'windows':len(entries)}


def train_joint_models(*, input_path, root, hand_models, run_name,
                       epochs=100, batch_size=64, learning_rate=1e-4, val_ratio=.2,
                       early_stopping_patience=20, seed=1234, device='cpu',
                       architecture=None, resume=None, init_checkpoint=None,
                       input_profile=PALM_LENGTH_PROFILE):
    """Train targets sequentially, preflighting every output before the first run.

    Train one canonical right-hand output model from both source sides.

    The selected target hand model remains independent, while left and right
    camera windows share one canonical output checkpoint.
    """
    if input_profile != PALM_LENGTH_PROFILE:
        raise ValueError('New training requires the palm-length input profile')
    if resume is not None and init_checkpoint is not None:
        raise ValueError('Resume and initialization checkpoint are mutually exclusive')
    if epochs < 1 or batch_size < 1 or early_stopping_patience < 1 or not np.isfinite(learning_rate) or learning_rate <= 0 or not 0 < val_ratio < 1:
        raise ValueError('Invalid training epochs/batch/learning rate/validation ratio/patience')
    names = list(hand_models)
    if not names or len(names) != len(set(names)):
        raise ValueError('Select unique target hand models')
    sides = ('left', 'right')
    specs = [load_hand_spec(name, 'right') for name in names]
    fks = {spec.hand_model:DifferentiableHandModel(spec) for spec in specs}
    geometries = {name:hand_loss_geometry(fk) for name,fk in fks.items()}
    if resume is not None and len(specs) != 1:
        raise ValueError('Resume requires one target model')
    runs = [training_run_directory(root,s.hand_model,'shared',PALM_LENGTH_PROFILE,run_name) for s in specs]
    for run in runs:
        if resume is None and run.exists():
            raise FileExistsError(f'Training directory already exists: {run}')
        if resume is not None and Path(resume).resolve() != (run/'model_last.pth').resolve():
            raise ValueError('Resume must explicitly select this shared run model_last.pth')
    source = resume if resume is not None else init_checkpoint
    loaded_sources = {}
    if source is not None:
        # A later target mismatch must fail before the first output directory.
        for spec in specs:
            loaded, payload = load_shared_joint_checkpoint(source,spec,'cpu')
            if architecture is not None and build_joint_model(spec,architecture).model_kwargs != loaded.model_kwargs:
                raise ValueError('Initialization/resume architecture mismatch')
            loaded_sources[spec.hand_model] = (loaded,payload)
    initialization = None if init_checkpoint is None else {'path':str(Path(init_checkpoint).resolve()),'sha256':_sha256(init_checkpoint)}
    if resume is not None:
        initialization = loaded_sources[specs[0].hand_model][1].get('training_config',{}).get('initialization')
    frames = len(load_twohand_h5(input_path)[0])
    train_end = int(frames*(1-val_ratio))
    val_start = train_end+2
    if train_end < 3 or frames-val_start < 3:
        raise ValueError('Training/validation splits must contain complete three-frame windows')
    train = TwoHandH5Dataset(input_path, frame_end=train_end, input_profile=PALM_LENGTH_PROFILE)
    validation = TwoHandH5Dataset(input_path, frame_start=val_start, input_profile=PALM_LENGTH_PROFILE)
    identity = {'source':str(Path(input_path).resolve()), 'sha256':_sha256(input_path),
                'frames':frames, 'train_raw_range':[0,train_end], 'validation_raw_range':[val_start,frames],
                'train_side_counts':train.side_counts(), 'validation_side_counts':validation.side_counts(),
                'input_profile':PALM_LENGTH_PROFILE}
    for side in sides:
        if not train.side_counts()[side] or not validation.side_counts()[side]:
            raise ValueError(f'No valid {side} windows in training/validation split')
    for spec, run in zip(specs,runs):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        selected_architecture = architecture
        if selected_architecture is None and source is not None:
            selected_architecture = loaded_sources[spec.hand_model][0].model_kwargs
        model = build_joint_model(spec, selected_architecture).to(device)
        if init_checkpoint is not None:
            model.load_state_dict(loaded_sources[spec.hand_model][0].state_dict(),strict=True)
        config = {'hand_model':spec.hand_model, 'target_side':'shared', 'source_sides':list(sides),
                  'spec_hash':spec.spec_hash, 'joint_names':list(spec.independent_joint_names),
                  'run_name':run_name, 'batch_size':batch_size, 'learning_rate':learning_rate,
                  'seed':seed, 'early_stopping_patience':early_stopping_patience,
                  'initialization':initialization,
                  'model_kwargs':model.model_kwargs, **palm_length_metadata(),
                  'loss':{'name':'hand_six_term_mse_v1', 'criterion':'MSELoss',
                          'weights':dict(zip(LOSS_NAMES, LOSS_WEIGHTS)),
                          'human_scale':'target_reference_palm_length_m',
                          'collision_threshold_m':COLLISION_THRESHOLD_M,
                          'tip_distance_scale':1000., 'thumb_distance_factor':.9,
                          'geometry':geometries[spec.hand_model]},
                  'optimizer':{'type':'AdamW','weight_decay':.01,'eps':1e-6},
                  'scheduler':{'factor':.5,'patience':8,'min_lr':1e-6}, 'gradient_clip_norm':10.}
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01, eps=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=8,min_lr=1e-6)
        first, best, stale = 0, float('inf'), 0
        if resume is not None:
            if json.loads((run/'config.json').read_text(encoding='utf-8')) != config or json.loads((run/'dataset.json').read_text(encoding='utf-8')) != identity:
                raise ValueError('Resume configuration or dataset identity mismatch')
            loaded, state = loaded_sources[spec.hand_model]
            if state.get('training_config') != config or state.get('dataset_identity') != identity:
                raise ValueError('Resume checkpoint configuration/data identity mismatch')
            model.load_state_dict(loaded.state_dict())
            optimizer.load_state_dict(state['optimizer_state'])
            scheduler.load_state_dict(state['scheduler_state'])
            first, best, stale = state['epoch'], state['best_loss'], state['stale_epochs']
            torch.set_rng_state(state['torch_rng'].cpu())
            random.setstate(state['python_rng'])
            if 'cuda_rng' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            if epochs <= first:
                raise ValueError('Resume epochs must exceed the completed epoch count')
            if stale >= early_stopping_patience:
                raise ValueError('Resume run already reached its configured early stopping condition')
        else:
            run.mkdir(parents=True, exist_ok=False)
            _json(run/'config.json',config)
            _json(run/'dataset.json',identity)
        fk = fks[spec.hand_model].to(device)
        with (run/'training.log').open('a',encoding='utf-8') as log, (run/'metrics.jsonl').open('a',encoding='utf-8') as metrics:
            log.write(f'{datetime.now().isoformat()} target={spec.key} device={device} start_epoch={first}\n')
            for epoch in range(first,epochs):
                if stale >= early_stopping_patience:
                    break
                training = _epoch(model,fk,train.samples,sides,batch_size,device,optimizer,seed+epoch,
                                  loss_geometry=geometries[spec.hand_model])
                valid = _epoch(model,fk,validation.samples,sides,batch_size,device,
                               loss_geometry=geometries[spec.hand_model])
                scheduler.step(valid['loss'])
                improved = valid['loss'] < best
                best, stale = (valid['loss'],0) if improved else (best,stale+1)
                row = {'epoch':epoch+1,'train':training,'validation':valid,'learning_rate':optimizer.param_groups[0]['lr']}
                line = json.dumps(row)
                metrics.write(line+'\n'); metrics.flush()
                log.write(line+'\n'); log.flush()
                payload = shared_joint_checkpoint_payload(model,spec)
                payload.update(epoch=epoch+1,best_loss=best,stale_epochs=stale,
                               optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
                               torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                               training_config=config,dataset_identity=identity)
                if torch.cuda.is_available():
                    payload['cuda_rng'] = torch.cuda.get_rng_state_all()
                if improved:
                    save_joint_checkpoint(run/'model_best.pth',payload)
                save_joint_checkpoint(run/'model_last.pth',payload)
                print(f'{spec.hand_model}/shared epoch={epoch+1} validation={valid["loss"]:.6f} saved={run}')
    return runs
