"""Select a target hand independently of the retargeting algorithm."""
from contextlib import ExitStack
import json
import time

import numpy as np

from teleoperation.cli.hand_joint_options import LEGACY_WORLD_PROFILE, LEGACY_NORMALIZED_PROFILE
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE, palm_length_metadata, palm_local_metadata
from teleoperation.contracts.hand import HandWindow
from teleoperation.data.hand_joint_recording import HandJointResultWriter, iter_joint_commands
from teleoperation.robots.hand_spec import HAND_MODELS, load_hand_spec
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from teleoperation.simulation.hand_executor import HandExecutor


def _sides(hand):
    return ('left','right') if hand == 'both' else (hand,)


def _selection(args):
    """Validate weights and preprocessing before any device or GUI is opened."""
    from teleoperation.retargeting.hand.device import resolve_device
    sides = _sides(args.hand)
    specs = {(args.hand_model,s):load_hand_spec(args.hand_model,s) for s in sides}
    paths = (args.checkpoint,args.left_checkpoint,args.right_checkpoint)
    if args.backend != 'nn' and any(p is not None for p in paths):
        raise ValueError('Checkpoint arguments only apply to --backend nn')
    if args.backend == 'dex':
        if args.input_profile != PALM_LENGTH_PROFILE:
            raise ValueError('Camera dex requires the palm-length input profile; use hand vr for calibrated VR')
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        retargeter = NormalizedDexRetargeter([args.hand_model],sides)
    elif args.backend == 'geometric':
        if args.hand_model != 'l21' or args.input_profile != LEGACY_NORMALIZED_PROFILE:
            raise ValueError('Geometric supports L21 with --input-profile legacy_l21_normalized')
        from teleoperation.retargeting.hand.geometric import GeometricHandRetargeter
        retargeter = GeometricHandRetargeter()
    elif args.input_profile == PALM_LENGTH_PROFILE:
        from teleoperation.retargeting.hand.checkpoint import load_joint_checkpoint
        from teleoperation.retargeting.hand.interface import JointNNRetargeter
        if args.hand == 'both':
            if args.checkpoint is not None or args.left_checkpoint is None or args.right_checkpoint is None:
                raise ValueError('New dual-side NN requires --left-checkpoint and --right-checkpoint')
            selected = {'left':args.left_checkpoint,'right':args.right_checkpoint}
        else:
            if args.checkpoint is None or args.left_checkpoint is not None or args.right_checkpoint is not None:
                raise ValueError('Single-side NN requires exactly --checkpoint')
            selected = {args.hand:args.checkpoint}
        device = resolve_device(args.device)
        models = {key:load_joint_checkpoint(selected[key[1]],spec,device)[0] for key,spec in specs.items()}
        retargeter = JointNNRetargeter(specs,models,device)
    else:
        if args.hand_model != 'l21':
            raise ValueError('Legacy NN checkpoints only support the L21 target')
        if not any(p is not None for p in paths):
            raise ValueError('Legacy NN requires explicit checkpoint selection')
        if args.hand != 'both' and (args.checkpoint is None or args.left_checkpoint is not None or args.right_checkpoint is not None):
            raise ValueError('Single-side legacy NN requires exactly --checkpoint')
        from teleoperation.applications.hand_realtime import load_palm_local_retargeter
        from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
        device = resolve_device(args.device)
        space = 'world' if args.input_profile == LEGACY_WORLD_PROFILE else 'normalized'
        model = load_palm_local_retargeter(args.checkpoint,device,left_checkpoint=args.left_checkpoint,
                                         right_checkpoint=args.right_checkpoint,landmark_space=space)
        if args.hand == 'both' and args.checkpoint is not None:
            from teleoperation.data.checkpoint import read_checkpoint
            metadata = read_checkpoint(args.checkpoint,'cpu')
            if metadata.get('hand_side') != 'shared':
                raise ValueError('One checkpoint for both targets requires explicit shared-side metadata')
        retargeter = PoseTransformerRetargeter(model,device)
    return specs, retargeter


def _commands(args, specs, scene=None):
    initial, guards = {}, {}
    if scene is not None and all(s.hand_model == 'l21' for s in specs.values()):
        from teleoperation.simulation.hand_commands import L21CollisionGuard
        for key,spec in specs.items():
            named = scene.joint_values(spec.side)
            initial[key] = [named[n] for n in spec.independent_joint_names]
            if not args.no_collision_guard:
                guards[key] = L21CollisionGuard(scene,spec.side,args.collision_margin_mm/1000)
    return JointCommandPipeline(specs,filter_mode=args.joint_filter,alpha=args.filter_alpha,
                                max_velocity=args.joint_max_velocity,kalman_q=args.kalman_q,
                                kalman_r=args.kalman_r,initial=initial,guards=guards)


def _scene(specs, headless):
    sides = tuple(spec.side for spec in specs.values())
    model = next(iter(specs.values())).hand_model
    if model == 'l21':
        from teleoperation.applications.realtime_hand_sim import RealtimeL21Scene
        scene = RealtimeL21Scene(sides,gui=not headless)
    else:
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        scene = ThreeRobotHandScene(sides,gui=not headless,robot_types=(model,))
    return scene, HandExecutor(specs,{key:scene.apply_native for key in specs})


def _metadata(args, commands):
    metadata = {'backend':args.backend,'input_profile':args.input_profile,'hand_model':args.hand_model,
            'checkpoint':None if args.checkpoint is None else str(args.checkpoint),
            'left_checkpoint':None if args.left_checkpoint is None else str(args.left_checkpoint),
            'right_checkpoint':None if args.right_checkpoint is None else str(args.right_checkpoint),
            'command_policy':commands.summary()}
    if args.input_profile == PALM_LENGTH_PROFILE:
        metadata['input_contract'] = palm_length_metadata()
    if args.backend == 'dex':
        from teleoperation.retargeting.hand.dex_native import VECTOR_PARAMETERS
        metadata['algorithm_parameters'] = {**VECTOR_PARAMETERS,'target_units':'m',
            'objective_relation':'metric_cost = reference_palm_length_m * dimensionless_cost',
            'huber_delta_rule':'.02 * reference_palm_length_m',
            'ftol_abs_rule':'1e-6 * reference_palm_length_m','xtol_abs_units':'rad'}
    return metadata


def _process(window, specs, retargeter, commands, *, live=False):
    solved = retargeter.retarget(window)
    results = {key:solved[key] for key in specs}
    accepted = {key:commands.update(result,timestamp=None if live else window.timestamp) for key,result in results.items()}
    targets = {}
    if window.metadata.get('input_profile') == PALM_LENGTH_PROFILE:
        from teleoperation.retargeting.hand.hand_model import build_hand_targets
        targets = {key:build_hand_targets(window,spec) for key,spec in specs.items()}
    return results, accepted, targets


def export(args):
    from teleoperation.learning.dataset import TwoHandH5Dataset
    specs, retargeter = _selection(args)
    profile = PALM_LENGTH_PROFILE if args.input_profile == PALM_LENGTH_PROFILE else None
    dataset = TwoHandH5Dataset(args.input,input_profile=profile)
    expected = palm_length_metadata() if profile else palm_local_metadata('world' if args.input_profile == LEGACY_WORLD_PROFILE else 'normalized')
    if dataset.coordinate_alignment != expected['coordinate_alignment']:
        raise ValueError('Export data and checkpoint input profile differ; explicitly align the source H5')
    samples = {row['frame_index']:row for row in dataset.samples}
    with ExitStack() as stack:
        scene = None
        if args.hand_model == 'l21' and not args.no_collision_guard:
            scene, _ = _scene(specs,True)
            stack.callback(scene.close)
        commands = _commands(args,specs,scene)
        metadata = _metadata(args,commands)
        from teleoperation.learning.trainer import _sha256
        metadata['input_sha256'] = _sha256(args.input)
        writer = stack.enter_context(HandJointResultWriter(args.output,specs,metadata))
        for index,(frame_id,timestamp) in enumerate(zip(dataset.frame_ids,dataset.timestamps)):
            sample = samples.get(index)
            hands = {side:None if sample is None or not sample[side+'_valid'] else sample[side+'_input'] for side in ('left','right')}
            metadata = {**expected, **({} if sample is None else sample['window_metadata'])}
            window = HandWindow(hands,float(timestamp),'h5',metadata)
            results, accepted, targets = _process(window,specs,retargeter,commands)
            writer.append(int(frame_id),window.timestamp,results,accepted,targets,metadata.get('palm_length_m'))
    print(f'saved={args.output} frames={dataset.frame_count} targets={list(specs)}')
    return 0


def run(args):
    from teleoperation.applications.hand_realtime import MediaPipeCameraWorkflow
    from teleoperation.applications.realtime_hand_sim import resolve_model_asset
    if args.frames < 0:
        raise ValueError('--frames must be nonnegative')
    specs, retargeter = _selection(args)
    asset = resolve_model_asset(args.model_asset_path)
    _camera_dependencies()
    # Validate command options before allocating devices.
    _commands(args,specs)
    if args.record_results is not None and args.record_results.exists():
        raise FileExistsError(args.record_results)
    profile = PALM_LENGTH_PROFILE if args.input_profile == PALM_LENGTH_PROFILE else None
    space = 'normalized' if args.input_profile == LEGACY_NORMALIZED_PROFILE else 'world'
    count = valid = 0
    with ExitStack() as stack:
        scene, executor = _scene(specs,args.headless)
        stack.callback(scene.close)
        commands = _commands(args,specs,scene)
        camera = stack.enter_context(MediaPipeCameraWorkflow(asset,args.camera_index,landmark_space=space,input_profile=profile))
        writer = None if args.record_results is None else stack.enter_context(HandJointResultWriter(args.record_results,specs,_metadata(args,commands)))
        try:
            while scene.alive() and (args.frames == 0 or count < args.frames):
                window = camera.next_window()
                timestamp = time.time()-camera._start_time if window is None else window.timestamp
                if window is None:
                    metadata = palm_length_metadata() if profile else palm_local_metadata(space)
                    window = HandWindow({'left':None,'right':None},timestamp,'mediapipe',metadata)
                # The same acquisition time is stored on both result and command.
                results, accepted, targets = _process(window,specs,retargeter,commands)
                for command in accepted.values():
                    executor.apply(command)
                if writer is not None:
                    writer.append(count,timestamp,results,accepted,targets,window.metadata.get('palm_length_m'))
                valid += sum(row.valid for row in results.values())
                count += 1
                scene.step()
        except KeyboardInterrupt:
            pass
        print(json.dumps({'frames':count,'valid_solutions':valid,'commands':commands.summary()},ensure_ascii=False))
    return 0


def _camera_dependencies():
    import importlib
    for name in ('cv2','mediapipe'):
        importlib.import_module(name)


def list_models(args):
    for name in HAND_MODELS:
        sides = {side:load_hand_spec(name,side) for side in ('left','right')}
        print(json.dumps({'hand_model':name,'sides':{side:{'independent_dof':spec.dof,'expanded_joints':len(spec.joint_names)} for side,spec in sides.items()},
                          'backends':{'nn':True,'dex':True,'geometric':name=='l21'},
                          'nn_requires_target_weights':True,'performance_acceptance':'pending'},ensure_ascii=False))
    return 0


def replay(args):
    specs = {(args.hand_model,s):load_hand_spec(args.hand_model,s) for s in _sides(args.hand)}
    # Validate the whole recording before opening the GUI, then stream it again.
    for _ in iter_joint_commands(args.input,specs):
        pass
    scene,executor = _scene(specs,args.headless)
    previous = None
    try:
        for timestamp,commands in iter_joint_commands(args.input,specs):
            if not scene.alive(): break
            if not args.headless and previous is not None:
                time.sleep(max(0.,timestamp-previous))
            for command in commands.values(): executor.apply(command)
            scene.step()
            previous = timestamp
    finally:
        scene.close()
    return 0
