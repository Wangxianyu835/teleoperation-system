"""Controlled GR1 arm diagnosis; production and upstream algorithms stay unchanged."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
DEPENDENCIES = ROOT / 'outputs/acceptance/teleopbench-gr1-20261111/dependencies'
SIDES = ('left', 'right')
COMMIT = '4cc410d149ce63c86f485b257cba3b3e0ec031d6'
URDF = 'assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf'
URDF_SHA = '56e9373e8a2023cf920fedc7913281a010d0ffbb38731d7d015a5307cd30e6d3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_error(actual, target):
    for matrix in (actual, target):
        matrix = np.asarray(matrix)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('Pose must be a finite 4x4 transform')
        r = matrix[:3, :3]
        if (not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(r), 1., atol=1e-6)
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)):
            raise ValueError('Pose rotation must belong to SO(3)')
    return (float(1000 * np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude())))


def candidate_status(q, lower, upper):
    if not np.isfinite(q).all(): return 'nonfinite'
    if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6): return 'out_of_limits'
    return 'within_limits'


def measure_candidate(adapter, side, values):
    """Read an IK candidate in world coordinates without changing live state."""
    if len(values) != len(adapter.indices) or not np.isfinite(values).all():
        raise ValueError('Candidate must cover every movable joint with finite values')
    p = adapter.p
    saved = [p.getJointState(adapter.body, k, physicsClientId=adapter.cid)[:2] for k in adapter.indices]
    try:
        adapter._teleport(side, values)
        position, quat = adapter.measure(side)
        t = np.eye(4)
        t[:3, 3] = position
        t[:3, :3] = Rotation.from_quat(quat).as_matrix()
        return t
    finally:
        for k, (q, v) in zip(adapter.indices, saved):
            p.resetJointState(adapter.body, k, q, targetVelocity=v, physicsClientId=adapter.cid)


def arm_name(name):
    return name.startswith(('left_', 'right_')) and any(k in name for k in ('shoulder', 'elbow', 'wrist'))


class ControlledArmModel:
    """Original Pinocchio chain vs geometry-free Bullet copy, both with 14 arm DoF."""
    def __init__(self, upstream, work):
        import pinocchio as pin
        import pybullet as bullet
        self.pin, self.p = pin, bullet
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.upstream = Path(upstream).resolve()
        revision = subprocess.check_output(['git', '-c', 'safe.directory=' + self.upstream.as_posix(),
                                            'rev-parse', 'HEAD'], cwd=self.upstream, text=True).strip()
        source = self.upstream / URDF
        if revision != COMMIT or digest(source) != URDF_SHA:
            raise ValueError('Official commit or model hash differs from pinned baseline')
        full = pin.buildModelFromUrdf(str(source))
        locked = [i for i, name in enumerate(full.names) if i and not arm_name(name)]
        self.model = pin.buildReducedModel(full, locked, pin.neutral(full))
        self.data = self.model.createData()
        self.names = tuple(self.model.names[1:])
        if len(self.names) != 14 or self.model.nq != 14:
            raise ValueError('Expected fourteen scalar arm joints')
        center = (self.model.lowerPositionLimit + self.model.upperPositionLimit) / 2
        half = .45 * (self.model.upperPositionLimit - self.model.lowerPositionLimit)
        self.lower, self.upper = center - half, center + half
        self.model.lowerPositionLimit = self.lower.copy()
        self.model.upperPositionLimit = self.upper.copy()
        margin = .05 * (self.upper - self.lower)
        self.reference = np.clip(np.zeros(14), self.lower + margin, self.upper - margin)
        for i, name in enumerate(self.names):
            if 'elbow_pitch' in name:
                self.reference[i] = np.clip(-.8, self.lower[i] + margin[i], self.upper[i] - margin[i])
        tree = ET.parse(source)
        for link in tree.getroot().findall('link'):
            for node in list(link.findall('visual')) + list(link.findall('collision')):
                link.remove(node)
        for joint in tree.getroot().findall('joint'):
            name = joint.get('name')
            if name in self.names:
                k = self.names.index(name)
                joint.find('limit').set('lower', str(self.lower[k]))
                joint.find('limit').set('upper', str(self.upper[k]))
            elif joint.get('type') != 'fixed':
                joint.set('type', 'fixed')
                for tag in ('axis', 'limit', 'mimic', 'dynamics'):
                    for node in joint.findall(tag): joint.remove(node)
        derived = self.work / 'controlled_model.urdf'
        tree.write(derived, encoding='utf-8', xml_declaration=True)
        self.cid = bullet.connect(bullet.DIRECT)
        self.body = bullet.loadURDF(str(derived), useFixedBase=True, physicsClientId=self.cid)
        self.indices = {bullet.getJointInfo(self.body, i, physicsClientId=self.cid)[1].decode(): i
                        for i in range(bullet.getNumJoints(self.body, physicsClientId=self.cid))}
        self.links = {bullet.getJointInfo(self.body, i, physicsClientId=self.cid)[12].decode(): i
                      for i in range(bullet.getNumJoints(self.body, physicsClientId=self.cid))}
        self.frames = {'left': 'L_hand_base_link', 'right': 'R_hand_base_link'}
        self.set_bullet(self.reference)

    def close(self):
        if self.p.isConnected(self.cid): self.p.disconnect(self.cid)

    def bullet_state(self):
        return np.array([self.p.getJointState(self.body, self.indices[n], physicsClientId=self.cid)[:2]
                         for n in self.names])

    def set_bullet(self, q):
        for name, value in zip(self.names, q):
            self.p.resetJointState(self.body, self.indices[name], float(value), physicsClientId=self.cid)

    @contextmanager
    def restored(self):
        before = self.bullet_state()
        try:
            yield
        finally:
            for name, (q, v) in zip(self.names, before):
                self.p.resetJointState(self.body, self.indices[name], float(q), targetVelocity=float(v), physicsClientId=self.cid)

    def pin_fk(self, q, side):
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.model.getFrameId(self.frames[side])].homogeneous.copy()

    def bullet_fk(self, q, side):
        with self.restored():
            self.set_bullet(q)
            state = self.p.getLinkState(self.body, self.links[self.frames[side]],
                                       computeForwardKinematics=True, physicsClientId=self.cid)
            t = np.eye(4)
            t[:3, 3] = state[4]
            t[:3, :3] = Rotation.from_quat(state[5]).as_matrix()
            return t

    def manifest(self):
        return {'official_commit': COMMIT, 'original_model_sha256': URDF_SHA,
                'derived_model_sha256': digest(self.work / 'controlled_model.urdf'),
                'joint_names': self.names, 'lower': self.lower.tolist(), 'upper': self.upper.tolist(),
                'reference': self.reference.tolist(), 'frames': self.frames,
                'dt': 1/30, 'updates_per_case': 60, 'input_hold_seconds': 2,
                'clock_contrast_integrated_seconds': .3, 'position_gate_mm': 10, 'orientation_gate_deg': 5,
                'diagnostic_overrides': ['non-arm joints fixed at zero', 'central 90 percent arm limits',
                                         'geometry removed', 'same endpoint in both engines']}


def generate_cases(model):
    result = [('reference', model.reference.copy())]
    margin = .01 * (model.upper - model.lower)
    for i, name in enumerate(model.names):
        for delta in (-.1, .1):
            q = model.reference.copy()
            q[i] = np.clip(q[i] + delta, model.lower[i] + margin[i], model.upper[i] - margin[i])
            result.append((f'{name}:{delta:+.1f}', q))
    return result


def run_fk(model):
    rows = []
    for case, q in generate_cases(model):
        for side in SIDES:
            mm, deg = pose_error(model.bullet_fk(q, side), model.pin_fk(q, side))
            rows.append({'case': case, 'side': side, 'position_mm': mm, 'orientation_deg': deg})
    if any(r['position_mm'] > .01 or np.deg2rad(r['orientation_deg']) > 1e-4 for r in rows):
        raise RuntimeError('FK agreement gate failed; do not compare IK yet')
    return rows


class BulletSolver:
    def __init__(self, model, pose):
        from teleoperation.simulation.arm_ik import PyBulletArmIK
        self.model, self.pose = model, pose
        self.candidates = []
        owner = self

        class TracedIK(PyBulletArmIK):
            def _ik_call(self, side, world, orientation, arguments):
                values = super()._ik_call(side, world, orientation, arguments)
                q = np.array([values[self.position_of[model.indices[name]]] for name in model.names])
                owner.candidates.append({'kind': 'position' if orientation is None else 'pose',
                                         'q': q.tolist(), 'fk': model.pin_fk(q, side).tolist()})
                return values

        with model.restored():
            model.set_bullet(model.reference)
            self.adapter = TracedIK(model.body, model.cid, robot_type='gr1_t2',
                                    ik_mode='position', rest_poses='loaded', lock_others=True,
                                    elbow_recovery_deg=0)
            self.adapter.ee = {s: (model.links[f], f) for s, f in model.frames.items()}
            self.adapter.neutral_orientation = {s: self.adapter._orientation(s) for s in SIDES}

    def solve(self, q, target, side, dt=1/30):
        pose_error(target, target)
        self.candidates = []
        start = time.perf_counter()
        with self.model.restored():
            self.model.set_bullet(q)
            values = self.adapter.solve(side, target[:3, 3],
                orientation=Rotation.from_matrix(target[:3, :3]).as_quat() if self.pose else None,
                source='world')
        command = np.array(q, copy=True)
        for name, value in values.items(): command[self.model.names.index(name)] = value
        return command, {'candidates': list(self.candidates),
                         'production': dict(self.adapter.last_solve[side]),
                         'solve_ms': (time.perf_counter()-start)*1000}


class PinkSolver:
    def __init__(self, model, pose):
        sys.path.insert(0, str(DEPENDENCIES))
        import pink
        import qpsolvers
        from pink.tasks import FrameTask, PostureTask
        self.model, self.pink = model, pink
        self.qp = qpsolvers.available_solvers[0]
        self.sides = {}
        for side in SIDES:
            inactive = [i for i,n in enumerate(model.model.names) if i and not n.startswith(side+'_')]
            reduced = model.pin.buildReducedModel(model.model, inactive, model.reference)
            data = reduced.createData()
            ids = [model.names.index(name) for name in reduced.names[1:]]
            config = pink.Configuration(reduced, data, model.reference[ids].copy())
            frame = FrameTask(frame=model.frames[side], position_cost=1., orientation_cost=1. if pose else 0., lm_damping=.5)
            posture = PostureTask(cost=.001, lm_damping=1., gain=.1)
            posture.set_target(model.reference[ids].copy())
            self.sides[side] = (config, frame, posture, ids)

    def solve(self, q, target, side, dt=1/30):
        pose_error(target, target)
        config, frame, posture, ids = self.sides[side]
        config.update(q[ids].copy())
        frame.set_target(self.model.pin.SE3(target[:3, :3], target[:3, 3]))
        start = time.perf_counter()
        velocity = self.pink.solve_ik(config, [frame, posture], dt, solver=self.qp)
        config.integrate_inplace(velocity, dt)
        command = np.array(q, copy=True)
        command[ids] = config.q
        return command, {'candidates': [], 'production': {},
                         'solve_ms': (time.perf_counter()-start)*1000, 'qp': self.qp}


def summarize(rows):
    result = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group and r['step'] == 59]
        result[group] = {'cases': len(subset),
            'position_median_mm': float(np.median([r['position_mm'] for r in subset])),
            'position_p95_mm': float(np.percentile([r['position_mm'] for r in subset], 95)),
            'orientation_median_deg': float(np.median([r['orientation_deg'] for r in subset])),
            'passes_position_10mm': sum(r['position_mm'] <= 10 for r in subset),
            'passes_pose_10mm_5deg': sum(r['position_mm'] <= 10 and r['orientation_deg'] <= 5 for r in subset),
            'rejected_updates': sum(bool(r['rejection']) for r in rows if r['group'] == group)}
    return result


def run_matrix(model):
    rows, traces = [], []
    for group, factory, pose, dt in [('A_position_bullet', BulletSolver, False, 1/30),
                                    ('B_pose_bullet', BulletSolver, True, 1/30),
                                    ('C_position_pink', PinkSolver, False, 1/30),
                                    ('D_pose_pink', PinkSolver, True, 1/30),
                                    ('C_position_pink_5ms', PinkSolver, False, .005),
                                    ('D_pose_pink_5ms', PinkSolver, True, .005)]:
        solver = factory(model, pose)
        for case, goal_q in generate_cases(model):
            for side in SIDES:
                target = model.pin_fk(goal_q, side)
                q = model.reference.copy()
                for step in range(60):
                    command, detail = solver.solve(q, target, side, dt)
                    mm, deg = pose_error(model.pin_fk(command, side), target)
                    production = detail['production']
                    row = {'group': group, 'case': case, 'side': side, 'step': step, 'dt': dt,
                           'position_mm': mm, 'orientation_deg': deg,
                           'joint_step_deg': float(np.rad2deg(np.max(np.abs(command-q)))),
                           'limit_excess_rad': float(max(0., np.max(model.lower-command), np.max(command-model.upper))),
                           'rejection': production.get('wrist_rejection') or '', 'solve_ms': detail['solve_ms']}
                    rows.append(row)
                    if step == 0 or row['rejection']:
                        traces.append({'group': group, 'case': case, 'side': side, 'step': step,
                                       'start_q': q.tolist(), 'target': target.tolist(), 'command': command.tolist(),
                                       **detail})
                    q = command
        print('matrix complete', group, flush=True)
    with (model.work/'matrix.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (model.work/'candidates.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    summary = summarize(rows)
    (model.work/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_probes(model):
    solver = BulletSolver(model, True)
    rng = np.random.RandomState(20261111)
    records = []
    for index in range(48):
        goal = model.lower + (.15 + .7 * rng.uniform(size=14)) * (model.upper - model.lower)
        for side in SIDES:
            target = model.pin_fk(goal, side)
            command, details = solver.solve(model.reference.copy(), target, side)
            old_mm, _ = pose_error(model.pin_fk(model.reference, side), target)
            mm, deg = pose_error(model.pin_fk(command, side), target)
            for candidate in details['candidates']:
                candidate['status'] = candidate_status(np.array(candidate['q']), model.lower, model.upper)
                candidate['error_mm'], candidate['error_deg'] = pose_error(np.array(candidate['fk']), target)
            records.append({'case': index, 'side': side, 'goal_q': goal.tolist(),
                            'target': target.tolist(), 'before_error_mm': old_mm,
                            'accepted_error_mm': mm, 'accepted_error_deg': deg,
                            'command_q': command.tolist(), **details})
    rejection_counts = {}
    for record in records:
        reason = record['production'].get('wrist_rejection') or 'accepted'
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    witnesses = []
    for record in records:
        if record['production'].get('wrist_rejection') != 'pose_position_conflict': continue
        for gain in np.arange(.1, 1.01, .1):
            goal = model.reference + gain * (np.array(record['goal_q']) - model.reference)
            target = model.pin_fk(goal, record['side'])
            command, detail = solver.solve(model.reference.copy(), target, record['side'])
            if detail['production'].get('wrist_rejection') == 'pose_position_conflict':
                old_mm, _ = pose_error(model.pin_fk(model.reference, record['side']), target)
                witnesses.append((old_mm, record['case'], record['side'], float(gain), goal, target, detail))
    if witnesses:
        old_mm, case, side, gain, goal, target, detail = min(witnesses, key=lambda item: item[0])
        result = {'case': case, 'side': side, 'gain': gain, 'before_mm': old_mm,
                  'start_q': model.reference.tolist(), 'goal_q': goal.tolist(),
                  'target': target.tolist(), 'first_call': detail, 'after_60': {}}
        for name, method in [('bullet', solver), ('pink', PinkSolver(model, True))]:
            q = model.reference.copy()
            for step in range(60): q, final = method.solve(q, target, side)
            result['after_60'][name] = {'q': q.tolist(), 'error': pose_error(model.pin_fk(q, side), target),
                                        'last': final['production']}
        # Explicit counterfactual experiment, never a production acceptance override.
        q = model.reference.copy()
        history = []
        for step in range(60):
            command, detail = solver.solve(q, target, side)
            used = False
            if detail['production'].get('wrist_rejection') == 'pose_position_conflict':
                position = np.array(detail['candidates'][0]['q'])
                if candidate_status(position, model.lower, model.upper) == 'within_limits':
                    delta = position-q
                    command = q+delta*min(1., .35/max(np.max(np.abs(delta)), 1e-12))
                    used = True
            mm, deg = pose_error(model.pin_fk(command, side), target)
            history.append({'step': step, 'position_mm': mm, 'orientation_deg': deg,
                            'diagnostic_position_candidate_used': used})
            q = command
        result['diagnostic_position_progress'] = history
        (model.work/'witness.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (model.work/'probe.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    summary = {'known_reachable_cases': len(records), 'rejection_counts': rejection_counts,
               'minimized_witness_available': bool(witnesses),
               'note': 'All goals are FK of valid joints; solver failure does not prove unreachable.'}
    (model.work/'probe-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values): return {}
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values))}


def run_replay(input_path, output, mode='both', max_frames=0):
    """Native recording ablation: change orientation input only, preserve all other code."""
    import h5py
    from teleoperation.robots.control_spec import load_robot_control_spec
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from teleoperation.applications.robot_session import RobotControlSession
    from teleoperation.inputs.visionpro import VisionProReplayInput
    from teleoperation.paths import ROBOTS_ROOT
    with h5py.File(input_path, 'r') as handle:
        stamps = handle['timestamps'][:]
        heads = handle['head_transform'][:]
        wrists = np.stack([handle[s+'_wrist_transform'][:] for s in SIDES], axis=1)
    if not np.isfinite(stamps).all() or not np.all(np.diff(stamps) > 0):
        raise ValueError('Recording timestamps must be finite and strictly increasing')
    n = len(stamps) if max_frames == 0 else min(len(stamps), max_frames)
    modes = ('baseline', 'position') if mode == 'both' else (mode,)
    manifest = {'input': str(input_path.resolve()), 'input_sha256': digest(input_path),
                'source_frames': len(stamps), 'requested_frames': n,
                'native_urdf_sha256': digest(ROBOTS_ROOT/'gr1/urdf/robot.urdf'),
                'modes': modes, 'changed_variable': 'orientation argument to driver.solve_each only',
                'common_model_control': False,
                'note': 'Native replay ablation; distinct from common-model matrix experiment.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summaries = {}
    for mode in modes:
        spec = load_robot_control_spec('gr1_t2', resource_overrides={'parts': 'arms', 'side': 'both'})
        group = SimulationGroup(gui=False)
        try:
            backend = RobotPartSceneBuilder().build(spec, group=group)
            session = RobotControlSession({'gr1_t2': backend}, source_kind='visionpro')
            source = VisionProReplayInput(input_path, 'both')
            adapter = backend.driver.ik['gr1_t2']
            original_call = adapter._ik_call
            calls = {s: [] for s in SIDES}
            latest_solutions = {}
            original_arm_solve = adapter.solve
            def traced_solve(side, *args, **kwargs):
                result = original_arm_solve(side, *args, **kwargs)
                latest_solutions[side] = dict(result)
                return result
            adapter.solve = traced_solve

            def traced(side, world, orientation, arguments):
                values = original_call(side, world, orientation, arguments)
                candidate = {name: values[adapter.position_of[k]] for k,name in adapter.arm[side]}
                lower = np.array([spec.parts[side+'_arm'].limits[j][0] for j in range(7)])
                upper = np.array([spec.parts[side+'_arm'].limits[j][1] for j in range(7)])
                status = candidate_status(np.array(list(candidate.values())), lower, upper)
                measured = measure_candidate(adapter, side, values) if status != 'nonfinite' else None
                calls[side].append({'kind': 'position' if orientation is None else 'pose',
                                    'joint_names': list(candidate), 'q': list(candidate.values()),
                                    'status': status, 'world_fk': None if measured is None else measured.tolist()})
                return values

            adapter._ik_call = traced
            if mode == 'position':
                original_solve_each = backend.driver.solve_each
                def position_call(*args, **kwargs):
                    kwargs['orientation'] = None
                    return original_solve_each(*args, **kwargs)
                backend.driver.solve_each = position_call
            arrays = {k: np.full(shape, np.nan) for k,shape in {
                'target': (n,2,3), 'command_fk': (n,2,3), 'actual': (n,2,3), 'q': (n,14),
                'upper': (n,2,3), 'elbow': (n,2,3), 'orientation_error_deg': (n,2),
                'head_reference_target': (n,2,3)}.items()}
            rows, rejected = [], Counter()
            improved_but_held = Counter()
            processed = 0
            failure = None
            phase = 'acquisition'
            trace_file = (output/(mode+'-candidates.jsonl')).open('x', encoding='utf-8')
            try:
                for i in range(n):
                    phase = 'acquisition'
                    raw = source.next_observation()
                    seconds = float(stamps[i]-stamps[0])
                    calls = {s: [] for s in SIDES}
                    before = {s: adapter.measure(s) for s in SIDES}
                    started = time.perf_counter()
                    phase = 'command_generation'
                    batch = session.next_batch(raw, i, seconds)
                    elapsed = (time.perf_counter()-started)*1000
                    controller = session.wrists['gr1_t2']
                    for j,side in enumerate(SIDES):
                        tracker = controller.trackers[side]
                        anchor = controller.anchors.get(('gr1_t2', side))
                        if anchor is None: continue
                        target_base = anchor[0]+tracker.displacement
                        target_world = np.asarray(adapter.to_world(target_base))
                        orientation = controller.orientations[side]['gr1_t2']
                        accepted = batch['gr1_t2'].parts[side+'_arm']
                        qdict = dict(zip(accepted.joint_names, accepted.positions))
                        values = [qdict.get(adapter.name_of[k], adapter.p.getJointState(
                            adapter.body,k,physicsClientId=adapter.cid)[0]) for k in adapter.indices]
                        command_fk = measure_candidate(adapter, side, values)
                        arrays['q'][i,j*7:(j+1)*7] = accepted.positions
                        arrays['target'][i,j] = target_base
                        arrays['command_fk'][i,j] = adapter.to_base(command_fk[:3,3])
                        detail = dict(adapter.last_solve.get(side, {}))
                        reason = detail.get('wrist_rejection') or ''
                        if reason: rejected[side+':'+reason] += 1
                        mm = float(1000*np.linalg.norm(command_fk[:3,3]-target_world))
                        old_mm = float(1000*np.linalg.norm(np.asarray(before[side][0])-target_world))
                        deg = (float(np.rad2deg((Rotation.from_matrix(command_fk[:3,:3]) *
                            Rotation.from_quat(orientation).inv()).magnitude())) if orientation is not None else float('nan'))
                        arrays['orientation_error_deg'][i,j] = deg
                        for candidate in calls[side]:
                            if candidate['world_fk'] is not None:
                                t = np.array(candidate['world_fk'])
                                candidate['error_mm'] = float(1000*np.linalg.norm(t[:3,3]-target_world))
                                candidate['error_deg'] = (float(np.rad2deg((Rotation.from_matrix(t[:3,:3]) *
                                    Rotation.from_quat(orientation).inv()).magnitude())) if orientation is not None else None)
                        valid_positions = [c for c in calls[side] if c['kind']=='position' and c['status']=='within_limits']
                        better = bool(reason and valid_positions and valid_positions[0]['error_mm'] < old_mm)
                        if better: improved_but_held[side] += 1
                        arrays['head_reference_target'][i,j] = 1.25*(
                            np.array([wrists[i,j,1,3]-heads[i,1,3], -wrists[i,j,0,3]+heads[i,0,3],
                                      wrists[i,j,2,3]-heads[i,2,3]])+np.array([0,0,.42]))
                        rows.append({'frame': i, 'time_s': seconds, 'side': side, 'observed': accepted.observed,
                                     'position_mm': mm, 'orientation_deg': deg, 'before_position_mm': old_mm,
                                     'position_candidate_mm': valid_positions[0]['error_mm'] if valid_positions else float('nan'),
                                     'rejection': reason, 'better_position_candidate_held': better,
                                     'solve_ms_both_arms': elapsed, 'tracker_state': tracker.status})
                        trace_file.write(json.dumps({'frame': i, 'side': side, 'target_world': target_world.tolist(),
                            'target_orientation': orientation, 'command': accepted.positions.tolist(),
                            'production': detail, 'candidates': calls[side]})+'\n')
                    dt = 1/30 if i==0 else float(stamps[i]-stamps[i-1])
                    try:
                        session.output.submit(batch, steps=max(1,min(24,round(min(dt,.1)*240))))
                    except (ValueError, RuntimeError) as error:
                        failure = {'frame': i, 'error': str(error), 'phase': 'RobotOutput.submit'}
                        break
                    for j,side in enumerate(SIDES):
                        arrays['actual'][i,j] = adapter.to_base(adapter.measure(side)[0])
                        points = adapter.arm_points(side)
                        arrays['upper'][i,j] = adapter.to_base(points['upper'])
                        arrays['elbow'][i,j] = adapter.to_base(points['elbow'])
                    processed = i+1
                    if i % 400 == 0: print(mode, i, n, flush=True)
            except (ValueError, RuntimeError) as error:
                actual_q = {name: adapter.p.getJointState(adapter.body, k, physicsClientId=adapter.cid)[0]
                            for name,k in adapter.index_of.items()}
                failure = {'frame': i, 'error': str(error), 'phase': phase,
                           'actual_joint_positions': actual_q, 'returned_solutions': latest_solutions,
                           'production': adapter.last_solve, 'candidates': calls}
                trace_file.write(json.dumps({'failure': failure})+'\n')
                print(mode, 'stopped with recorded failure', i, str(error), flush=True)
            finally:
                trace_file.close(); source.close(); session.close()
            arrays['timestamps'] = stamps[:n]
            np.savez_compressed(output/(mode+'.npz'), **arrays)
            with (output/(mode+'.csv')).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['frame'])
                writer.writeheader(); writer.writerows(rows)
            summary = {'frames_processed': processed, 'requested_frames': n, 'failure': failure,
                       'rejections': dict(rejected), 'better_position_candidate_held': dict(improved_but_held),
                       'ends': list(adapter.ee.values()), 'sides': {}}
            for side in SIDES:
                subset = [r for r in rows if r['side']==side and r['time_s']>=2]
                summary['sides'][side] = {key: distribution([r[key] for r in subset]) for key in
                    ('position_mm','orientation_deg','position_candidate_mm','solve_ms_both_arms')}
            summaries[mode] = summary
            (output/(mode+'-summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
        finally:
            group.close()
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('fk', 'matrix', 'probe', 'replay'))
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('both', 'baseline', 'position'), default='both')
    parser.add_argument('--frames', type=int, default=0)
    args = parser.parse_args(argv)
    if args.frames < 0: parser.error('--frames must be nonnegative')
    if args.stage == 'replay' and (args.input is None or not args.input.is_file()):
        parser.error('--stage replay requires an existing --input H5')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == 'replay':
        results = run_replay(args.input, args.output, args.mode, args.frames)
        print(json.dumps(results, indent=2), flush=True)
        return int(any(row['failure'] is not None for row in results.values()))
    model = ControlledArmModel(args.upstream, args.output)
    try:
        rows = run_fk(model)
        (args.output/'manifest.json').write_text(json.dumps(model.manifest(), indent=2), encoding='utf-8')
        (args.output/'fk.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps({'stage': args.stage, 'fk_cases': len(rows),
                          'max_position_mm': max(r['position_mm'] for r in rows),
                          'max_orientation_deg': max(r['orientation_deg'] for r in rows)}), flush=True)
        if args.stage == 'matrix': print(json.dumps(run_matrix(model), indent=2), flush=True)
        if args.stage == 'probe': print(json.dumps(run_probes(model), indent=2), flush=True)
    finally:
        model.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
