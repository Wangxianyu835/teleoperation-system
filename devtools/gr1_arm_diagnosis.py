"""Controlled GR1 arm diagnosis; production and upstream algorithms stay unchanged."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
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
                'dt': 1/30, 'hold_seconds': 2, 'position_gate_mm': 10, 'orientation_gate_deg': 5,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('fk', 'matrix'))
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    model = ControlledArmModel(args.upstream, args.output)
    try:
        rows = run_fk(model)
        (args.output/'manifest.json').write_text(json.dumps(model.manifest(), indent=2), encoding='utf-8')
        (args.output/'fk.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps({'stage': args.stage, 'fk_cases': len(rows),
                          'max_position_mm': max(r['position_mm'] for r in rows),
                          'max_orientation_deg': max(r['orientation_deg'] for r in rows)}), flush=True)
        if args.stage == 'matrix': print(json.dumps(run_matrix(model), indent=2), flush=True)
    finally:
        model.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
