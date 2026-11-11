"""Controlled GR1 arm diagnosis; production and upstream algorithms stay unchanged."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('fk',))
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
    finally:
        model.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
