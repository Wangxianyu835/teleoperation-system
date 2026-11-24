"""Explicit test composition for canonical-window algorithm tests."""
from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.retargeting.hand.topology import ensure_hand25


class CanonicalWindowFixture:
    def __init__(self, receptive_field=3, scale_factor=1.0):
        self.scale_factor = scale_factor
        self.buffer = TemporalBuffer(receptive_field, reset_on_missing=False)
    def update(self, left_hand=None, right_hand=None, timestamp=None, source="unknown", metadata=None):
        hands = {side: None if points is None else ensure_hand25(points, self.scale_factor) for side, points in (("left", left_hand), ("right", right_hand))}
        return self.update_canonical(hands["left"], hands["right"], timestamp, source, metadata)
    def update_canonical(self, left_hand=None, right_hand=None, timestamp=None, source="unknown", metadata=None):
        window = self.buffer.append(CanonicalHandFrame({"left": left_hand, "right": right_hand}, timestamp, source, metadata or {}))
        return None if window is None else window.to_payload()
    def reset(self): self.buffer.reset()
    def reset_side(self, side): self.buffer.reset_side(side)


class Tron2ASceneFixture:
    """Exercise the shared scene/output seam with the original physical assertions."""

    def __init__(self, urdf_path, *, gui=False, display='simplified', attach_hands=False,
                 safe_q=None, hand_mount_pose=None, hand_reference_only=False):
        from types import SimpleNamespace
        import numpy as np
        from teleoperation.robots.control_spec import load_robot_control_spec
        from teleoperation.simulation.robot_backends import SimulationGroup
        from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
        from teleoperation.simulation.robot_output import RobotOutput
        self.group = SimulationGroup(gui=gui)
        self.p, self.client = self.group.scene.p, self.group.scene.cid
        self.time_step = self.group.time_step
        self._temporary = self.group.scene._temporary
        try:
            self.spec = load_robot_control_spec('tron2a_l21', resource_overrides={
                'urdf_path': urdf_path, 'parts': 'both' if attach_hands else 'arms'})
            calibration = SimpleNamespace(
                safe_q=safe_q or {s: np.zeros(7) for s in ('left','right')},
                hand_mount_pose=hand_mount_pose or {s:[0,0,0,0,0,0,1] for s in ('left','right')})
            self.backend = RobotPartSceneBuilder().build(self.spec, group=self.group,
                calibration=calibration, geometry=display)
            self.output = RobotOutput({'tron2a_l21':self.backend})
            self.robot = self.backend.robot['body']
            self.joint_indices = self.backend.names[self.robot]
            self.arm_indices = {s:[self.joint_indices[n] for n in self.spec.parts[s+'_arm'].joint_names]
                                for s in ('left','right')}
            self.hand_bodies = {s:self.backend.part_bodies[s+'_hand'] for s in self.spec.hand_specs}
            self.hand_indices = {s:[self.backend.names[b][n] for n in self.spec.parts[s+'_hand'].joint_names]
                                 for s,b in self.hand_bodies.items()}
            self.joint_info = {i:self.p.getJointInfo(self.robot,i,physicsClientId=self.client)
                               for i in self.joint_indices.values()}
            self._frame = 0
        except BaseException:
            self.group.close()
            raise

    def apply_arms(self, joints):
        from dataclasses import replace
        from teleoperation.contracts.robot_control import RobotPartCommand
        frame = self.spec.reference_frame(self._frame, self._frame*self.time_step)
        self._frame += 1
        parts = dict(frame.parts)
        for side, q in joints.items():
            part = self.spec.parts[side+'_arm']
            parts[side+'_arm'] = RobotPartCommand(part.joint_names,q,None,'none',False,'holding')
        frame = replace(frame,parts=parts)
        self.backend.validate(frame)
        self.backend.write(frame)

    def actual_state(self):
        record = self.backend.robot
        drift = {self.joint_info[i][1].decode():self.p.getJointState(
            self.robot,i,physicsClientId=self.client)[0]-q
            for i,q in zip(record['hold_ids'],record['hold_targets'])}
        return {'arms':self.backend.read_arm_state(),'noncontrolled_joint_drift':drift}

    def step(self): self.group.advance(1)
    def is_connected(self): return self.group.alive()
    def close(self): self.group.close()
