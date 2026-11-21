"""Compose the preprint MediaPipe/Paper mapping with native robot commands."""
import numpy as np

from teleoperation.contracts.observations import RawTeleopObservation
from teleoperation.contracts.robot_control import RobotPartCommand
from teleoperation.simulation.arm_profiles import profile_for
from teleoperation.simulation.arm_realtime import RealtimeArmBridge


class MediaPipeRobotArms:
    def __init__(self, backends, args):
        self.backends = dict(backends)
        self.sides = ('left', 'right') if args.side == 'both' else (args.side,)
        links, anchors = {}, {}
        for name, backend in self.backends.items():
            driver = backend.driver
            driver.orientation_scope = args.arm_orientation_scope
            driver.slew_m_per_s = .6
            driver.lock_others = True
            links[name], anchors[name] = {}, {}
            for adapter in driver.ik.values():
                adapter.ik_solver = args.arm_ik_solver
                adapter.ik_mode = 'pose'
                adapter.orientation_scope = args.arm_orientation_scope
                adapter.ik_lock_others = True
                for side in self.sides:
                    points = adapter.arm_points(side)
                    shoulder = np.asarray(points['upper'])
                    elbow = np.asarray(points['elbow'])
                    wrist = np.asarray(points['wrist'])
                    links[name][side] = {
                        'upper': float(np.linalg.norm(elbow - shoulder)),
                        'fore': float(np.linalg.norm(wrist - elbow)),
                    }
                    anchors[name][side] = {
                        'shoulder': list(adapter.to_base(shoulder)),
                        'elbow': list(adapter.to_base(elbow)),
                        'wrist': list(adapter.to_base(wrist)),
                    }
            driver.set_shoulder_anchors({name: anchors[name]})
        self.bridge = RealtimeArmBridge(
            profiles=tuple(profile_for(name) for name in self.backends),
            sides=self.sides, calibration_frames=args.arm_calibrate_frames,
            calibrate_mode=args.arm_calibrate_mode, map_mode='posture',
            orientation_mode='hand_base', robot_links=links, anchors=anchors,
        )

    def solve(self, raw, dt):
        updates = {name: {} for name in self.backends}
        if not isinstance(raw, RawTeleopObservation):
            return updates
        targets = self.bridge.update(
            arms_world=raw.upper_body.world_arms, hands=raw.hands.world_hands,
            frame_dt=dt,
        )
        orientations = self.bridge.orientation_target()
        hand_frames = self.bridge.hand_frame_target()
        bends = self.bridge.elbow_bend_target()
        for name, backend in self.backends.items():
            driver = backend.driver
            driver.update_slew(dt)
            for side in self.sides:
                target = targets[name][side]
                if target.position is None or not raw.upper_body.valid[side]:
                    continue
                reports = driver.solve_each(
                    side, {name: target.position}, source='base',
                    orientation=orientations[side], orientation_frame='neutral',
                    elbows={} if target.elbow is None else {name: target.elbow},
                    elbow_bend_deg=bends[side], hand_frame=hand_frames[side],
                )
                names = backend.spec.parts[side + '_arm'].joint_names
                updates[name][side + '_arm'] = RobotPartCommand(
                    names, [reports[name]['joints'][joint] for joint in names],
                    raw.upper_body.timestamp, 'unix_seconds', True, 'tracking',
                )
        return updates

    def summary(self):
        return {'algorithm': 'mediapipe-posture-paper', 'sides': list(self.sides),
                'calibration': self.bridge.status(),
                'robots': {name: {'ik_solver': backend.driver.ik[name].ik_solver,
                                  'ik_mode': backend.driver.ik[name].ik_mode}
                           for name, backend in self.backends.items()}}
