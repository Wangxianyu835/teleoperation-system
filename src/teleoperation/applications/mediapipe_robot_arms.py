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


class Tron2ACameraArms:
    """Translate calibrated arm results into the same commands as other robots."""

    def __init__(self, backend, controller):
        from collections import Counter
        self.backend, self.controller = backend, controller
        self.counts = {side: Counter() for side in controller.sides}
        self.errors = {side: {} for side in controller.sides}
        self.last_source = {side: None for side in controller.sides}
        self.was_observed = {side: False for side in controller.sides}

    def solve(self, raw, control_seconds, *, new_observation=True):
        body = raw.upper_body if new_observation and isinstance(raw, RawTeleopObservation) else None
        self.controller.update(body, control_seconds)
        parts = {}
        for side in self.controller.sides:
            safety = self.controller.controllers[side]
            diag = self.controller.diagnostics[side]
            observed = bool(diag['command_observed'])
            if observed:
                self.last_source[side] = body.timestamp
            if 'timeout_return' in safety.status:
                status = 'returning'
            elif observed:
                status = 'tracking'
            elif 'loss_braking' in safety.status and np.any(np.abs(safety.velocity) > 1e-10):
                status = 'braking'
            else:
                status = 'initialized' if safety.status == 'initial_hold' else 'holding'
            # Safety motion is an explicit update even when no sensor target is valid.
            parts[side + '_arm'] = RobotPartCommand(
                self.backend.spec.parts[side + '_arm'].joint_names, safety.q,
                self.last_source[side], 'unix_seconds', observed, status,
                source_is_new=bool(observed),
            )
            self.counts[side][status] += 1
            self.counts[side]['valid_observations' if diag['observation_valid'] else 'invalid_observations'] += 1
            if diag['ik_attempted']:
                self.counts[side]['ik_success' if diag['ik_success'] else 'ik_failure'] += 1
            self.counts[side]['workspace_rejected'] += int(diag['workspace_rejected'])
            self.counts[side]['orientation_held'] += int(diag['orientation_held'])
            if self.was_observed[side] and not observed:
                self.counts[side]['tracking_lost'] += 1
            if observed and not self.was_observed[side] and self.counts[side]['tracking_lost']:
                self.counts[side]['tracking_recovered'] += 1
            self.was_observed[side] = observed
        return parts

    def capture_actual(self):
        from teleoperation.retargeting.arm.ik import _pose_matrix, _pose_error
        actual = self.backend.read_arm_state()
        for side in self.controller.sides:
            diag = self.controller.diagnostics.get(side)
            if diag is None:
                continue
            solver = self.controller.solvers[side]
            target = _pose_matrix(diag['target_pose'])
            command_fk = solver.specification.robot.fkine(self.controller.controllers[side].q).A
            values = {'ik_position_error_m': diag['ik_position_error_m'],
                      'ik_orientation_error_rad': diag['ik_orientation_error_rad']}
            position, rotation = _pose_error(command_fk, _pose_matrix(actual[side]['pose']))
            values.update(actual_command_position_error_m=position, actual_command_orientation_error_rad=rotation)
            for prefix, pose in (('command', command_fk), ('actual', _pose_matrix(actual[side]['pose']))):
                position, rotation = _pose_error(target, pose)
                values[prefix + '_position_error_m'] = position
                values[prefix + '_orientation_error_rad'] = rotation
            for name, value in values.items():
                if value is not None and np.isfinite(value):
                    item = self.errors[side].setdefault(name, {'count': 0, 'sum': 0., 'max': 0.})
                    item['count'] += 1; item['sum'] += float(value)
                    item['max'] = max(item['max'], float(value))

    def summary(self):
        return {'ik_solver': 'roboticstoolbox-lm', 'mapping': 'calibrated-pose-world',
                'orientation_mode': self.controller.arm_retargeter.orientation_mode,
                'sides': {side: {'counts': dict(self.counts[side]),
                    'metrics': {name: {'mean': item['sum'] / item['count'], 'max': item['max']}
                                for name, item in self.errors[side].items()}}
                          for side in self.controller.sides}}


class CameraArmControllers:
    """One observation fans out to independently configured robot arm solvers."""

    def __init__(self, backends, args, tron_controller=None):
        native = {name: backend for name, backend in backends.items() if name != 'tron2a_l21'}
        self.native = MediaPipeRobotArms(native, args) if native else None
        self.tron = (Tron2ACameraArms(backends['tron2a_l21'], tron_controller)
                     if 'tron2a_l21' in backends else None)

    def validate_observation(self, raw):
        if self.tron is None or raw is None:
            return
        if not isinstance(raw, RawTeleopObservation):
            raise ValueError('TRON2A arms require combined same-frame camera observations')
        metadata = raw.hands.metadata
        if (metadata.get('arm_world_space') != 'mediapipe_pose_world_meters'
                or metadata.get('timestamp_unit') != 'unix_seconds'
                or raw.hands.timestamp != raw.upper_body.timestamp):
            raise ValueError('TRON2A camera source, timebase or same-frame timestamps differ')
        if raw.upper_body.coordinate_frame != 'mediapipe_pose_normalized' or raw.upper_body.units != 'normalized':
            raise ValueError('TRON2A requires combined normalized/world Pose data')

    def solve(self, raw, dt, *, control_seconds, new_observation=True):
        updates = self.native.solve(raw, dt) if self.native is not None and new_observation else {}
        if self.tron is not None:
            updates['tron2a_l21'] = self.tron.solve(raw, control_seconds, new_observation=new_observation)
        return updates

    def capture_actual(self):
        if self.tron is not None:
            self.tron.capture_actual()

    def summary(self):
        native = self.native.summary() if self.native is not None else {}
        robots = dict(native.get('robots', {}))
        if self.tron is not None:
            robots['tron2a_l21'] = self.tron.summary()
        return {'algorithm': 'per-robot-camera-arms', 'robots': robots,
                'calibration': native.get('calibration')}

    def preview_status(self):
        if self.tron is None:
            return 'camera arms: MediaPipe / paper'
        return 'TRON2A LM | ' + ' | '.join(
            f"{side}: {diag['status']} IK={'OK' if diag['ik_success'] else '--'}"
            for side,diag in self.tron.controller.diagnostics.items())
