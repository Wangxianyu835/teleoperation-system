"""Shared SMPL-X arm mapping, wrist and bounded command step."""
import numpy as np


def measure_camera_arm_geometry(driver, anchor='upper'):
    links, anchors = {}, {}
    for name, adapter in driver.ik.items():
        links[name], anchors[name] = {}, {}
        for side in driver.sides:
            points = adapter.arm_points(side)
            shoulder, elbow, wrist = (np.asarray(points[key], dtype=float) for key in (anchor, 'elbow', 'wrist'))
            links[name][side] = dict(upper=float(np.linalg.norm(elbow - shoulder)), fore=float(np.linalg.norm(wrist - elbow)))
            anchors[name][side] = dict(shoulder=list(adapter.to_base(shoulder)),
                                       elbow=list(adapter.to_base(elbow)), wrist=list(adapter.to_base(wrist)))
    return links, anchors

def rigid_elbow_robots(per_side):
    return {name for name, target in per_side.items() if target.elbow is not None
            and (target.info or {}).get('elbow_target') == 'posture'}


def _camera_arm_frame(bridge, driver, wrist_control, body, wrists, dt_s, *,
                           elbow_follow='operator', on_error=None, execute=True):
    metadata = getattr(body, 'metadata', {})
    session = metadata.get('session_id')
    if session:
        identity = (session, metadata.get('model_hash'))
        previous = vars(bridge).get('_camera_identity')
        if previous is not None and previous != identity:
            bridge.reset(recalibrate=True)
            wrist_control.reset()
        bridge._camera_identity = identity
    commands = {}
    targets = bridge.update(arms_world=body, hands=None, frame_dt=dt_s)
    orientations = wrist_control.update(wrists, dt_s)
    bends = bridge.elbow_bend_target() if elbow_follow == 'operator' else None
    neutral = bridge.elbow_bend_neutral() if bends is not None else None
    driver.update_slew(dt_s)
    for side in driver.sides:
        # Missing body data must not even enter IK: its warm starts temporarily
        # move the displayed robot although the accepted command stays held.
        if not execute and (body is None or not body.valid.get(side, False)):
            continue
        per_side = {name: values[side] for name, values in targets.items()}
        positions = {name: target.position for name, target in per_side.items() if target.position is not None}
        if len(positions) != len(per_side):
            continue
        elbows = {name: target.elbow for name, target in per_side.items() if target.elbow is not None} or None
        baseline = None if neutral is None else {
            name: neutral[side] if (target.info or {}).get('posture_frame') == 'delta' else None
            for name, target in per_side.items()}
        try:
            operation = driver.apply_each if execute else driver.solve_each
            commands[side] = operation(side, positions, source='base', orientation=orientations[side],
                              orientation_frame='absolute', elbows=elbows,
                              elbow_rigid=rigid_elbow_robots(per_side) or None,
                              elbow_bend_baseline=baseline,
                              elbow_bend_deg=None if bends is None else bends[side])
        except Exception as error:
            if on_error is None:
                raise
            on_error(side, error)
    return targets, commands

def apply_camera_arm_frame(bridge, driver, wrist_control, body, wrists, dt_s, **kwargs):
    return _camera_arm_frame(bridge, driver, wrist_control, body, wrists, dt_s, **kwargs)[0]


def solve_camera_arm_frame(bridge, driver, wrist_control, body, wrists, dt_s, **kwargs):
    return _camera_arm_frame(bridge, driver, wrist_control, body, wrists, dt_s, execute=False, **kwargs)
