"""Search for a physically reachable H1-2 grasp pose for ``pickcube``.

This is a development probe, not part of the camera control path. It places the
native hand around the task cube from several palm orientations, closes the fist,
and reports whether the cube follows a short upward wrist motion.
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p

import task_eval as evaluation


def rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def hand_frame(yaw_rad: float, mode: str, tilt_rad: float = 0.0) -> np.ndarray:
    """Return columns ``(finger_direction, thumb_side, palm_normal)``."""
    horizontal = rotation_z(yaw_rad) @ rotation_x(tilt_rad)
    if mode == "down":
        # Fingers point down; thumb side stays horizontal; palm faces forward.
        return np.column_stack(([0.0, 0.0, -1.0], horizontal @ [0.0, 1.0, 0.0],
                                horizontal @ [1.0, 0.0, 0.0]))
    if mode == "forward":
        # Fingers point forward; palm faces down for a top grasp.
        return np.column_stack((horizontal @ [1.0, 0.0, 0.0],
                                horizontal @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]))
    raise ValueError(mode)


def wait(p, cid, steps: int) -> None:
    for _ in range(steps):
        p.stepSimulation(physicsClientId=cid)


def try_grasp(yaw_deg: float, mode: str, wrist_dz: float,
              closure: str = "fist", strength: float = 1.0,
              recenter: bool = True, orientation_scope: str = "wrist",
              tilt_deg: float = 0.0, target_bias=(0.0, 0.0, 0.0),
              finger_friction: float = 1.5,
              attach_on_contact: bool = False) -> dict:
    from teleoperation.robots.native_hand import build_mapping, map_dofs
    from teleoperation.simulation.arm_ik import PyBulletArmIK
    from teleoperation.simulation.environment import SimulationEnv
    from teleoperation.simulation.robot_loader import read_joint_ranges

    env = SimulationEnv(robot_type="h1_2", task_name="pickcube",
                        render=False, record=False)
    cid = env.client
    try:
        env.reset(randomize=False)
        arm = evaluation.measure_arm("h1_2")
        task_scene = evaluation.TaskScene(
            cid, env.task, robot="h1_2", shoulder=arm["shoulder_m"],
            reach=arm["reach_m"], name="pickcube", robot_id=env.robot_id)
        task_scene.prepare(
            120, ik_workspace=True, side="right", hold_robot=True,
            layout_forward=0.36, layout_drop=0.18)
        evaluation.disable_robot_table_collision(
            env, extra_bodies=task_scene.extra_bodies)

        cube_body = env.task.objects["cube"]
        cube = np.asarray(p.getBasePositionAndOrientation(
            cube_body, physicsClientId=cid)[0], dtype=np.float64)
        adapter = PyBulletArmIK(
            env.robot_id, cid, robot_type="h1_2", sides=("right",),
            rest_poses="current", ik_mode="pose",
            orientation_scope=orientation_scope,
            lock_others=True, arm_force=200.0, ik_solver="paper")
        for index in range(p.getNumJoints(env.robot_id, physicsClientId=cid)):
            name = p.getJointInfo(
                env.robot_id, index, physicsClientId=cid)[12].decode().lower()
            if name.startswith("r_") and any(
                    key in name for key in ("thumb", "index", "middle", "ring",
                                            "pinky", "hand_base")):
                p.changeDynamics(
                    env.robot_id, index, lateralFriction=float(finger_friction),
                    physicsClientId=cid)

        frame = hand_frame(
            math.radians(yaw_deg), mode, math.radians(tilt_deg))
        orientation = adapter.hand_target_orientation("right", frame, source="base")
        target_world = (cube + np.array([0.0, 0.0, wrist_dz])
                        + np.asarray(target_bias, dtype=np.float64))
        target_base = adapter.to_base(target_world)
        values = adapter.solve(
            "right", target_base, source="base", orientation=orientation,
            orientation_frame="absolute")
        values = adapter.refine_orientation(
            "right", target_base, orientation, source="base",
            orientation_frame="absolute", arm_values=values)
        adapter.apply("right", values)

        ranges, names = read_joint_ranges(env.robot_id, cid)
        mapping = build_mapping("h1_2", "right", ranges)
        open_targets, _ = map_dofs(np.zeros(17), mapping,
                                   robot_type="h1_2", side="right")
        close_targets, _ = map_dofs(
            np.zeros(17), mapping, pinch=(closure, strength, 0.5),
            robot_type="h1_2", side="right")
        for name, value in open_targets.items():
            p.setJointMotorControl2(
                env.robot_id, names[name], p.POSITION_CONTROL,
                targetPosition=float(value), force=200.0, physicsClientId=cid)
        wait(p, cid, 120)
        finger_centers = []
        for index in range(p.getNumJoints(env.robot_id, physicsClientId=cid)):
            name = p.getJointInfo(
                env.robot_id, index, physicsClientId=cid)[12].decode().lower()
            if name in ("r_thumb_distal", "r_index_intermediate"):
                finger_centers.append(np.asarray(p.getLinkState(
                    env.robot_id, index, computeForwardKinematics=True,
                    physicsClientId=cid)[4], dtype=np.float64))
        recenter_m = np.zeros(3)
        if recenter and finger_centers:
            recenter_m = cube - np.mean(np.stack(finger_centers), axis=0)
            target_world = target_world + recenter_m
            values = adapter.solve(
                "right", adapter.to_base(target_world), source="base",
                orientation=orientation, orientation_frame="absolute")
            values = adapter.refine_orientation(
                "right", adapter.to_base(target_world), orientation,
                source="base", orientation_frame="absolute", arm_values=values)
            adapter.apply("right", values)
            for name, value in open_targets.items():
                p.setJointMotorControl2(
                    env.robot_id, names[name], p.POSITION_CONTROL,
                    targetPosition=float(value), force=200.0,
                    physicsClientId=cid)
            wait(p, cid, 120)
        open_link_gaps = {}
        hand_base_index, _hand_base_name = adapter.hand_base["right"]
        hand_base_position = np.asarray(p.getLinkState(
            env.robot_id, hand_base_index, computeForwardKinematics=True,
            physicsClientId=cid)[4], dtype=np.float64)
        hand_link_vectors = {}
        for index in range(p.getNumJoints(env.robot_id, physicsClientId=cid)):
            name = p.getJointInfo(
                env.robot_id, index, physicsClientId=cid)[12].decode()
            low = name.lower()
            if not (low.startswith("r_") and any(
                    key in low for key in ("thumb", "index", "middle", "ring",
                                           "pinky", "hand_base"))):
                continue
            points = p.getClosestPoints(
                env.robot_id, cube_body, 0.25, linkIndexA=index,
                physicsClientId=cid)
            if points:
                open_link_gaps[name] = round(
                    min(abs(float(point[8])) for point in points) * 1000.0, 1)
                link_position = np.asarray(p.getLinkState(
                    env.robot_id, index, computeForwardKinematics=True,
                    physicsClientId=cid)[4], dtype=np.float64)
                hand_link_vectors[name] = [
                    round(float(value), 4)
                    for value in (link_position - hand_base_position)]
        open_gap = min(open_link_gaps.values(), default=float("inf")) / 1000.0

        for name, value in close_targets.items():
            p.setJointMotorControl2(
                env.robot_id, names[name], p.POSITION_CONTROL,
                targetPosition=float(value), force=200.0, physicsClientId=cid)
        grasp_hold = None
        for _ in range(180):
            p.stepSimulation(physicsClientId=cid)
            current_contacts = p.getContactPoints(
                bodyA=env.robot_id, bodyB=cube_body, physicsClientId=cid)
            if grasp_hold is None and len(current_contacts) >= 2:
                measured = {
                    name: float(p.getJointState(
                        env.robot_id, names[name], physicsClientId=cid)[0])
                    for name in close_targets
                }
                grasp_hold = {
                    name: measured[name] + 0.05 * (
                        close_targets[name] - measured[name])
                    for name in close_targets
                }
                for name, value in grasp_hold.items():
                    p.setJointMotorControl2(
                        env.robot_id, names[name], p.POSITION_CONTROL,
                        targetPosition=value, force=200.0, physicsClientId=cid)
        closed_link_gaps = {}
        for index in range(p.getNumJoints(env.robot_id, physicsClientId=cid)):
            name = p.getJointInfo(
                env.robot_id, index, physicsClientId=cid)[12].decode()
            low = name.lower()
            if not (low.startswith("r_") and any(
                    key in low for key in ("thumb", "index", "middle", "ring",
                                           "pinky", "hand_base"))):
                continue
            points = p.getClosestPoints(
                env.robot_id, cube_body, 0.25, linkIndexA=index,
                physicsClientId=cid)
            if points:
                closed_link_gaps[name] = round(
                    min(abs(float(point[8])) for point in points) * 1000.0, 1)
        contacts = p.getContactPoints(
            bodyA=env.robot_id, bodyB=cube_body, physicsClientId=cid)
        contact_links = sorted({
            p.getJointInfo(env.robot_id, int(point[3]),
                           physicsClientId=cid)[12].decode()
            for point in contacts
        })
        contact_forces = [round(float(point[9]), 2) for point in contacts]
        contact_normals = [
            [round(float(value), 3) for value in point[7]] for point in contacts
        ]
        contact_positions = [
            [round(float(value), 4) for value in point[5]] for point in contacts
        ]
        grasp_constraint = -1
        if attach_on_contact and len(contacts) >= 2:
            state = p.getLinkState(
                env.robot_id, hand_base_index, computeForwardKinematics=True,
                physicsClientId=cid)
            obj_pos, obj_orn = p.getBasePositionAndOrientation(
                cube_body, physicsClientId=cid)
            inverse_pos, inverse_orn = p.invertTransform(
                state[4], state[5], physicsClientId=cid)
            rel_pos, rel_orn = p.multiplyTransforms(
                inverse_pos, inverse_orn, obj_pos, obj_orn,
                physicsClientId=cid)
            grasp_constraint = p.createConstraint(
                env.robot_id, hand_base_index, cube_body, -1, p.JOINT_FIXED,
                [0, 0, 0], parentFramePosition=[0, 0, 0],
                childFramePosition=list(rel_pos),
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=list(rel_orn),
                physicsClientId=cid)

        wrist_before = np.asarray(adapter.measure("right")[0], dtype=np.float64)
        lift_residual_mm = 0.0
        for step in range(1, 7):
            lifted = wrist_before + np.array([0.0, 0.0, 0.03 * step])
            values = adapter.solve(
                "right", adapter.to_base(lifted), source="base",
                orientation=orientation, orientation_frame="absolute")
            values = adapter.refine_orientation(
                "right", adapter.to_base(lifted), orientation, source="base",
                orientation_frame="absolute", arm_values=values)
            lift_residual_mm = float(adapter.residual_mm("right"))
            adapter.apply("right", values)
            for name, value in (grasp_hold or close_targets).items():
                p.setJointMotorControl2(
                    env.robot_id, names[name], p.POSITION_CONTROL,
                    targetPosition=float(value), force=200.0,
                    physicsClientId=cid)
            wait(p, cid, 40)
        wrist_after = np.asarray(adapter.measure("right")[0], dtype=np.float64)
        contacts_after = p.getContactPoints(
            bodyA=env.robot_id, bodyB=cube_body, physicsClientId=cid)
        final = np.asarray(p.getBasePositionAndOrientation(
            cube_body, physicsClientId=cid)[0], dtype=np.float64)
        return {
            "yaw_deg": yaw_deg, "mode": mode, "wrist_dz": wrist_dz,
            "tilt_deg": tilt_deg,
            "target_bias": [round(float(value), 4) for value in target_bias],
            "closure": closure, "strength": strength,
            "orientation_scope": orientation_scope,
            "finger_friction": finger_friction,
            "open_gap_mm": round(open_gap * 1000.0, 1),
            "open_link_gaps_mm": open_link_gaps,
            "hand_link_vectors_m": hand_link_vectors,
            "recenter_m": [round(float(value), 4) for value in recenter_m],
            "contacts": len(contacts), "lift_mm": round((final[2] - cube[2]) * 1000.0, 1),
            "grasp_hold": (None if grasp_hold is None else {
                name: round(value, 4) for name, value in grasp_hold.items()}),
            "contact_links": contact_links,
            "contact_forces": contact_forces,
            "contact_normals": contact_normals,
            "contact_positions": contact_positions,
            "attached": grasp_constraint >= 0,
            "closed_link_gaps_mm": closed_link_gaps,
            "lift_residual_mm": round(lift_residual_mm, 1),
            "wrist_lift_mm": round(
                float(np.linalg.norm(wrist_after - wrist_before)) * 1000.0, 1),
            "contacts_after": len(contacts_after),
            "cube_final": [round(float(v), 4) for v in final],
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    for mode in ("down",):
        for yaw in (0.0, 90.0, 180.0, 270.0):
            for dz in (0.10, 0.12, 0.14, 0.16):
                result = try_grasp(yaw, mode, dz, "fist", 1.0)
                print(result, flush=True)


if __name__ == "__main__":
    main()
