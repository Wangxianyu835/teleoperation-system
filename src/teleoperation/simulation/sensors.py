import pybullet as p
from collections import defaultdict
from teleoperation.contracts.recording import SimulationSample

class SimulationSensors:
    def collect(self, robot_id: int, task_objects: dict,
                    client_id: int, elapsed: float):
        """记录一步数据"""
        self.episode_data = {"joint_positions": [], "joint_velocities": [], "camera_rgb": [], "head_camera_rgb": [], "timestamps": []}
        self.obj_pose_history = defaultdict(list)
        # 关节状态
        num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
        if num_joints > 0:
            joint_states = p.getJointStates(robot_id, range(num_joints),
                                            physicsClientId=client_id)
            self.episode_data['joint_positions'].append(
                [s[0] for s in joint_states])
            self.episode_data['joint_velocities'].append(
                [s[1] for s in joint_states])

        # 物体6D pose
        for name, obj_id in task_objects.items():
            pos, orn = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=client_id)
            self.obj_pose_history[name].append((pos, orn))

        # 第三人称相机（320x240）
        self._record_camera('camera_rgb', client_id,
                            eye=[1.5, 0, 1.8],
                            target=[0.5, 0, 0.8])

        # 头部第一人称相机
        self._record_camera('head_camera_rgb', client_id,
                            eye=[0, 0, 1.5],
                            target=[0.5, 0, 0.8])

        self.episode_data['timestamps'].append(elapsed)

        cameras = {name: values[0] for name, values in self.episode_data.items() if name in ("camera_rgb", "head_camera_rgb") and values}
        return SimulationSample(elapsed, self.episode_data["joint_positions"][0] if self.episode_data["joint_positions"] else None, self.episode_data["joint_velocities"][0] if self.episode_data["joint_velocities"] else None, {name: poses[0] for name, poses in self.obj_pose_history.items()}, cameras)

    def _record_camera(self, key: str, client_id: int,
                       eye: list, target: list):
        """记录相机图像"""
        width, height = 320, 240
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=client_id,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height,
            nearVal=0.1, farVal=10.0,
            physicsClientId=client_id,
        )
        try:
            img = p.getCameraImage(width, height, view_matrix, proj_matrix,
                                   physicsClientId=client_id)
            self.episode_data[key].append({
                'rgb': img[2],
                'depth': img[3],
                'seg': img[4],
            })
        except Exception:
            pass
