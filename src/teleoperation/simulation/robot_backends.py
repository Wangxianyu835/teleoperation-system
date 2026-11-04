"""Robot-specific motor bindings; the group alone owns physical advancement."""
import numpy as np
from teleoperation.contracts.hand_control import HandJointCommand


class SimulationGroup:
    def __init__(self,scene=None,*,gui=False):
        if scene is None:
            from .robot_part_scene import RobotPartWorld
            scene=RobotPartWorld(gui)
        self.scene=scene
        self.group_id=f'pybullet:{scene.cid}'
        self.time_step=1/240.
        self.closed=False

    def advance(self,steps):
        self.scene.steps_per_frame=steps
        self.scene.step()

    def close(self):
        if not self.closed:
            self.closed=True
            self.scene.close()


class NativeRobotBackend:
    def __init__(self,spec,scene,physics_group,driver=None):
        self.spec,self.scene,self.physics_group=spec,scene,physics_group
        self.robot=next(r for r in scene.robots if r['type']==spec.robot_id)
        self.robot_id=spec.robot_id
        if driver is None and any(k.endswith('_arm') for k in spec.parts):
            from .arm_scene_driver import ArmSceneDriver
            driver=ArmSceneDriver(scene,sides=tuple(k.split('_')[0] for k in spec.parts if k.endswith('_arm')),
                                  robots=(spec.robot_id,),track_posture=False)
        self.driver=driver
        self.names={scene.p.getJointInfo(self.robot['body'],i,physicsClientId=scene.cid)[1].decode():i
                    for i in range(scene.p.getNumJoints(self.robot['body'],physicsClientId=scene.cid))}

    def validate(self,frame):
        self.spec.validate_frame(frame)
        if not self.scene.alive():raise RuntimeError('Simulation connection is closed')

    def write(self,frame):
        for key,part in frame.parts.items():
            side,kind=key.split('_')
            if kind=='hand':
                hand=self.spec.hand_specs[side]
                self.scene.apply_native(HandJointCommand(hand.hand_model,side,hand.spec_hash,
                    part.joint_names,part.positions,frame.timestamp))
            else:
                self.driver.ik[self.spec.robot_id].apply(side,dict(zip(part.joint_names,part.positions)))

    def read_state(self):
        return {key:np.array([self.scene.p.getJointState(self.robot['body'],self.names[n],physicsClientId=self.scene.cid)[0]
                             for n in part.joint_names]) for key,part in self.spec.parts.items()}

    def close(self):pass  # This binding borrows its group's connection.


class Tron2AL21Backend:
    def __init__(self,spec,scene,physics_group,robot,part_bodies,names,world,calibration):
        self.spec,self.scene,self.physics_group=spec,scene,physics_group
        self.robot,self.part_bodies,self.names,self.world=robot,part_bodies,names,world
        self.robot_id=spec.robot_id; self.driver=None; self.arm_controller=None
        if world is not None:
            from teleoperation.applications.dual import Tron2ADualArmController
            self.arm_controller=Tron2ADualArmController(calibration,spec.urdf_path,input_mode='smplerx')

    def validate(self,frame):
        self.spec.validate_frame(frame)
        if not self.scene.alive():raise RuntimeError('Simulation connection closed')

    def write(self,frame):
        p=self.scene.p
        for key,part in frame.parts.items():
            body=self.part_bodies[key]
            for n,q in zip(part.joint_names,part.positions):
                info=p.getJointInfo(body,self.names[body][n],physicsClientId=self.scene.cid)
                p.setJointMotorControl2(body,self.names[body][n],p.POSITION_CONTROL,targetPosition=float(q),
                    force=max(float(info[10]),1.) if key.endswith('_arm') else 200.,physicsClientId=self.scene.cid)

    def read_state(self):
        return {key:np.array([self.scene.p.getJointState(self.part_bodies[key],self.names[self.part_bodies[key]][n],
                    physicsClientId=self.scene.cid)[0] for n in part.joint_names]) for key,part in self.spec.parts.items()}

    def close(self):pass
