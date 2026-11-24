"""Robot-specific motor bindings; the group alone owns physical advancement."""
import numpy as np
from teleoperation.contracts.hand_control import HandJointCommand


def write_joint_positions(p, client, body, names, joint_names, positions, *, force):
    for name, value in zip(joint_names, positions):
        p.setJointMotorControl2(body, names[name], p.POSITION_CONTROL,
                                targetPosition=float(value), force=force, physicsClientId=client)


class SimulationGroup:
    def __init__(self,scene=None,*,gui=False,owns_scene=True,advance_frame=None):
        if scene is None:
            from .robot_part_scene import RobotPartWorld
            scene=RobotPartWorld(gui)
        self.scene=scene
        self.group_id=f'pybullet:{scene.cid}'
        self.time_step=1/240.
        self.closed=False
        self.owns_scene=owns_scene
        self.advance_frame=advance_frame

    def alive(self):
        return self.scene.alive()

    @property
    def frames(self):
        return self.scene.frames

    def advance(self,steps):
        self.scene.steps_per_frame=steps
        (self.advance_frame or self.scene.step)()

    def close(self):
        if not self.closed:
            self.closed=True
            if self.owns_scene:self.scene.close()


class NativeRobotBackend:
    def __init__(self,spec,scene,physics_group,driver=None,*,create_driver=True):
        self.spec,self.scene,self.physics_group=spec,scene,physics_group
        self.robot=next(r for r in scene.robots if r['type']==spec.robot_id)
        self.bullet_binding = (scene.p, scene.cid, self.robot['body'])
        self.robot_id=spec.robot_id
        if create_driver and driver is None and any(k.endswith('_arm') for k in spec.parts):
            from .arm_scene_driver import ArmSceneDriver
            driver=ArmSceneDriver(scene,sides=tuple(k.split('_')[0] for k in spec.parts if k.endswith('_arm')),
                                  robots=(spec.robot_id,),track_posture=False)
        self.driver=driver
        self.names={scene.p.getJointInfo(self.robot['body'],i,physicsClientId=scene.cid)[1].decode():i
                    for i in range(scene.p.getNumJoints(self.robot['body'],physicsClientId=scene.cid))}
        if not create_driver:
            active={self.names[n] for key,part in spec.parts.items() if key.endswith('_arm') for n in part.joint_names}
            kept=[(i,q) for i,q in zip(self.robot['hold_ids'],self.robot['hold_targets']) if i not in active]
            self.robot['hold_ids']=[i for i,q in kept];self.robot['hold_targets']=[q for i,q in kept]

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
                if self.driver is None:
                    from .arm_ik import ARM_FORCE
                    write_joint_positions(self.scene.p, self.scene.cid, self.robot['body'],
                                          self.names, part.joint_names, part.positions, force=ARM_FORCE)
                    continue
                self.driver.ik[self.spec.robot_id].apply(side,dict(zip(part.joint_names,part.positions)))
                diagnostics=getattr(self.driver,'diagnostics',None)
                if diagnostics is not None:
                    diagnostics.command(self.spec.robot_id,side,dict(zip(part.joint_names,part.positions)))
                if self.spec.robot_id==next(iter(self.driver.ik)):
                    self.driver.applied[side]+=1

    def read_state(self):
        return {key:np.array([self.scene.p.getJointState(self.robot['body'],self.names[n],physicsClientId=self.scene.cid)[0]
                             for n in part.joint_names]) for key,part in self.spec.parts.items()}

    def close(self):pass  # This binding borrows its group's connection.


class Tron2AL21Backend:
    def __init__(self,spec,scene,physics_group,robot,part_bodies,names,calibration):
        self.spec,self.scene,self.physics_group=spec,scene,physics_group
        self.robot,self.part_bodies,self.names=robot,part_bodies,names
        self.bullet_binding = (scene.p, scene.cid, robot['body'])
        self.robot_id=spec.robot_id; self.driver=None
        self.execution_config=({name:{side:np.asarray(value,dtype=float).tolist() for side,value in getattr(calibration,name).items()}
                                for name in ('safe_q','hand_mount_pose')} if calibration is not None else {})
        self.execution_config['base_pose']=list(robot['base_pose'])
        # The camera controller produces commands; this binding only executes them.

    def validate(self,frame):
        self.spec.validate_frame(frame)
        if not self.scene.alive():raise RuntimeError('Simulation connection closed')

    def write(self,frame):
        p=self.scene.p
        for key,part in frame.parts.items():
            body=self.part_bodies[key]
            for n,q in zip(part.joint_names,part.positions):
                info=p.getJointInfo(body,self.names[body][n],physicsClientId=self.scene.cid)
                options=({'force':max(float(info[10]),1.),'maxVelocity':max(float(info[11]),.01),
                          'positionGain':.3,'velocityGain':1.} if key.endswith('_arm') else {'force':200.})
                p.setJointMotorControl2(body,self.names[body][n],p.POSITION_CONTROL,targetPosition=float(q),
                    physicsClientId=self.scene.cid,**options)

    def read_state(self):
        return {key:np.array([self.scene.p.getJointState(self.part_bodies[key],self.names[self.part_bodies[key]][n],
                    physicsClientId=self.scene.cid)[0] for n in part.joint_names]) for key,part in self.spec.parts.items()}

    def read_arm_state(self):
        from teleoperation.robots.specification import TRON2A_ARM_JOINTS
        p=self.scene.p; result={}
        inverse=p.invertTransform(self.robot['base_pose'],[0,0,0,1])
        for side,index in self.robot.get('ee_indices',{}).items():
            body=self.robot['body']
            states=p.getJointStates(body,[self.names[body][n] for n in TRON2A_ARM_JOINTS[side]],physicsClientId=self.scene.cid)
            ee=p.getLinkState(body,index,computeForwardKinematics=True,physicsClientId=self.scene.cid)
            pose=p.multiplyTransforms(*inverse,ee[4],ee[5])
            result[side]={'q':np.array([v[0] for v in states]),
                          'velocity':np.array([v[1] for v in states]),'pose':np.array((*pose[0],*pose[1]))}
        return result

    def close(self):pass
