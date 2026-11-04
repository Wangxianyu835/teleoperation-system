"""Selected original robot geometry with invisible kinematic support only."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from teleoperation.paths import ROBOTS_ROOT
from teleoperation.robots.specification import ROBOT_SPECS,ARM_NEUTRAL
from teleoperation.contracts.hand_control import HandJointCommand


def visible_part_links(root,spec):
    if spec.display=='both':return {n.get('name') for n in root.findall('link')}
    joints={j.get('name'):j for j in root.findall('joint')}
    visible=set()
    for key,part in spec.parts.items():
        visible.update(joints[n].find('child').get('link') for n in part.joint_names)
        if key.endswith('_hand'):visible.add(spec.hand_specs[key.split('_')[0]].root_link)
    while True:
        added={j.find('child').get('link') for j in joints.values()
               if j.get('type')=='fixed' and j.find('parent').get('link') in visible
               and not (spec.display=='arms' and any(token in j.find('child').get('link').lower()
                                                    for token in ('palm','hand_base')))}
        if added<=visible:break
        visible.update(added)
    return visible


def prepare_visible_urdf(source,destination,visible):
    source=Path(source).resolve()
    root=ET.parse(source).getroot()
    for link in root.findall('link'):
        if link.get('name') not in visible:
            for node in list(link.findall('visual'))+list(link.findall('collision')):link.remove(node)
    for mesh in root.findall('.//mesh'):
        raw=mesh.get('filename')
        if raw.startswith('package://'):
            suffix=raw.split('/meshes/',1)
            base=source.parent.parent if source.parent.name.lower()=='urdf' else source.parent
            path=base/'meshes'/suffix[1] if len(suffix)==2 else source.parent/raw[10:]
        else:
            path=Path(raw)
            if not path.is_absolute():path=source.parent/path
        if not path.is_file():raise FileNotFoundError(f'Visible robot mesh missing: {path}')
        mesh.set('filename',path.resolve().as_posix())
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)
    return Path(destination)


class RobotPartWorld:
    def __init__(self,gui=False):
        import pybullet as p
        self.p=p; self.gui=gui; self.cid=p.connect(p.GUI if gui else p.DIRECT)
        if self.cid<0:raise RuntimeError('Cannot create simulation connection')
        self._temporary=tempfile.TemporaryDirectory(prefix='robot_parts_')
        self.robots=[]; self.visible_links={}; self.visible_indices={}; self.frames=0
        self.steps_per_frame=8; self.closed=False
        self.borrowed_resources=[]
        self.camera_target=[0.,0.,.5]; self.camera_distance=1.
        p.setGravity(0,0,-9.81,physicsClientId=self.cid)
        p.setTimeStep(1/240,physicsClientId=self.cid)
        from .urdf_loader import create_ground
        create_ground(p,physicsClientId=self.cid)

    def load_native(self,spec,base_pose=None):
        source=spec.urdf_path or ROBOTS_ROOT/ROBOT_SPECS[spec.robot_id]['urdf']
        root=ET.parse(source).getroot(); visible=visible_part_links(root,spec)
        destination=Path(self._temporary.name)/(spec.robot_id+'.urdf')
        prepare_visible_urdf(source,destination,visible)
        from .urdf_loader import load_urdf
        pos=base_pose or (0.,0.,ROBOT_SPECS[spec.robot_id]['base_z'])
        body=load_urdf(destination,basePosition=pos,useFixedBase=True,physicsClientId=self.cid)
        self._register(spec,body,visible)
        self.frame_scene()
        return body

    def _register(self,spec,body,visible):
        p=self.p; names={}; links={p.getBodyInfo(body,physicsClientId=self.cid)[0].decode():-1}
        movable={}
        for i in range(p.getNumJoints(body,physicsClientId=self.cid)):
            info=p.getJointInfo(body,i,physicsClientId=self.cid)
            name=info[1].decode(); names[name]=i; links[info[12].decode()]=i
            if info[2]!=p.JOINT_FIXED:
                low,high=info[8:10]
                movable[i]=float(np.clip(ARM_NEUTRAL.get(name,0.),low,high)) if low<=high else 0.
        for key,part in spec.parts.items():
            for name,value in zip(part.joint_names,part.reference):
                if key.endswith('_hand'):movable[names[name]]=value
        for i,value in movable.items():p.resetJointState(body,i,value,physicsClientId=self.cid)
        hand_ids={names[n] for key,part in spec.parts.items() if key.endswith('_hand') for n in part.joint_names}
        held=[i for i in movable if i not in hand_ids]
        plans={side:(hand,names) for side,hand in spec.hand_specs.items()}
        self.robots.append({'type':spec.robot_id,'body':body,'plans':plans,
                            'hold_ids':held,'hold_targets':[movable[i] for i in held]})
        self.visible_links[body]=set(visible)
        self.visible_indices[body]={links[n] for n in visible if n in links}

    def alive(self):return not self.closed and bool(self.p.isConnected(self.cid))

    def apply_native(self,command):
        if not isinstance(command,HandJointCommand):raise ValueError('Expected accepted hand command')
        robot=next(r for r in self.robots if r['type']==command.hand_model)
        hand,names=robot['plans'][command.side]
        if command.spec_hash!=hand.spec_hash or command.joint_names!=hand.joint_names:raise ValueError('Native hand identity mismatch')
        values=command.named_positions()
        expected=hand.expand([values[n] for n in hand.independent_joint_names])
        if any(abs(values[n]-expected[n])>1e-6 for n in values):raise ValueError('Invalid hand coupling')
        for n,value in values.items():
            self.p.setJointMotorControl2(robot['body'],names[n],self.p.POSITION_CONTROL,
                targetPosition=value,force=200.,physicsClientId=self.cid)

    def step(self):
        for robot in self.robots:
            for index,value in zip(robot['hold_ids'],robot['hold_targets']):
                self.p.setJointMotorControl2(robot['body'],index,self.p.POSITION_CONTROL,
                    targetPosition=value,force=500.,physicsClientId=self.cid)
        for _ in range(self.steps_per_frame):self.p.stepSimulation(physicsClientId=self.cid)
        self.frames+=1

    def frame_scene(self):
        boxes=[self.p.getAABB(body,index,physicsClientId=self.cid)
               for body,indices in self.visible_indices.items() for index in indices]
        if boxes:
            low=np.min([b[0] for b in boxes],axis=0); high=np.max([b[1] for b in boxes],axis=0)
            self.camera_target=((low+high)/2).tolist()
            self.camera_distance=max(.18,float(np.linalg.norm(high-low))*1.3)
        if self.gui:
            self.p.resetDebugVisualizerCamera(self.camera_distance,45,-20,self.camera_target,physicsClientId=self.cid)

    def render(self,width=320,height=320):
        self.frame_scene()
        view=self.p.computeViewMatrixFromYawPitchRoll(self.camera_target,self.camera_distance,45,-20,0,2)
        projection=self.p.computeProjectionMatrixFOV(55,width/height,.01,30.)
        image=self.p.getCameraImage(width,height,view,projection,renderer=self.p.ER_TINY_RENDERER,physicsClientId=self.cid)
        return np.asarray(image[2],dtype=np.uint8).reshape(height,width,4)

    def close(self):
        if self.closed:return
        self.closed=True
        for resource in self.borrowed_resources:resource.close()
        if self.p.isConnected(self.cid):self.p.disconnect(self.cid)
        self._temporary.cleanup()


class RobotPartSceneBuilder:
    def build(self,spec,*,group,parts=None,side=None,base_pose=None,calibration=None,geometry='full',
              execution_only=False,execution_config=None):
        if parts is not None and parts!=spec.display or side is not None and side!=spec.side:
            raise ValueError('Scene selection must match the control specification')
        if spec.robot_id=='tron2a_l21':
            if execution_only and any(key.endswith('_arm') for key in spec.parts):
                if execution_config is None:raise ValueError('Recorded TRON2A execution configuration is required')
                from types import SimpleNamespace
                calibration=SimpleNamespace(safe_q=execution_config['safe_q'],hand_mount_pose=execution_config['hand_mount_pose'])
            return self._tron(spec,group,base_pose,calibration,geometry,execution_only)
        group.scene.load_native(spec,base_pose)
        from .robot_backends import NativeRobotBackend
        return NativeRobotBackend(spec,group.scene,group,create_driver=not execution_only)

    def _tron(self,spec,group,base_pose,calibration,geometry,execution_only=False):
        from .robot_backends import Tron2AL21Backend
        from .urdf_loader import load_urdf
        scene=group.scene; p=scene.p; part_bodies={}; names={}; main=None
        has_arms=any(k.endswith('_arm') for k in spec.parts)
        if has_arms:
            if calibration is None:raise ValueError('TRON2A arm scene requires explicit calibration')
            from .tron2a_world import Tron2ADualArmWorld
            root=ET.parse(spec.urdf_path).getroot()
            visible=visible_part_links(root,spec)
            world=Tron2ADualArmWorld(spec.urdf_path,client_id=scene.cid,owns_client=False,
                base_pose=base_pose or (0,0,0),gui=False,display=geometry,visible_links=visible,
                attach_hands=bool(spec.hand_specs),safe_q=calibration.safe_q,hand_mount_pose=calibration.hand_mount_pose)
            scene.borrowed_resources.append(world); main=world.robot
            names[main]=world.joint_indices
            for side in ('left','right'):
                if side+'_arm' in spec.parts:part_bodies[side+'_arm']=main
            for side,body in world.hand_bodies.items():
                part_bodies[side+'_hand']=body
                names[body]={p.getJointInfo(body,i,physicsClientId=scene.cid)[1].decode():i
                             for i in range(p.getNumJoints(body,physicsClientId=scene.cid))}
            scene.visible_links[main]=visible
            linkmap={info[12].decode():i for i in range(p.getNumJoints(main,physicsClientId=scene.cid))
                     for info in (p.getJointInfo(main,i,physicsClientId=scene.cid),)}
            linkmap[p.getBodyInfo(main,physicsClientId=scene.cid)[0].decode()]=-1
            scene.visible_indices[main]={linkmap[n] for n in visible if n in linkmap}
        else:
            world=None
            for side,hand in spec.hand_specs.items():
                root=ET.parse(hand.urdf_path).getroot(); visible={n.get('name') for n in root.findall('link')}
                destination=Path(scene._temporary.name)/('l21_'+side+'.urdf')
                prepare_visible_urdf(hand.urdf_path,destination,visible)
                offset=np.asarray(base_pose or (0,0,.3),dtype=float)+np.array((-.13 if side=='left' else .13,0,0))
                body=load_urdf(destination,basePosition=offset.tolist(),useFixedBase=True,physicsClientId=scene.cid)
                part_bodies[side+'_hand']=body
                names[body]={p.getJointInfo(body,i,physicsClientId=scene.cid)[1].decode():i for i in range(p.getNumJoints(body,physicsClientId=scene.cid))}
                scene.visible_links[body]=visible
        for key,body in part_bodies.items():
            if key.endswith('_hand'):
                visible={p.getJointInfo(body,i,physicsClientId=scene.cid)[12].decode() for i in range(p.getNumJoints(body,physicsClientId=scene.cid))}
                visible.add(p.getBodyInfo(body,physicsClientId=scene.cid)[0].decode())
                scene.visible_links[body]=visible
                scene.visible_indices[body]={item[1] for item in p.getVisualShapeData(body,physicsClientId=scene.cid)}
                for n,q in zip(spec.parts[key].joint_names,spec.parts[key].reference):p.resetJointState(body,names[body][n],q,physicsClientId=scene.cid)
        record={'type':spec.robot_id,'body':main if main is not None else next(iter(part_bodies.values())),
                'plans':{},'hold_ids':[],'hold_targets':[]}
        if main is not None:
            active={names[main][n] for k,part in spec.parts.items() if k.endswith('_arm') for n in part.joint_names}
            record['hold_ids']=[i for i in range(p.getNumJoints(main,physicsClientId=scene.cid))
                                if p.getJointInfo(main,i,physicsClientId=scene.cid)[2]!=p.JOINT_FIXED and i not in active]
            record['hold_targets']=[p.getJointState(main,i,physicsClientId=scene.cid)[0] for i in record['hold_ids']]
        scene.robots.append(record); scene.frame_scene()
        return Tron2AL21Backend(spec,scene,group,record,part_bodies,names,world,calibration,create_controller=not execution_only)
