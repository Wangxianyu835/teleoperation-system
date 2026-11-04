"""Assemble accepted parts without repeating retargeting, IK or filtering."""
from dataclasses import replace
from teleoperation.contracts.robot_control import RobotCommandFrame


class RobotCommandAssembler:
    def __init__(self,specs,initial):
        self.specs=dict(specs)
        if set(self.specs)!=set(initial): raise ValueError('Every target requires an explicit initial command')
        self.previous=dict(initial)
        self.times={key:None for key in specs}
        self.frames={key:None for key in specs}
        for key,spec in self.specs.items(): spec.validate_frame(initial[key])

    def assemble(self,robot_id,frame_id,timestamp,updates):
        spec=self.specs[robot_id]
        if set(updates)-set(spec.parts): raise ValueError('Update contains disabled or unknown parts')
        if self.times[robot_id] is not None and (timestamp<=self.times[robot_id] or frame_id<=self.frames[robot_id]):
            raise ValueError('Control time and frame_id must strictly increase')
        parts={}
        for key,previous in self.previous[robot_id].parts.items():
            update=updates.get(key)
            if update is None:
                parts[key]=replace(previous,observed=False,status='initialized' if previous.status=='initialized' else 'holding')
            else:
                if (update.observed and previous.source_timestamp is not None and
                    update.source_timebase==previous.source_timebase and update.source_timestamp<=previous.source_timestamp):
                    parts[key]=replace(previous,observed=False,status='holding')
                else: parts[key]=update
        frame=RobotCommandFrame(robot_id,spec.spec_hash,frame_id,timestamp,parts)
        spec.validate_frame(frame)
        self.previous[robot_id]=frame
        self.times[robot_id],self.frames[robot_id]=timestamp,frame_id
        return frame
