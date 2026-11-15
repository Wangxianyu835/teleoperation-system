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

    def prepare(self,robot_id,frame_id,timestamp,updates):
        """Build and validate a candidate without accepting any control state."""
        spec=self.specs[robot_id]
        if set(updates)-set(spec.parts): raise ValueError('Update contains disabled or unknown parts')
        if self.times[robot_id] is not None and (timestamp<=self.times[robot_id] or frame_id<=self.frames[robot_id]):
            raise ValueError('Control time and frame_id must strictly increase')
        parts={}
        for key,previous in self.previous[robot_id].parts.items():
            update=updates.get(key)
            if update is None:
                parts[key]=replace(previous,observed=False,source_is_new=False,
                                   status='initialized' if previous.status=='initialized' else 'holding')
            else:
                if update.observed and not self._valid_target(update,previous,timestamp):
                    parts[key]=replace(previous,observed=False,status='holding',source_is_new=False)
                else: parts[key]=update
        frame=RobotCommandFrame(robot_id,spec.spec_hash,frame_id,timestamp,parts)
        spec.validate_frame(frame)
        return frame

    @staticmethod
    def _valid_target(update,previous,timestamp):
        if update.target_expires_time is not None and update.target_expires_time<timestamp:
            return False
        same_clock=update.source_timebase==previous.source_timebase
        if same_clock and previous.source_timestamp is not None:
            if update.source_timestamp<previous.source_timestamp:return False
            if update.source_timestamp==previous.source_timestamp:
                return (not update.source_is_new and update.target_sequence is not None and
                        update.target_sequence==previous.target_sequence and
                        update.target_expires_time is not None and
                        update.target_expires_time==previous.target_expires_time)
        if not update.source_is_new:return False
        if previous.target_sequence is not None:
            if update.target_sequence is None or update.target_sequence<=previous.target_sequence:return False
        return True

    def commit(self,frame):
        """Accept a candidate only after the caller confirms successful output."""
        robot_id=frame.robot_id
        self.specs[robot_id].validate_frame(frame)
        if self.times[robot_id] is not None and (frame.timestamp<=self.times[robot_id] or frame.frame_id<=self.frames[robot_id]):
            raise ValueError('Control time and frame_id must strictly increase')
        self.previous[robot_id]=frame
        self.times[robot_id],self.frames[robot_id]=frame.timestamp,frame.frame_id

    def assemble(self,robot_id,frame_id,timestamp,updates):
        """Legacy eager acceptance; transactional callers use prepare and commit."""
        frame=self.prepare(robot_id,frame_id,timestamp,updates)
        self.commit(frame)
        return frame
