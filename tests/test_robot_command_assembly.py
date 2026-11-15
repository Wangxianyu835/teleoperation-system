import unittest
from dataclasses import replace
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.contracts.robot_control import RobotPartCommand
from teleoperation.applications.robot_commands import RobotCommandAssembler


class RobotCommandAssemblyTests(unittest.TestCase):
    def transaction_fixture(self):
        from teleoperation.robots.control_spec import RobotControlSpec, PartSpec
        spec=RobotControlSpec('g1','g1',None,{'left_arm':PartSpec(('joint',),((-2.,2.),),(0.,))},{},'a'*64,'arms','left')
        return RobotCommandAssembler({'g1':spec},{'g1':spec.reference_frame(0,0.)})

    def test_prepare_is_speculative_until_commit(self):
        assembler=self.transaction_fixture()
        pending=assembler.prepare('g1',1,.1,{})
        self.assertEqual(assembler.previous['g1'].frame_id,0)
        self.assertIsNone(assembler.times['g1'])
        retry=assembler.prepare('g1',1,.1,{})
        assembler.commit(retry)
        self.assertEqual(assembler.previous['g1'].frame_id,1)
        with self.assertRaises(ValueError):assembler.commit(pending)

    def test_valid_repeated_target_tracks_until_expiry_and_rejects_reorder(self):
        assembler=self.transaction_fixture()
        update=RobotPartCommand(('joint',),[.5],100.,'body_s',True,'tracking',7,.3,True)
        assembler.assemble('g1',1,.1,{'left_arm':update})
        repeated=replace(update,positions=[.6],source_is_new=False)
        frame=assembler.assemble('g1',2,.2,{'left_arm':repeated})
        self.assertEqual(frame.parts['left_arm'].status,'tracking')
        self.assertAlmostEqual(frame.parts['left_arm'].positions[0],.6)
        for index,change in enumerate(({'source_timestamp':99.}, {'target_sequence':6},
                                       {'target_sequence':8}, {'source_is_new':True}),3):
            frame=assembler.assemble('g1',index,.2+index*.001,{'left_arm':replace(repeated,**change)})
            self.assertEqual(frame.parts['left_arm'].status,'holding')
            self.assertAlmostEqual(frame.parts['left_arm'].positions[0],.6)
        expired=assembler.assemble('g1',8,.31,{'left_arm':repeated})
        self.assertEqual(expired.parts['left_arm'].status,'holding')

    def test_new_sequence_requires_new_source_time_and_legacy_duplicates_hold(self):
        assembler=self.transaction_fixture()
        legacy=RobotPartCommand(('joint',),[.5],100.,'body_s',True,'tracking')
        assembler.assemble('g1',1,.1,{'left_arm':legacy})
        self.assertEqual(assembler.assemble('g1',2,.2,{'left_arm':legacy}).parts['left_arm'].status,'holding')
        target=replace(legacy,source_timestamp=101.,target_sequence=7,target_expires_time=.8)
        assembler.assemble('g1',3,.3,{'left_arm':target})
        invalid=replace(target,source_timestamp=102.,target_sequence=6)
        self.assertEqual(assembler.assemble('g1',4,.4,{'left_arm':invalid}).parts['left_arm'].source_timestamp,101.)
        newer=replace(target,source_timestamp=102.,target_sequence=8)
        self.assertEqual(assembler.assemble('g1',5,.5,{'left_arm':newer}).parts['left_arm'].target_sequence,8)

    def test_target_metadata_validation(self):
        base=RobotPartCommand(('joint',),[0.],100.,'body_s',True,'tracking')
        for changes in ({'target_sequence':-1},{'target_sequence':True},{'target_sequence':1.5},
                        {'target_expires_time':float('nan')},{'source_is_new':'false'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(base,**changes)

    def test_repeated_target_cannot_extend_validity_or_change_source_identity(self):
        assembler=self.transaction_fixture()
        target=RobotPartCommand(('joint',),[.5],100.,'body_s',True,'tracking',7,.3,True)
        assembler.assemble('g1',1,.1,{'left_arm':target})
        repeated=replace(target,source_is_new=False)
        boundary=assembler.assemble('g1',2,.3,{'left_arm':repeated})
        self.assertTrue(boundary.parts['left_arm'].observed)
        for index,change in enumerate(({'target_expires_time':.8},{'source_timebase':'another_clock'},
                                       {'source_timestamp':101.}),3):
            frame=assembler.assemble('g1',index,.3+index*.001,{'left_arm':replace(repeated,**change)})
            self.assertEqual(frame.parts['left_arm'].status,'holding')
            self.assertEqual(frame.parts['left_arm'].source_timestamp,100.)

    def test_metadata_cannot_be_dropped_to_bypass_sequence_order(self):
        assembler=self.transaction_fixture()
        target=RobotPartCommand(('joint',),[.5],100.,'body_s',True,'tracking',7,.5,True)
        assembler.assemble('g1',1,.1,{'left_arm':target})
        missing=replace(target,source_timestamp=101.,target_sequence=None)
        frame=assembler.assemble('g1',2,.2,{'left_arm':missing})
        self.assertEqual(frame.parts['left_arm'].status,'holding')
        self.assertEqual(frame.parts['left_arm'].target_sequence,7)

    def test_independent_part_holding_preserves_origin_time(self):
        spec=load_robot_control_spec('g1')
        initial=spec.reference_frame(0,0.)
        assembler=RobotCommandAssembler({'g1':spec},{'g1':initial})
        first=assembler.assemble('g1',0,0.,{})
        q=first.parts['left_arm'].positions.copy(); q[0]=.1
        update=RobotPartCommand(first.parts['left_arm'].joint_names,q,101.,'camera_monotonic_s',True,'tracking')
        second=assembler.assemble('g1',1,.033,{'left_arm':update})
        self.assertAlmostEqual(second.parts['left_arm'].positions[0],.1)
        np.testing.assert_array_equal(second.parts['left_hand'].positions,first.parts['left_hand'].positions)
        self.assertFalse(second.parts['left_hand'].observed)
        third=assembler.assemble('g1',2,.066,{})
        self.assertEqual(third.parts['left_arm'].source_timestamp,101.)
        self.assertEqual(third.parts['left_arm'].status,'holding')
        self.assertFalse(third.parts['left_arm'].observed)

    def test_bad_update_does_not_advance_state_and_targets_are_isolated(self):
        specs={n:load_robot_control_spec(n,resource_overrides={'parts':'hands','side':'right'}) for n in ('g1','h1_2')}
        assembler=RobotCommandAssembler(specs,{n:s.reference_frame(0,0.) for n,s in specs.items()})
        a=assembler.assemble('g1',0,0.,{})
        wrong=specs['h1_2'].reference_frame(0,0.).parts['right_hand']
        with self.assertRaises(ValueError): assembler.assemble('g1',1,.1,{'right_hand':wrong})
        b=assembler.assemble('g1',1,.1,{})
        np.testing.assert_array_equal(a.parts['right_hand'].positions,b.parts['right_hand'].positions)
        with self.assertRaises(ValueError): assembler.assemble('g1',1,.1,{})
        other=assembler.assemble('h1_2',0,0.,{})
        self.assertEqual(other.frame_id,0)
