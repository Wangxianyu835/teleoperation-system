"""Native dex contracts and boundaries; optional solver checks live in acceptance."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleoperation.robots.native_hand import NativeHandModel
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.contracts.hand_control import HandJointResult, HandJointCommand
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.dex_native import DexCalibration, DexInputAdapter
from teleoperation.retargeting.hand.topology import expand_hand21_to25


def open_points():
    points = np.zeros((21, 3), dtype=np.float32)
    points[1:5] = [[-.025, .025, 0], [-.04, .04, 0], [-.05, .055, 0], [-.06, .07, 0]]
    for start, x, y in ((5, -.025, .065), (9, 0, .075), (13, .02, .07), (17, .04, .055)):
        points[start:start + 4] = [[x, y + i * .02, .001 * i] for i in range(4)]
    return points


def camera_observation(points=None, timestamp=0.):
    points = open_points() if points is None else points
    return RawHandFrame({"left": points, "right": points}, timestamp, "mediapipe", {
        "world_landmark_space": "mediapipe_world_meters", "world_length_unit": "m"},
        world_hands={"left": points, "right": points})


def vr_observation(points=None, timestamp=0.):
    points = expand_hand21_to25(open_points() if points is None else points)
    hands = {}
    for side in ("left", "right"):
        value = np.tile(np.eye(4), (25, 1, 1))
        # Both sensors encode anatomical geometry with proper rotations, not
        # the upstream left-to-right reflection embedded in its model profile.
        value[:, :3, 3] = points
        hands[side] = value
    return RawHandFrame(hands, timestamp, "visionpro", geometry_encoding={s: "transforms25" for s in hands})


class DexInputTests(unittest.TestCase):
    def test_native_result_requires_named_finite_coordinates(self):
        valid = HandJointResult("g1", "left", "a"*64, ("a", "b"), np.zeros(2), 0.)
        self.assertEqual(valid.named_positions(), {"a": 0., "b": 0.})
        with self.assertRaises(ValueError):
            HandJointResult("g1", "left", "a"*64, ("a", "b"), np.zeros(18), 0.)
        with self.assertRaises(ValueError):
            HandJointResult("g1", "left", "a"*64, ("a", "a"), np.zeros(2), 0.)
    def test_camera_and_explicit_vr_profiles_produce_equivalent_geometry(self):
        from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
        camera = MediaPipePalmLocalProcessor('world').process_frame(camera_observation())
        vr = DexInputAdapter("visionpro").adapt(vr_observation())
        for side in ("left", "right"):
            np.testing.assert_allclose(camera.hands[side], vr.hands[side], atol=1e-5)
        self.assertNotIn("dex_input_profile", camera.metadata)

    def test_translation_and_rotation_do_not_change_local_hand_geometry(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for source, make in (("visionpro", vr_observation),):
            adapter = DexInputAdapter(source)
            first = adapter.adapt(make())
            second = adapter.adapt(make(open_points() @ rotation.T + [2, -3, 5]))
            for side in ("left", "right"):
                np.testing.assert_allclose(first.hands[side], second.hands[side], atol=1e-5)

    def test_normalized_wrong_units_and_repeated_alignment_are_rejected(self):
        raw = camera_observation()
        for metadata in ({}, {"world_landmark_space": "mediapipe_world_meters", "world_length_unit": "mm"},
                         {**raw.metadata, "coordinate_alignment": "already-aligned"}):
            with self.assertRaises(ValueError):
                DexInputAdapter("camera").adapt(RawHandFrame(raw.hands, 0., "mediapipe", metadata, world_hands=raw.world_hands))

    def test_invalid_vr_side_does_not_contaminate_other_side(self):
        raw = vr_observation()
        for bad in (np.zeros((25, 4, 4)), np.full((25, 4, 4), np.nan)):
            result = DexInputAdapter("visionpro").adapt(RawHandFrame(
                {"left": bad, "right": raw.hands["right"]}, 0., "visionpro", geometry_encoding=raw.geometry_encoding))
            self.assertIsNone(result.hands["left"])
            self.assertIsNotNone(result.hands["right"])

    def test_calibration_is_fixed_persisted_and_never_overwrites(self):
        adapter = DexInputAdapter("visionpro")
        frames = [adapter.adapt(vr_observation(timestamp=i / 30)) for i in range(61)]
        calibration = DexCalibration.fit(frames, 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            self.assertEqual(DexCalibration.load(path), calibration)
            with self.assertRaises(FileExistsError):
                calibration.save(path)
        with self.assertRaises(ValueError):
            DexCalibration.fit(frames[:5], 0.)


class NativeHandModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def model(self, robot="h1_2", side="right"):
        return NativeHandModel(robot, side, Path(self.directory.name))

    def test_six_models_cover_named_joints_and_fingertips(self):
        for robot, counts in (("h1_2", (12, 6, 5)), ("gr1_t2", (11, 6, 5)), ("g1", (7, 7, 3))):
            for side in ("left", "right"):
                model = self.model(robot, side)
                self.assertEqual((len(model.joint_names), len(model.independent_names), len(model.tip_frames)), counts)
                self.assertEqual(len(set(model.joint_names)), counts[0])
                self.assertEqual(set(model.expand(model.bounds.mean(1))), set(model.joint_names))

    def test_mimic_limits_and_velocities_are_pulled_back_to_drivers(self):
        model = self.model()
        index = model.independent_names.index("R_thumb_proximal_pitch_joint")
        np.testing.assert_allclose(model.bounds[index], [0, .5])
        self.assertAlmostEqual(model.velocity_limits[index], .5 / 2.4)
        expanded = model.expand(model.bounds[:, 1])
        self.assertAlmostEqual(expanded["R_thumb_distal_joint"], 1.2)
        self.assertAlmostEqual(expanded["R_thumb_intermediate_joint"], .8)

    def test_all_coupled_boundary_commands_obey_original_limits(self):
        for robot in ("h1_2", "gr1_t2", "g1"):
            for side in ("left", "right"):
                model = self.model(robot, side)
                for values in model.bounds.T:
                    for name, value in model.expand(values).items():
                        low, high = model.limits[name]
                        self.assertGreaterEqual(value, low - 1e-12)
                        self.assertLessEqual(value, high + 1e-12)

    def test_bad_commands_are_rejected_without_implicit_clamping(self):
        model = self.model()
        for values in (np.zeros(18), np.full(6, np.nan), model.bounds[:, 1] + 1):
            with self.assertRaises(ValueError):
                model.expand(values)

    def test_configuration_is_reproducible(self):
        first, second = self.model(), self.model()
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(first.urdf_path.read_bytes(), second.urdf_path.read_bytes())


class NativeCommandTests(unittest.TestCase):
    setUp = NativeHandModelTests.setUp
    model = NativeHandModelTests.model
    def test_rate_limit_mimic_missing_recovery_and_clock_independence(self):
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        model = self.model()
        key = (model.robot_type, model.side)
        models = {key: model}
        expanded = model.expand(model.bounds[:, 1])
        spec = load_hand_spec(*key)
        result = HandJointResult(*key, spec.spec_hash, spec.independent_joint_names, model.bounds[:, 1], 0.)
        runs = []
        for host_interval in (0., 1/13, 1/30):
            pipeline = JointCommandPipeline({key: spec}, filter_mode="lowpass")
            commands = []
            for index in range(10):
                # Host clock has no role when recording timestamps are passed.
                from unittest.mock import patch
                with patch("teleoperation.simulation.hand_commands.time.monotonic", return_value=index*host_interval):
                    before = pipeline.previous[key].copy()
                    named = pipeline.update(result, timestamp=index/30).named_positions()
                after = pipeline.previous[key]
                self.assertTrue(np.all(np.abs(after-before) <= model.velocity_limits/30 + 1e-10))
                for n,v in model.expand(after).items():
                    self.assertAlmostEqual(v,named[n])
                commands.append(after.copy())
            held = pipeline.missing(key)
            before = pipeline.previous[key].copy()
            self.assertEqual(held.named_positions(), model.expand(before))
            pipeline.update(result, timestamp=100.)
            self.assertTrue(np.all(np.abs(pipeline.previous[key]-before) <= model.velocity_limits/30+1e-10))
            runs.append(commands)
        np.testing.assert_array_equal(runs[0], runs[1]); np.testing.assert_array_equal(runs[0], runs[2])

    def test_native_h5_round_trip_and_config_rejection(self):
        import h5py
        from teleoperation.data.hand_joint_recording import HandJointResultWriter, iter_joint_commands
        model = self.model()
        key = (model.robot_type, model.side)
        named = model.expand(model.bounds.mean(1))
        spec = load_hand_spec(*key)
        result = HandJointResult(*key, spec.spec_hash, spec.independent_joint_names, model.bounds.mean(1), 0.)
        path = Path(self.directory.name) / "native.h5"
        with HandJointResultWriter(path, {key:spec}, {}) as writer:
            command = HandJointCommand(*key, spec.spec_hash, spec.joint_names, np.array([named[n] for n in spec.joint_names]), 0.)
            writer.append(0, 0., {key:result}, {key:command})
            with self.assertRaises(ValueError):
                writer.append(1, 0., {key:result}, {key:command})
        rows = list(iter_joint_commands(path, {key:spec}))
        np.testing.assert_array_equal(rows[0][1][key].positions, command.positions)
        with h5py.File(path,"r+") as stream:
            stream["/".join(key)].attrs["spec_hash"]="bad"
        with self.assertRaises(ValueError):
            list(iter_joint_commands(path,{key:spec}))

    def test_scene_writes_only_selected_named_hand_and_rejects_bad_commands(self):
        from unittest.mock import Mock
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        model = self.model()
        key = (model.robot_type, model.side)
        names = {n:i for i,n in enumerate(model.joint_names)}
        scene = object.__new__(ThreeRobotHandScene)
        scene.p=Mock(); scene.cid=1; scene.sides=("right",); scene.alive=lambda:True
        scene.robots=[{"type":"h1_2","body":42,"plans":{"right":(load_hand_spec(*key),names)}}]
        scene.applied={"right":0}
        named=model.expand(model.bounds.mean(1))
        spec = load_hand_spec(*key)
        result=HandJointCommand(*key, spec.spec_hash, model.joint_names,np.array([named[n] for n in model.joint_names]),0.)
        scene.apply_native(result)
        calls=scene.p.setJointMotorControl2.call_args_list
        self.assertEqual(len(calls),len(model.joint_names))
        self.assertEqual({c.args[1] for c in calls},set(names.values()))
        self.assertEqual(scene.native_commands[key],named)


class DexRealtimeTests(unittest.TestCase):
    def test_latest_source_drops_backlog_and_closes(self):
        import threading
        from unittest.mock import Mock
        from teleoperation.applications.dex_native import LatestCameraInput
        source=Mock()
        produced=threading.Event()
        allow=threading.Event()
        release=threading.Event()
        count=[0]
        def read(**kwargs):
            count[0]+=1
            if count[0]==2:
                allow.wait(2.)
            if count[0]==5:
                produced.set()
                release.wait(2.)
            return count[0]
        source.next_observation.side_effect=read
        source.release.side_effect=lambda **kwargs:release.set()
        source._cap.release.side_effect=lambda:release.set()
        wrapper=LatestCameraInput(source)
        first=wrapper.next_observation(include_world=True)
        allow.set()
        self.assertTrue(produced.wait(2.))
        self.assertEqual(wrapper.next_observation(include_world=True),4)
        self.assertEqual(first,1)
        wrapper.close()
        self.assertFalse(wrapper.thread.is_alive())

    def test_dex_cli_and_factory_reject_wrong_profile_without_loading_devices(self):
        from teleoperation.cli import build_parser
        from teleoperation.applications.hand_control import _selection
        args=build_parser().parse_args(["hand","vr","--backend","dex","--input-h5","unused.h5",
            "--dex-calibration","unused.json","--scene","robots","--hand","both","--hand-mapping","same"])
        self.assertIsNone(args.checkpoint)
        with self.assertRaisesRegex(ValueError,"requires"):
            from types import SimpleNamespace
            _selection(SimpleNamespace(input_profile="normalized"))

    def test_existing_vr_route_requires_checkpoint_before_opening_devices(self):
        from teleoperation.cli import build_parser
        from teleoperation.applications.visionpro_hand import run
        args=build_parser().parse_args(["hand","vr","--ip","unused"])
        with self.assertRaisesRegex(ValueError,"requires --checkpoint"):
            run(args)


class DexVRRecordingTests(unittest.TestCase):
    def test_camera_fixed_calibration_rejected_before_reading_input(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from teleoperation.applications.dex_native import calibrate
        with patch("teleoperation.applications.dex_native.recording_frames") as read:
            with self.assertRaisesRegex(ValueError, "visionpro only"):
                calibrate(SimpleNamespace(source="camera"))
            read.assert_not_called()

    def test_vr_saves_replayable_native_commands_including_missing_frame(self):
        from unittest.mock import Mock, patch
        from teleoperation.cli import build_parser
        from teleoperation.applications.dex_native import visionpro_run
        from teleoperation.data.hand_joint_recording import iter_joint_commands
        from teleoperation.retargeting.hand.dex_native import PROFILES
        specs = {(r,"right"):load_hand_spec(r,"right") for r in ("h1_2","gr1_t2","g1")}
        from types import SimpleNamespace
        retargeter = Mock(specs=specs, models=dict.fromkeys(specs))
        def solve(window):
            valid = window.hands["right"] is not None
            return {key:HandJointResult(*key,spec.spec_hash,spec.independent_joint_names,
                spec.bounds.mean(1) if valid else None,window.timestamp,valid) for key,spec in specs.items()}
        retargeter.retarget.side_effect = solve
        retargeter.targets.side_effect = lambda frame,key: None if frame.hands[key[1]] is None else np.zeros((len(specs[key].tip_frames),3))
        source = Mock()
        source.next_observation.side_effect = [vr_observation(timestamp=1.),
            RawHandFrame({"left":None,"right":None},2.,"visionpro"),StopIteration]
        scene = Mock(robots=[{}, {}, {}])
        scene.alive.return_value = True
        wrists = Mock()
        wrists.summary.return_value = {"applied_frames":{}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"native.h5"
            args = build_parser().parse_args(["hand","vr","--backend","dex","--ip","unused",
                "--dex-calibration",str(Path(directory)/"fixed.json"),"--scene","robots","--headless",
                "--fps","0","--output-h5",str(output)])
            with patch("teleoperation.applications.dex_native.DexCalibration.load",return_value=SimpleNamespace(profile=PROFILES["visionpro"],source_sha256="",segment=(0.,0.))), \
                 patch("teleoperation.applications.dex_native.DexNativeRetargeter",return_value=retargeter), \
                 patch("teleoperation.inputs.visionpro.VisionProInput",return_value=source), \
                 patch("teleoperation.applications.three_robot_scene.ThreeRobotHandScene",return_value=scene), \
                 patch("teleoperation.simulation.wrist_control.vr_arm_controller",return_value=wrists):
                self.assertEqual(visionpro_run(args),0)
            rows = list(iter_joint_commands(output,specs))
            self.assertEqual([row[0] for row in rows],[1.,2.])
            for key in specs:
                np.testing.assert_array_equal(rows[0][1][key].positions,rows[1][1][key].positions)
            import h5py
            with h5py.File(output) as stream:
                self.assertEqual(stream.attrs["schema"],"hand_joint_results_v1")
                for key in specs:
                    group=stream["/".join(key)]
                    np.testing.assert_array_equal(group["valid"][:],[True,False])
                    self.assertTrue(np.isnan(group["raw_positions"][1]).all())
            source.close.assert_called_once()
            scene.close.assert_called_once()
