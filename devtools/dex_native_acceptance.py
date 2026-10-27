"""Reproducible dex-native acceptance; reports live outside tracked source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from teleoperation.paths import PROJECT_ROOT, ROBOTS_ROOT
from teleoperation.robots.specification import ROBOT_SPECS, _is_hand_joint


def inventory():
    import pybullet as p
    from teleoperation.simulation.urdf_loader import load_urdf

    report = {"python": sys.version, "packages": {}, "robots": {}, "assets": {}}
    for name in ("dex-retargeting", "nlopt", "numpy", "pybullet", "mediapipe", "torch"):
        report["packages"][name] = importlib.metadata.version(name)
    for module in ("dex_retargeting", "pinocchio", "nlopt", "mediapipe", "torch"):
        __import__(module)
    import pinocchio
    report["packages"]["pinocchio"] = pinocchio.__version__
    assert tuple(map(int, pinocchio.__version__.split(".")[:3])) >= (3, 3, 1)
    provenance = PROJECT_ROOT / "outputs/tmp_dex_native/wheels/dex_retargeting-0.5.0+winconda-py3-none-any.provenance.json"
    if provenance.exists():
        reference = json.loads(provenance.read_text(encoding="utf-8"))
        installed = importlib.metadata.distribution("dex-retargeting")
        for name, digest in reference["code_sha256"].items():
            assert hashlib.sha256(Path(installed.locate_file(name)).read_bytes()).hexdigest() == digest
        report["dex_provenance"] = reference
    client = p.connect(p.DIRECT)
    try:
        for robot, spec in ROBOT_SPECS.items():
            path = Path(ROBOTS_ROOT) / spec["urdf"]
            root = ET.parse(path).getroot()
            body = load_urdf(str(path), useFixedBase=True, physicsClientId=client)
            loaded = {p.getJointInfo(body, i, physicsClientId=client)[1].decode()
                      for i in range(p.getNumJoints(body, physicsClientId=client))}
            report["robots"][robot] = {}
            for side in ("left", "right"):
                joints = [j for j in root.findall("joint")
                          if j.get("type") != "fixed" and _is_hand_joint(j.get("name"), side)]
                rows = [{"name": j.get("name"), "limit": j.find("limit").attrib,
                         "mimic": None if j.find("mimic") is None else j.find("mimic").attrib}
                        for j in joints]
                expected = {"h1_2": (12, 6), "gr1_t2": (11, 6), "g1": (7, 7)}[robot]
                assert (len(rows), sum(r["mimic"] is None for r in rows)) == expected
                assert all(r["name"] in loaded for r in rows)
                report["robots"][robot][side] = rows
            p.removeBody(body, physicsClientId=client)
        assets = [Path(ROBOTS_ROOT) / spec["urdf"] for spec in ROBOT_SPECS.values()]
        assets += list((Path(ROBOTS_ROOT) / "inspire_hand").glob("inspire_hand_*.urdf"))
        assets += list((Path(ROBOTS_ROOT) / "unitree_hand").glob("unitree_dex3_*.urdf"))
        for path in assets:
            report["assets"][path.relative_to(PROJECT_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        p.disconnect(client)
    return report


def kinematics():
    import numpy as np
    import pybullet as p
    from teleoperation.robots.native_hand import NativeHandModel
    from teleoperation.simulation.urdf_loader import load_urdf
    report = {"stage": 2, "hands": {}, "passed": True}
    client = p.connect(p.DIRECT)
    try:
        for robot, spec in ROBOT_SPECS.items():
            body = load_urdf(str(Path(ROBOTS_ROOT) / spec["urdf"]), useFixedBase=True, physicsClientId=client)
            movable = [i for i in range(p.getNumJoints(body, physicsClientId=client))
                       if p.getJointInfo(body, i, physicsClientId=client)[2] != p.JOINT_FIXED]
            for side in ("left", "right"):
                model = NativeHandModel(robot, side)
                rng = np.random.default_rng(0)
                poses = rng.uniform(model.bounds[:, 0], model.bounds[:, 1], (98, len(model.independent_names)))
                poses = np.vstack((poses, model.bounds.T))
                maximum = 0.
                for values in poses:
                    before = np.array(p.getJointStates(body, movable, physicsClientId=client), dtype=object)
                    expected = model.positions(values)
                    actual = model.bullet_positions(p, body, client, values)
                    maximum = max(maximum, float(np.max(np.linalg.norm(expected - actual, axis=1))))
                    after = np.array(p.getJointStates(body, movable, physicsClientId=client), dtype=object)
                    assert np.array_equal(before[:, :2], after[:, :2]), "FK query changed physical state"
                passed = maximum <= 0.0001
                report["hands"][f"{robot}/{side}"] = {"poses": len(poses), "max_position_error_m": maximum,
                    "config_hash": model.config_hash, "independent": list(model.independent_names),
                    "effective_limits": model.bounds.tolist(), "effective_velocities": model.velocity_limits.tolist(),
                    "passed": passed}
                report["passed"] &= passed
            p.removeBody(body, physicsClientId=client)
    finally:
        p.disconnect(client)
    return report


def solvers():
    import numpy as np
    import torch
    from teleoperation.robots.native_hand import NativeHandModel
    from teleoperation.retargeting.hand.dex_native import NativeVectorSolver
    torch.set_num_threads(1)
    report = {"stage": 4, "hands": {}, "passed": True}
    keys = [("g1", "right")] + [(r, s) for r in ROBOT_SPECS for s in ("left", "right")
                                   if (r, s) != ("g1", "right")]
    for key in keys:
        model = NativeHandModel(*key)
        solver = NativeVectorSolver(model)
        rng = np.random.default_rng(0)
        poses = rng.uniform(model.bounds[:, 0], model.bounds[:, 1], (98, len(model.independent_names)))
        poses = np.vstack((poses, model.bounds.T))
        means, maxima, timings, failures = [], [], [], []
        for index, pose in enumerate(poses):
            target = model.positions(pose)
            result = solver.solve(target, index / 30)
            timings.append(result.solve_ms)
            if not result.valid:
                failures.append({"frame": index, "error": result.error})
                means.append(float("inf")); maxima.append(float("inf"))
                continue
            named = result.named_values()
            actual = model.positions([named[n] for n in model.independent_names])
            errors = np.linalg.norm(target - actual, axis=1)
            means.append(float(errors.mean())); maxima.append(float(errors.max()))
        target = model.positions(poses[0])
        solver.reset(); first = solver.solve(target, 10.)
        solver.reset(); second = solver.solve(target, 10.)
        repeatable = first.valid and second.valid and np.max(np.abs(first.values - second.values)) <= 1e-5
        previous = solver.sequence.last_qpos.copy()
        invalid = solver.solve(np.full_like(target, np.nan), 11.)
        assert not invalid.valid and np.array_equal(previous, solver.sequence.last_qpos)
        good_fraction = float(np.mean(np.asarray(means) <= .005))
        passed = not failures and good_fraction >= .95 and max(maxima) <= .01 and repeatable
        report["hands"]["/".join(key)] = {"poses": 100, "fraction_mean_error_le_5mm": good_fraction,
            "maximum_finger_error_mm": max(maxima) * 1000, "mean_error_mm": float(np.mean(means)) * 1000,
            "solve_p95_ms": float(np.percentile(timings, 95)), "failures": failures,
            "repeatable": bool(repeatable), "passed": bool(passed)}
        report["passed"] &= bool(passed)
    return report


def recording_evaluation(path):
    """Evaluate executed vs raw fingertip vectors, excluding calibration rows."""
    import h5py
    import numpy as np
    from teleoperation.robots.native_hand import NativeHandModel
    report = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "metric_scope": "human target fitting in native palm frame, not hardware accuracy", "hands": {}}
    with h5py.File(path, "r") as stream:
        excluded = np.asarray(stream["is_calibration"], dtype=bool)
        report["frames"] = len(excluded)
        report["calibration_frames_excluded"] = int(excluded.sum())
        report["evaluation_indices_sha256"] = hashlib.sha256(np.flatnonzero(~excluded).astype("<i8").tobytes()).hexdigest()
        for r in ROBOT_SPECS:
            for side in ("left", "right"):
                model = NativeHandModel(r, side)
                group = stream[f"{r}/{side}"]
                assert group.attrs["config_hash"] == model.config_hash
                names = json.loads(group.attrs["joint_names"])
                targets = np.asarray(group["target_vectors"])
                input_valid = np.isfinite(targets).all(axis=(1,2)) & ~excluded
                valid = np.asarray(group["valid"], dtype=bool) & input_valid
                row = {"input_valid": int(input_valid.sum()), "solved": int(valid.sum()),
                       "valid_fraction": float(valid.sum()/input_valid.sum()) if input_valid.any() else None}
                robot = model.kinematics
                order = [names.index(n) for n in robot.dof_joint_names]
                link_ids = [robot.get_link_index(n) for n in model.tip_frames]
                for field in ("raw_angles", "command_angles"):
                    angles = np.asarray(group[field])
                    means, distances, pinch = [], [], []
                    for index in np.flatnonzero(valid):
                        robot.compute_forward_kinematics(angles[index, order])
                        positions = np.array([robot.get_link_pose(i)[:3,3] for i in link_ids])
                        error = np.linalg.norm(positions-targets[index],axis=1)*1000
                        means.append(error.mean()); distances.extend(error)
                        pinch.append(np.abs(np.linalg.norm(positions[1:3]-positions[0],axis=1)-
                            np.linalg.norm(targets[index,1:3]-targets[index,0],axis=1))*1000)
                    delta = np.abs(np.diff(angles,axis=0))
                    finite_delta=delta[np.isfinite(delta)]
                    hits = sum(np.count_nonzero(np.isclose(angles[valid,i], model.limits[n][0],atol=1e-5) |
                                                 np.isclose(angles[valid,i], model.limits[n][1],atol=1e-5)) for i,n in enumerate(names))
                    row[field] = {"mean_tip_error_mm": float(np.mean(means)) if means else None,
                        "p95_tip_error_mm": float(np.percentile(distances,95)) if distances else None,
                        "thumb_index_middle_distance_error_mm": np.mean(pinch,axis=0).tolist() if pinch else None,
                        "limit_joint_fraction": float(hits/(valid.sum()*len(names))) if valid.any() else None,
                        "mean_joint_delta_rad": float(finite_delta.mean()) if len(finite_delta) else None,
                        "max_joint_delta_rad": float(finite_delta.max()) if len(finite_delta) else None}
                times=np.asarray(group["solve_ms"])[valid]
                row["solve_mean_ms"] = float(times.mean()) if len(times) else None
                row["solve_p95_ms"] = float(np.percentile(times,95)) if len(times) else None
                report["hands"][f"{r}/{side}"] = row
        frame_ms=sum(np.asarray(stream[f"{r}/{s}/solve_ms"]) for r in ROBOT_SPECS for s in ("left","right"))
        sample=frame_ms[100:]
        report["performance"]={"warmup_frames":100,"sample_frames":len(sample),
            "six_hand_p95_ms":float(np.percentile(sample,95)) if len(sample) else None,
            "p95_le_33ms":bool(len(sample)>=1000 and np.percentile(sample,95)<=33)}
        report["validity_passed"]=all(v["valid_fraction"] is not None and v["valid_fraction"]>=.99 for v in report["hands"].values())
        report["passed"] = report["validity_passed"]
    return report


def legacy_comparison(input_path, calibration_path, checkpoint, output, source="visionpro", method="checkpoint"):
    """Run the existing upstream VR model and legacy L21 native mapping as-is."""
    import numpy as np
    import torch
    import time
    from teleoperation.applications.dex_native import file_hash, command_result
    from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline
    from teleoperation.contracts.hand import NativeHandResult, L21HandAngles
    from teleoperation.data.hand_recording import iter_dex_observations, NativeHandResultWriter
    from teleoperation.data.urdf import parse_joint_limits
    from teleoperation.paths import L21_ROOT
    from teleoperation.robots.l21 import MAPPING_18
    from teleoperation.robots.native_hand import NativeHandModel, build_mapping, map_dofs
    from teleoperation.retargeting.hand.angles import canonical_to_native_angles
    from teleoperation.retargeting.hand.coordinates import OPERATOR_TO_CANONICAL
    from teleoperation.retargeting.hand.dex_native import DexInputAdapter, DexCalibration
    from teleoperation.retargeting.hand.transhandr_vr import load_transhandr_vr_model, TransHandRVRRetargeter
    from teleoperation.simulation.hand_commands import HandCommandPipeline
    torch.set_num_threads(1)
    calibration=DexCalibration.load(calibration_path)
    adapter=DexInputAdapter(source)
    if source == "visionpro":
        pipeline=TransHandRVisionProPipeline()
        model=load_transhandr_vr_model(checkpoint)
        predictors={s:TransHandRVRRetargeter(model,source_side=s,target_side=s) for s in ("left","right")}
    else:
        from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
        from teleoperation.applications.realtime_hand_sim import build_retargeter
        pipeline=MediaPipePalmLocalPipeline("world")
        camera_retargeter,_,_=build_retargeter(method,checkpoint or Path("unused"),"cpu",15,0.,"world")
    models={(r,s):NativeHandModel(r,s) for r in ROBOT_SPECS for s in ("left","right")}
    limits={}
    for side in ("left","right"):
        bounds=parse_joint_limits(L21_ROOT/side/f"linkerhand_l21_{side}.urdf")
        limits[side]=np.array([(0.,0.) if n is None else bounds[n] for n in MAPPING_18])
    commands=HandCommandPipeline(("left","right"),limits)
    mappings={k:build_mapping(*k,m.limits) for k,m in models.items()}
    transforms={k:OPERATOR_TO_CANONICAL[k[1]].T@m.palm_reference()[0].T for k,m in models.items()}
    digest=file_hash(input_path)
    metadata={"method":f"{source}_{method}_legacy_L21_mapping", "input_sha256":digest,
              "checkpoint_sha256":file_hash(checkpoint) if checkpoint else None,"calibration":vars(calibration),
              "coupling":"legacy independent mapped joints, not the dex mimic-constrained command path"}
    output.parent.mkdir(parents=True,exist_ok=True)
    with NativeHandResultWriter(output,models,metadata) as writer:
        writer.stream.attrs["schema"]="dex_native_comparison_v1"
        for raw in iter_dex_observations(input_path,source):
            frame=adapter.adapt(raw)
            window=pipeline.process_window(raw)
            angles,accepted,side_ms={},{},{}
            if source == "camera":
                start=time.perf_counter()
                camera_output=None if window is None else camera_retargeter.retarget(window)
                camera_ms=(time.perf_counter()-start)*1000
            for side in ("left","right"):
                start=time.perf_counter()
                if source == "visionpro":
                    result=None if window is None else getattr(predictors[side].retarget(window),side+"_angles")
                    side_ms[side]=(time.perf_counter()-start)*1000
                else:
                    result=None if camera_output is None else getattr(camera_output,side+"_angles")
                    side_ms[side]=camera_ms/2
                if result is None:
                    angles[side]=None; commands.missing(side)
                    accepted[side]=commands.previous[side].copy()
                else:
                    angles[side]=canonical_to_native_angles(result.values,side) if source=="visionpro" else result.values
                    accepted[side]=commands.update(side,angles[side],timestamp=raw.timestamp)
            results,applied,targets={},{},{}
            for key,m in models.items():
                side=key[1]
                values=None
                if angles[side] is not None:
                    named,_=map_dofs(L21HandAngles(angles[side]).native_mapping_dofs().values,mappings[key])
                    values=np.array([named[n] for n in m.joint_names])
                results[key]=NativeHandResult(*key,m.joint_names,values,raw.timestamp,m.config_hash,
                    values is not None,"missing/warmup" if values is None else None,side_ms[side]/3)
                applied[key]=map_dofs(L21HandAngles(accepted[side]).native_mapping_dofs().values,mappings[key])[0]
                targets[key]=(None if frame.hands[side] is None else
                    frame.hands[side][m.human_tip_indices]@transforms[key]*calibration.scale_for(m))
            writer.append(raw.timestamp,results,applied,targets,
                digest==calibration.source_sha256 and calibration.segment[0]<=raw.timestamp<=calibration.segment[1])
    return recording_evaluation(output)


def snapshots(path, directory):
    import h5py
    import numpy as np
    import pybullet as p
    from PIL import Image
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.robots.native_hand import link_extent
    scene=ThreeRobotHandScene(gui=False)
    directory.mkdir(parents=True,exist_ok=True)
    report={"source":str(path),"frame":100,"images":[],"mode":"accepted-command pose for visual review"}
    try:
        with h5py.File(path) as stream:
            for robot in scene.robots:
                ids=[]
                for side in ("left","right"):
                    group=stream[f"{robot['type']}/{side}"]
                    names=json.loads(group.attrs["joint_names"])
                    index=min(100,len(group["command_angles"])-1)
                    values=group["command_angles"][index]
                    lookup=robot["plans"][side][1]
                    for name,value in zip(names,values):
                        p.resetJointState(robot["body"],lookup[name],float(value),physicsClientId=scene.cid)
                        ids.append(lookup[name])
                center,size=link_extent(robot["body"],ids,scene.cid)
                base=p.getBasePositionAndOrientation(robot["body"],physicsClientId=scene.cid)[0]
                direction=center[:2]-np.asarray(base[:2])
                yaw=float(np.degrees(np.arctan2(direction[1],direction[0])))
                view=p.computeViewMatrixFromYawPitchRoll(center,max(.4,float(np.linalg.norm(size))*1.4),yaw,-18,0,2)
                projection=p.computeProjectionMatrixFOV(50,1.5,.01,10)
                rgba=p.getCameraImage(900,600,view,projection,renderer=p.ER_TINY_RENDERER,physicsClientId=scene.cid)[2]
                output=directory/f"{robot['type']}_native.png"
                Image.fromarray(np.asarray(rgba,dtype=np.uint8).reshape(600,900,4)).convert("RGB").save(output)
                report["images"].append(str(output))
    finally:
        scene.close()
    return report


def performance(input_path, calibration_path):
    import time
    import numpy as np
    import torch
    from teleoperation.applications.dex_native import recording_frames
    from teleoperation.retargeting.hand.dex_native import DexCalibration,DexNativeRetargeter
    torch.set_num_threads(1)
    calibration=DexCalibration.load(calibration_path)
    source="visionpro" if "visionpro" in calibration.profile else "camera"
    frames=recording_frames(input_path,source)
    if len(frames)<1100:
        raise ValueError("Performance requires at least 1100 recorded frames")
    retargeter=DexNativeRetargeter(calibration)
    times=[]
    for frame in frames[:1100]:
        start=time.perf_counter()
        retargeter.retarget(frame)
        times.append((time.perf_counter()-start)*1000)
    sample=times[100:]
    return {"warmup_frames":100,"sample_frames":len(sample),"six_hand_p95_ms":float(np.percentile(sample,95)),
        "mean_ms":float(np.mean(sample)),"passed":bool(np.percentile(sample,95)<=33),
        "scope":"six native solvers including wrapper, preprocessing/capture/render excluded",
        "input_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),"profile":calibration.profile}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("inventory", "kinematics", "solvers", "recording", "vr-comparison", "camera-comparison", "snapshots", "performance"), default="inventory")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result-h5", type=Path)
    parser.add_argument("--method", choices=("geometric", "checkpoint"), default="checkpoint")
    args = parser.parse_args()
    if args.stage == "performance":
        if args.input is None or args.calibration is None:
            parser.error("performance requires input and calibration")
        report=performance(args.input,args.calibration)
    elif args.stage == "snapshots":
        if args.input is None:
            parser.error("snapshots requires native result input")
        report=snapshots(args.input,args.output.parent/"snapshots")
    elif args.stage in ("vr-comparison", "camera-comparison"):
        if any(v is None for v in (args.input,args.calibration,args.result_h5)) or (args.method=="checkpoint" and args.checkpoint is None):
            parser.error("comparison requires input, calibration, result-h5 and checkpoint for a checkpoint method")
        source="visionpro" if args.stage=="vr-comparison" else "camera"
        if source=="visionpro" and args.method!="checkpoint":
            parser.error("Existing VR baseline only supports the explicit upstream checkpoint route")
        report=legacy_comparison(args.input,args.calibration,args.checkpoint,args.result_h5,source,args.method)
    elif args.stage == "recording":
        if args.input is None:
            parser.error("recording stage requires --input native-results.h5")
        report = recording_evaluation(args.input)
    else:
        report = {"inventory": inventory, "kinematics": kinematics, "solvers": solvers}[args.stage]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"stage={args.stage} passed={report.get('passed', True)}")
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
