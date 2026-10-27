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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("inventory", "kinematics", "solvers"), default="inventory")
    args = parser.parse_args()
    report = {"inventory": inventory, "kinematics": kinematics, "solvers": solvers}[args.stage]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"stage={args.stage} passed={report.get('passed', True)}")
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
