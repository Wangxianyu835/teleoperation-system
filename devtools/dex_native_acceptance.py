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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[OK] six native hands: H1 6/12, GR1 6/11, G1 7/7")


if __name__ == "__main__":
    main()
