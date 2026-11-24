"""PyBullet URDF loading helpers that work under non-ASCII project paths."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import time
import numpy as np


def load_urdf(urdf_path: str | Path, **kwargs: Any) -> int:
    """Load a URDF through an ASCII-safe relative filename.

    PyBullet on Windows can fail when an absolute URDF path contains
    non-ASCII characters.  Loading through a relative filename while the
    URDF directory is the current working directory avoids that limitation.
    """
    import pybullet as p

    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF was not found: {path}")

    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        return p.loadURDF(path.name, **kwargs)
    finally:
        os.chdir(previous)


def create_ground(pybullet: Any, physicsClientId: int = 0) -> int:
    """Create the standard ground plane without loading pybullet_data assets."""
    half_extents = [100.0, 100.0, 5.0]
    collision = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=physicsClientId,
    )
    visual = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        physicsClientId=physicsClientId,
    )
    return pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, -5.0],
        physicsClientId=physicsClientId,
    )


def prepare_physics(call, positional, *, client_id, timeout=15.0):
    """Wait for the selected GUI server to answer a setup call."""
    deadline = time.perf_counter() + timeout
    while True:
        try:
            call(*positional, physicsClientId=client_id)
            return
        except Exception as error:
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"PyBullet did not answer within {timeout:.0f} s "
                    f"({type(error).__name__}: {error}); rerun with --headless."
                ) from error
            time.sleep(0.1)


def prepare_tron2a_urdf(urdf_path: str | Path, directory: Path, *, display="full", visible_links=None) -> Path:
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    if display not in ("full", "simplified"):
        raise ValueError("Display must be full or simplified")
    source = Path(urdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TRON2A URDF was not found: {source}")
    tree = ET.parse(source)
    root = tree.getroot()
    if visible_links is not None:
        for link in root.findall("link"):
            if link.get("name") not in visible_links:
                for node in list(link.findall("visual")) + list(link.findall("collision")):
                    link.remove(node)
    missing = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            suffix = filename.split("/meshes/", 1)
            path = source.parent.parent / "meshes" / suffix[1] if len(suffix) == 2 else source.parent / filename[10:]
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = source.parent / path
        if display == "full":
            if not path.is_file():
                missing.append(str(path))
            mesh.set("filename", path.resolve().as_posix())
    if missing:
        raise FileNotFoundError("TRON2A mesh resources missing; install them or explicitly use --display simplified: " + "; ".join(sorted(set(missing))))
    if display == "simplified":
        for link in root.findall("link"):
            if visible_links is not None and link.get("name") not in visible_links:
                continue
            # Replace mesh proxies only; keep all original joints and inertias.
            for element in link.findall("visual") + link.findall("collision"):
                geometry = element.find("geometry")
                if geometry is not None and geometry.find("mesh") is not None:
                    geometry.remove(geometry.find("mesh"))
                    ET.SubElement(geometry, "sphere", radius="0.035")
            # Rods between link origin and child joint origins make the chain visible.
            for joint in root.findall("joint"):
                if joint.find("parent").get("link") != link.get("name"):
                    continue
                origin = joint.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                length = np.linalg.norm(xyz)
                if length < 0.02:
                    continue
                visual = ET.SubElement(link, "visual")
                rotation, _ = Rotation.align_vectors([xyz / length], [[0, 0, 1]])
                ET.SubElement(visual, "origin", xyz=" ".join(map(str, xyz / 2)), rpy=" ".join(map(str, rotation.as_euler("xyz"))))
                ET.SubElement(ET.SubElement(visual, "geometry"), "cylinder", radius="0.015", length=str(length))
                material = ET.SubElement(visual, "material", name="preview")
                ET.SubElement(material, "color", rgba="0.3 0.65 0.85 1")
    target = directory / "robot_pybullet.urdf"
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target
