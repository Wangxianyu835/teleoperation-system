"""Native hand geometry and URDF coupling for target-specific and VR solvers."""
import numpy as np

PREFIX = {
    'h1_2':   {'left': 'L_', 'right': 'R_'},
    'gr1_t2': {'left': 'L_', 'right': 'R_'},
    'g1':     {'left': 'left_hand_', 'right': 'right_hand_'},
}


def link_extent(robot_id, link_ids, client):
    """合并若干 link 的 AABB -> (中心, 各轴边长向量)；无有效 link 时返回 (None, None)

    用途：给「手部特写」相机算目标点与距离。
    [注意] 必须用 AABB，不能用关节原点 —— 关节原点只是 link 坐标系原点，
    手指网格可能偏离好几厘米。而且三台机器人的手在【世界】里的方位
    完全不同（实测右手：H1-2 x=+0.36 / GR1-T2 x=-0.19 / G1 x=+0.25），
    用固定 yaw 的相机会把镜头塞进 GR1-T2 的躯干内部，只能拍到面片背面。

    返回 size 用【向量】而不是标量：`--hand both` 时两只手合起来是一个
    横长条（实测 GR1-T2 宽 0.55 m），用最大边长会算错相机距离，
    用对角线长度 |size| 才是稳的。
    """
    import pybullet as p
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for i in link_ids:
        try:
            a, b = p.getAABB(robot_id, i, physicsClientId=client)
        except p.error:
            continue
        lo = np.minimum(lo, np.asarray(a, dtype=float))
        hi = np.maximum(hi, np.asarray(b, dtype=float))
    if not np.isfinite(lo).all():
        return None, None
    return (lo + hi) / 2.0, (hi - lo)



class NativeHandModel:
    """Actual hand subtree, named independent coordinates and URDF mimic rules.

    Generated, geometry-free URDFs are optimization resources, never replacements
    for the displayed whole-body models.
    """

    def __init__(self, robot_type, side, cache_dir=None):
        import copy
        import hashlib
        from pathlib import Path
        import xml.etree.ElementTree as ET
        from teleoperation.paths import PROJECT_ROOT, ROBOTS_ROOT
        from teleoperation.robots.specification import ROBOT_SPECS

        if robot_type not in ROBOT_SPECS or side not in ("left", "right"):
            raise ValueError("Unknown native hand")
        self.robot_type, self.side = robot_type, side
        self.source_path = Path(ROBOTS_ROOT) / ROBOT_SPECS[robot_type]["urdf"]
        source = ET.parse(self.source_path).getroot()
        prefix = PREFIX[robot_type][side]
        self.root_link = ({"h1_2": prefix + "hand_base_link",
                           "gr1_t2": side[0] + "_hand_base_link",
                           "g1": prefix + "palm_link"})[robot_type]
        selected, links = [], {self.root_link}
        while True:
            added = [j for j in source.findall("joint") if j not in selected
                     and j.find("parent").get("link") in links]
            if not added:
                break
            selected.extend(added)
            links.update(j.find("child").get("link") for j in added)
        tree = ET.Element("robot", name=f"dex_{robot_type}_{side}")
        for link in source.findall("link"):
            if link.get("name") in links:
                node = copy.deepcopy(link)
                for visual in list(node.findall("visual")) + list(node.findall("collision")):
                    node.remove(visual)
                tree.append(node)
        tree.extend(copy.deepcopy(j) for j in selected)
        active = [j for j in selected if j.get("type") != "fixed"]
        self.joint_names = tuple(j.get("name") for j in active)
        self.independent_names = tuple(j.get("name") for j in active if j.find("mimic") is None)
        expected = {"h1_2": (12, 6), "gr1_t2": (11, 6), "g1": (7, 7)}[robot_type]
        if (len(active), len(self.independent_names)) != expected:
            raise ValueError(f"Unexpected hand topology: {robot_type}/{side}")
        self.limits = {j.get("name"): (float(j.find("limit").get("lower")),
                                       float(j.find("limit").get("upper"))) for j in active}
        velocities = {j.get("name"): float(j.find("limit").get("velocity")) for j in active}
        mimic = {j.get("name"): j.find("mimic").attrib for j in active if j.find("mimic") is not None}
        def relation(name, visiting=()):
            if name in visiting:
                raise ValueError("Cyclic mimic chain")
            if name not in mimic:
                return name, 1., 0.
            rule = mimic[name]
            driver, a, b = relation(rule["joint"], (*visiting, name))
            multiplier, offset = float(rule.get("multiplier", 1)), float(rule.get("offset", 0))
            return driver, a * multiplier, b * multiplier + offset
        self.relations = {name: relation(name) for name in self.joint_names}
        self.bounds = np.array([self.limits[n] for n in self.independent_names], dtype=float)
        self.velocity_limits = np.array([velocities[n] for n in self.independent_names])
        for name, (driver, a, b) in self.relations.items():
            index = self.independent_names.index(driver)
            low, high = self.limits[name]
            if a == 0:
                if not low <= b <= high:
                    raise ValueError("Impossible constant mimic joint")
                continue
            projected = sorted(((low - b) / a, (high - b) / a))
            self.bounds[index, 0] = max(self.bounds[index, 0], projected[0])
            self.bounds[index, 1] = min(self.bounds[index, 1], projected[1])
            self.velocity_limits[index] = min(self.velocity_limits[index], velocities[name] / abs(a))
        if np.any(self.bounds[:, 0] > self.bounds[:, 1]) or np.any(self.velocity_limits <= 0):
            raise ValueError("Infeasible native joint constraints")
        self.fingers = ("thumb", "index", "middle") if robot_type == "g1" else (
            "thumb", "index", "middle", "ring", "pinky")
        self.human_tip_indices = np.array([4, 9, 14, 19, 24][:len(self.fingers)])
        self.tip_frames, self.tip_offsets = [], []
        donor_bytes = b""
        if robot_type != "gr1_t2":
            donor = Path(ROBOTS_ROOT) / (f"inspire_hand/inspire_hand_{side}.urdf" if robot_type == "h1_2"
                                           else f"unitree_hand/unitree_dex3_{side}.urdf")
            donor_bytes = donor.read_bytes()
            donor_tree = ET.fromstring(donor_bytes)
        for finger in self.fingers:
            if robot_type == "gr1_t2":
                frame = prefix + finger + "_tip"
                self.tip_frames.append(frame)
                self.tip_offsets.append((frame, np.zeros(3)))
                continue
            node = next(j for j in donor_tree.findall("joint")
                        if j.find("child").get("link") == finger + "_tip")
            parent = node.find("parent").get("link")
            if robot_type == "h1_2":
                parent = prefix + parent
            if parent not in links:
                raise ValueError(f"Tip donor parent is absent: {parent}")
            frame = f"dex_{side}_{finger}_tip"
            fixed = copy.deepcopy(node)
            fixed.set("name", frame + "_joint")
            fixed.find("parent").set("link", parent)
            fixed.find("child").set("link", frame)
            tree.append(ET.Element("link", name=frame))
            tree.append(fixed)
            xyz = np.fromstring(node.find("origin").get("xyz"), sep=" ")
            self.tip_frames.append(frame)
            self.tip_offsets.append((parent, xyz))
        xml = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
        self.config_hash = hashlib.sha256(self.source_path.read_bytes() + donor_bytes + xml).hexdigest()
        directory = Path(cache_dir) if cache_dir else PROJECT_ROOT / "outputs/tmp_dex_native/models"
        directory.mkdir(parents=True, exist_ok=True)
        self.urdf_path = directory / f"{robot_type}_{side}_{self.config_hash[:16]}.urdf"
        if not self.urdf_path.exists():
            self.urdf_path.write_bytes(xml)
        self._kinematics = None

    @property
    def kinematics(self):
        if self._kinematics is None:
            try:
                from dex_retargeting.robot_wrapper import RobotWrapper
            except ImportError as error:
                raise ImportError("Native dex requires the environment documented in docs/dex-native.md") from error
            self._kinematics = RobotWrapper(str(self.urdf_path))
        return self._kinematics

    def expand(self, independent):
        values = np.asarray(independent, dtype=float)
        if values.shape != (len(self.independent_names),) or not np.isfinite(values).all():
            raise ValueError("Invalid independent native hand coordinates")
        if np.any(values < self.bounds[:, 0] - 1e-8) or np.any(values > self.bounds[:, 1] + 1e-8):
            raise ValueError("Independent command exceeds coupled joint limits")
        drivers = dict(zip(self.independent_names, np.clip(values, *self.bounds.T)))
        return {name: drivers[driver] * a + b for name, (driver, a, b) in self.relations.items()}

    def positions(self, independent, frames=None):
        robot = self.kinematics
        joints = self.expand(independent)
        robot.compute_forward_kinematics(np.array([joints[n] for n in robot.dof_joint_names]))
        return np.array([robot.get_link_pose(robot.get_link_index(n))[:3, 3]
                         for n in (frames or self.tip_frames)])

    def palm_reference(self):
        from teleoperation.retargeting.hand.coordinates import build_palm_basis
        prefix = PREFIX[self.robot_type][self.side]
        suffix = "_0_link" if self.robot_type == "g1" else "_proximal"
        points = self.positions(np.clip(np.zeros(len(self.independent_names)), *self.bounds.T),
                                [self.root_link, prefix + "index" + suffix, prefix + "middle" + suffix])
        return build_palm_basis(points, (0, 1, 2)), float(np.linalg.norm(points[2] - points[0]))

    def bullet_positions(self, p, body, client, independent):
        """Query original whole-body geometry, restoring positions AND velocities."""
        infos = [p.getJointInfo(body, i, physicsClientId=client)
                 for i in range(p.getNumJoints(body, physicsClientId=client))]
        joints = {info[1].decode(): i for i, info in enumerate(infos)}
        links = {info[12].decode(): i for i, info in enumerate(infos)}
        movable = [i for i, info in enumerate(infos) if info[2] != p.JOINT_FIXED]
        saved = p.getJointStates(body, movable, physicsClientId=client)
        try:
            for name, value in self.expand(independent).items():
                p.resetJointState(body, joints[name], value, physicsClientId=client)
            def pose(name):
                state = p.getLinkState(body, links[name], computeForwardKinematics=True, physicsClientId=client)
                return state[4], state[5]
            inverse = p.invertTransform(*pose(self.root_link))
            result = []
            for name, offset in self.tip_offsets:
                world = p.multiplyTransforms(*pose(name), offset, [0, 0, 0, 1])
                result.append(p.multiplyTransforms(*inverse, *world)[0])
            return np.array(result)
        finally:
            for index, state in zip(movable, saved):
                p.resetJointState(body, index, state[0], targetVelocity=state[1], physicsClientId=client)
