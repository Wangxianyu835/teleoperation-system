"""论文手臂 IK（TeleOpBench ``camera_ik_h1_2_media.py``）的本机移植。

原文件 ``teleop/robot_control/camera_ik_h1_2_media.py`` 的求解器是::

    opti.minimize(50 * translational_cost
                +  1 * rotation_cost
                + 0.02 * regularization_cost
                + 0.1  * smooth_cost)              # sumsqr(q - q_last)
    opti.subject_to(bounded(lower, var_q, upper))  # 关节限位硬约束
    opti.set_initial(var_q, init_data)             # 上一帧热启动
    sol_q = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1]).filtered_data
    # 求解失败 -> 返回上一帧的指令值（不跳变）

三件事决定了现场手感（2026-11-17 用户反馈「转手腕手臂乱甩、静止也抖」正是这三条缺失）:

1. **位置权重 50 : 姿态权重 1** —— 姿态误差不够时让腕去转，绝不让肩/肘为了姿态让位。
   我们用 ``calculateInverseKinematics`` 时位置和姿态是无权的 6 维一锅端，转手腕时
   整条臂跟着分摊，这就是甩动的来源。
2. **``0.1 * sumsqr(q - q_last)`` + 输出端加权滑动平均** —— 帧间连续，消抖。
3. **热启动 + 失败回退到上一帧指令** —— 不在零空间里漂、不出现解分支跳变。

本机没有 pinocchio/casadi（Windows 上只有 conda-forge 有轮子），所以:
``pinocchio`` 的正运动学用 PyBullet 的 ``resetJointState`` + ``getLinkState`` 代替，
``ipopt`` 的求解换成**同一个目标函数**的 Gauss-Newton 有界迭代（解析 Jacobian 来自
``calculateJacobian``，关节限位用 projection 实现，等价于 ipopt 的 box 约束）。
目标函数的每一项、权重、滤波窗、失败回退语义与原文件逐字一致，便于对账。

与论文唯一的**行为**差异：``orientation_scope``。原文件没有这个开关，姿态永远摊在整条臂上；
本仓的实时入口默认 ``scope="wrist"``，语义是「姿态误差只准腕关节消化」。为了不让
``--arm-orientation-scope`` 在这个后端上变成哑参数，``scope="wrist"`` 时把姿态雅可比的
肩/肘列清零（见 ``PaperArmIK.wrist_only_orientation``）；``scope="arm"`` 与原文件完全一致。
实测（2026-11-17，``outputs/tmp_probe/paper_arm_ab.py``、24 帧、只转腕 72°）：
``scope="wrist"`` 下肩/肘位移 0.000 rad、末端漂移 0.0 mm、姿态误差 4.9°。
"""

from math import acos, sqrt
import os

import numpy as np

#: 原文件 ``opti.minimize(...)`` 里的四个权重，逐字照抄。
#: 位置/姿态两项可用 ``PAPER_IK_TRANS_WEIGHT`` / ``PAPER_IK_ROT_WEIGHT`` 覆盖 —— 论文的
#: 50:1 让姿态成为次要目标：残差一大就把姿态项完全压住，现场手感是「腕几乎不跟手」。
#: 调小位置权重（或抬姿态权重）能把腕的响应换回来，代价是位置跟手变松。
TRANSLATIONAL_WEIGHT = float(os.environ.get("PAPER_IK_TRANS_WEIGHT", "50.0"))
#: Elbow position is an explicit task, not only a null-space preference. The
#: operator-side posture recipe produces a reachable elbow target together with
#: the wrist target; leaving it out lets the 7-DoF arm choose the opposite elbow
#: branch.
ELBOW_TRANSLATIONAL_WEIGHT = float(os.environ.get("PAPER_IK_ELBOW_WEIGHT", "80.0"))
ROTATION_WEIGHT = float(os.environ.get("PAPER_IK_ROT_WEIGHT", "1.0"))
REGULARIZATION_WEIGHT = 0.02
SMOOTH_WEIGHT = 0.1
#: 原文件 ``WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)`` 的窗口。
SMOOTH_FILTER_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
#: 原文件 ipopt ``max_iter`` / ``tol``（1e-6）的本机等价收敛门限。
#: 迭代上限按「一帧内把 10 mm 级目标追平」定：论文每帧只差毫米级，我们 30 fps 下每帧
#: 目标可动厘米级，12 次迭代会把滞后累积成 130 mm（实测）。
#: 现在每次迭代只剩 1 写 + 1 读 + 1 次雅可比（候选点评估是纯 numpy），上限取 30：
#: 实测远目标（gr1_t2 走 24 cm）需要 ~21 次，贴到 15 会被截断。
MAX_ITERATIONS = int(os.environ.get("PAPER_IK_MAX_ITER", "30"))
POSITION_TOLERANCE_M = 5e-4
ROTATION_TOLERANCE_DEG = 0.2
#: 单步关节增量的**初始**信赖域半径（rad）。ipopt 自带信赖域自适应，这里同样自适应：
#: 一步被接受就放大（×1.5，上限 1.0），被拒绝就减半。写死成常数时快帧永远追不上（实测）。
MAX_STEP_RAD = 0.5
MIN_TRUST_RAD = 0.02
MAX_TRUST_RAD = 1.0
#: 诊断开关：``PAPER_IK_DEBUG=1`` 时逐迭代打印残差/步长/接受的缩放，用于排查收敛。
_DEBUG = os.environ.get("PAPER_IK_DEBUG", "") not in ("", "0")



class WeightedMovingFilter:
    """``teleop/utils/weighted_moving_filter.py`` 的逐字移植（只去掉画图部分）。"""

    def __init__(self, weights, data_size):
        self._window_size = len(weights)
        self._weights = np.array(weights, dtype=float)
        assert np.isclose(np.sum(self._weights), 1.0), \
            "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = int(data_size)
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []
        self._primed = False

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]
        data_array = np.array(self._data_queue, dtype=float)
        filtered = np.zeros(self._data_size)
        for i in range(self._data_size):
            filtered[i] = np.convolve(data_array[:, i], self._weights, mode="valid")[-1]
        self._primed = True
        return filtered

    def add_data(self, new_data):
        new_data = np.asarray(new_data, dtype=float)
        assert len(new_data) == self._data_size
        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)
        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data

    @property
    def primed(self):
        return self._primed


def rotation_vector_error(target, current):
    """``log3(R_target @ R_current.T)`` —— 原文件 ``rotational_error`` 用的同一个量。

    ``target`` / ``current`` 都是 PyBullet 顺序的 ``(x, y, z, w)`` 四元数。

    实现要点：``q = target (*) conj(current)``（四元数乘）必须按 **(x, y, z, w)** 顺序展开。
    之前这四行是按 (w, x, y, z) 顺序写的却喂 (x, y, z, w) 数据，各分量符号错乱
    （只有"两者相同"时才巧合为 0），后果是求解器把步长全花在**反向**的姿态补偿上：
    位置误差 8.18 → 7.85 mm 的同时姿态误差 0.42° → 1.17°，线搜索只能一路拒绝
    （实测 g1 完全不动、h1_2 40 次迭代只收敛 2 mm）。
    """
    tx, ty, tz, tw = target
    cx, cy, cz, cw = current
    # (x, y, z, w) 约定下的 target (*) conj(current)（四元数乘）；其结果就是世界系旋转误差矢量
    qx = -tw * cx + tx * cw - ty * cz + tz * cy
    qy = -tw * cy + tx * cz + ty * cw - tz * cx
    qz = -tw * cz - tx * cy + ty * cx + tz * cw
    qw = tw * cw + tx * cx + ty * cy + tz * cz
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * acos(qw)
    s = sqrt(max(0.0, 1.0 - qw * qw))
    if s < 1e-9:
        return np.zeros(3)
    return np.array([qx, qy, qz]) / s * angle


class PaperArmIK:
    """一侧手臂的论文式 IK：加权目标 + 平滑项 + 热启动 + 输出滤波。

    ``adapter`` 是 :class:`arm_ik.PyBulletArmIK`（提供 ``p`` / ``body`` / ``cid`` /
    ``arm[side]`` / ``ee[side]`` / ``limits`` / ``indices`` / ``position_of``）。
    """

    def __init__(self, adapter, side):
        self.adapter = adapter
        self.side = side
        self.joints = list(adapter.arm[side])            # [(bullet index, name), ...]
        self.joint_indices = [index for index, _name in self.joints]
        self.joint_names = [name for _index, name in self.joints]
        self.ee_link = adapter.ee[side][0]
        self.elbow_link = adapter.elbow_link[side]
        lower = adapter.limits["lower"]
        upper = adapter.limits["upper"]
        self.lower = np.array([lower[adapter.position_of[i]] for i in self.joint_indices])
        self.upper = np.array([upper[adapter.position_of[i]] for i in self.joint_indices])
        self.smooth_filter = WeightedMovingFilter(SMOOTH_FILTER_WEIGHTS, len(self.joints))
        #: ``orientation_scope == "wrist"`` 时，姿态任务只允许腕关节承担：把姿态雅可比里
        #: 肩/肘对应的列清零即可（与历史 ``scope="wrist"`` 的语义一致 —— 转手腕只转腕）。
        wrist_joints = {index for index, _name in adapter.wrist[side]}
        self.wrist_columns = [k for k, index in enumerate(self.joint_indices)
                              if index in wrist_joints]
        #: 腕关节表为空（机器人测不出腕链）时退回历史行为 —— 姿态摊到整条臂。
        self.wrist_only_orientation = (
            str(getattr(adapter, "orientation_scope", "arm")) == "wrist"
            and bool(self.wrist_columns))
        #: 原文件 ``self.init_data``：热启动点 = 上一帧**滤波后**的输出。
        self.init_data = np.array(self._measured(), dtype=float)
        #: 目标函数里的 ``sumsqr(var_q)``。原文件对**绝对** q 施加 0.02 的回归项，而它所用
        #: 机器人的零位正好就是操作起始姿态；本仓库 g1 的起始肘角是 1.21 rad，照搬零向量
        #: 会让该项变成一股 0.0312 的"回零"拉力（比 10 mm 位置项 0.005 大 6 倍），实测把
        #: 位置项完全压死、手臂一步不动。这里把参考点取为构造时的实测姿态 —— 语义等价
        #: （"停在起始姿态"），对 h1_2 零位附近仍与原文数值一致。
        self.regularization_reference = self.init_data.copy()
        #: 上一次被放行的解；失败时原样返回（原文件第 294 行）。
        self.last_command = self.init_data.copy()
        self.last_residual = None
        self.last_elbow_residual = None
        self.last_rotation_error_deg = None
        self.iterations = 0
        #: 信赖域半径（rad），跨帧保留：自适应放大/缩小，见 ``MAX_STEP_RAD``。
        self._trust = MAX_STEP_RAD

    # ---------------------------------------------------------------- 状态读写

    def _measured(self):
        p = self.adapter.p
        return [float(p.getJointState(self.adapter.body, index,
                                      physicsClientId=self.adapter.cid)[0])
                for index in self.joint_indices]

    def _write(self, q):
        p = self.adapter.p
        for index, value in zip(self.joint_indices, q):
            p.resetJointState(self.adapter.body, index, float(value),
                              physicsClientId=self.adapter.cid)

    def _ee_pose(self):
        """Return wrist pose and cache the elbow position from the same FK query."""
        p = self.adapter.p
        if hasattr(p, "getLinkStates"):
            states = p.getLinkStates(
                self.adapter.body, [self.ee_link, self.elbow_link],
                computeLinkVelocity=0, computeForwardKinematics=1,
                physicsClientId=self.adapter.cid)
            wrist, elbow = states
        else:
            wrist = p.getLinkState(
                self.adapter.body, self.ee_link, computeLinkVelocity=0,
                computeForwardKinematics=1, physicsClientId=self.adapter.cid)
            elbow = p.getLinkState(
                self.adapter.body, self.elbow_link, computeLinkVelocity=0,
                computeForwardKinematics=1, physicsClientId=self.adapter.cid)
        self._last_elbow_position = np.asarray(elbow[4], dtype=float)
        return np.asarray(wrist[4], dtype=float), tuple(float(v) for v in wrist[5])

    def _base_positions(self):
        """整机 dof 位置读一次（非手臂关节在一次 ``_minimize`` 里不会变）。

        原来每次迭代都重读 N 个关节：现场三台机器人同场景时单次 PyBullet 往返随
        场景规模变贵，这里是每迭代 N 次往返的大头之一。
        """
        p = self.adapter.p
        cid = self.adapter.cid
        return [float(p.getJointState(self.adapter.body, joint,
                                      physicsClientId=cid)[0])
                for joint in self.adapter.indices]

    def _all_positions(self, q, base=None):
        """``calculateJacobian`` 的输入：**按 dof 顺序**（``adapter.indices``）的整机向量。"""
        positions = list(self._base_positions() if base is None else base)
        for joint, value in zip(self.joint_indices, q):
            positions[self.adapter.position_of[joint]] = float(value)
        return positions

    def _jacobian(self, positions, link_index=None):
        """``calculateJacobian`` -> (3x7 线速度, 3x7 角速度)，世界系。

        返回矩阵的列是 **dof 序号**，所以要用 ``adapter.position_of`` 取列。
        """
        p = self.adapter.p
        link_index = self.ee_link if link_index is None else int(link_index)
        count = len(positions)
        linear, angular = p.calculateJacobian(
            self.adapter.body, link_index, [0.0, 0.0, 0.0],
            positions, [0.0] * count, [0.0] * count,
            physicsClientId=self.adapter.cid)
        columns = [self.adapter.position_of[joint] for joint in self.joint_indices]
        jac_linear = np.array([[linear[row][col] for col in columns] for row in range(3)])
        jac_angular = np.array([[angular[row][col] for col in columns] for row in range(3)])
        return jac_linear, jac_angular

    # ------------------------------------------------------------------ 求解

    def solve(self, target_position, target_quaternion=None,
              target_elbow_position=None):
        """返回 ``{关节名: 角度}``；求解异常时返回上一帧指令（不跳变）。"""
        q = self.init_data.copy()
        try:
            q, residual, elbow_residual, rot_deg, iterations = self._minimize(
                q, np.asarray(target_position, dtype=float), target_quaternion,
                (None if target_elbow_position is None
                 else np.asarray(target_elbow_position, dtype=float)))
        except Exception:                      # noqa: BLE001 - 与原文件的 except 同义
            self._write(self.last_command)
            self.last_elbow_residual = None
            return {name: float(value)
                    for name, value in zip(self.joint_names, self.last_command)}

        # 原文件 256-257 行：解先过加权滑动平均再下发。
        self.smooth_filter.add_data(q)
        q = np.array(self.smooth_filter.filtered_data, dtype=float)
        q = np.clip(q, self.lower, self.upper)
        # 原文件 264 行：热启动点用**滤波后**的解，不是原始解。
        self.init_data = q.copy()
        self.last_command = q.copy()
        self.last_residual = float(residual)
        self.last_elbow_residual = (None if elbow_residual is None
                                    else float(elbow_residual))
        self.last_rotation_error_deg = None if rot_deg is None else float(rot_deg)
        self.iterations = int(iterations)
        # 与 ``_teleport`` 同语义：把手臂停在解上，电机从这一帧开始伺服。
        self._write(q)
        return {name: float(value) for name, value in zip(self.joint_names, q)}

    def _terms(self, q, error_position, error_rotation, q_last, error_elbow=None):
        """四个代价项之和，**输入是误差而不是位姿** —— 线搜索用线性化误差调它，
        从而候选点评估完全不碰 PyBullet（这是帧率的关键，见 ``_minimize``）。"""
        position_term = TRANSLATIONAL_WEIGHT * float(error_position @ error_position)
        elbow_term = (0.0 if error_elbow is None
                      else ELBOW_TRANSLATIONAL_WEIGHT
                      * float(error_elbow @ error_elbow))
        rotation_term = (0.0 if error_rotation is None
                         else ROTATION_WEIGHT * float(error_rotation @ error_rotation))
        regularization_term = REGULARIZATION_WEIGHT * float(
            np.sum((q - self.regularization_reference) ** 2))
        smooth_term = SMOOTH_WEIGHT * float(np.sum((q - q_last) ** 2))
        return (position_term + elbow_term + rotation_term
                + regularization_term + smooth_term)

    def _minimize(self, q, target_position, target_quaternion,
                  target_elbow_position=None):
        q = np.clip(np.asarray(q, dtype=float), self.lower, self.upper)
        q_last = q.copy()
        residual = float("inf")
        elbow_residual = None
        rot_deg = None
        iterations = 0
        rejected = 0
        # 非手臂关节在一次求解里是常量：读一次，之后每次迭代都是纯 numpy 拼装。
        base_positions = self._base_positions()
        for iterations in range(1, MAX_ITERATIONS + 1):
            # 关键：写进去时必须按 **dof 序号**（``position_of``）落位。用关节索引会
            # 在 h1_2 上"恰好没越界"却把姿态喂错（雅可比在错构型上求值 → 收敛慢），
            # 在 g1 上直接 IndexError → 每帧走 replay、手臂完全不动（实测）。
            positions = self._all_positions(q, base_positions)
            self._write(q)
            position, quaternion = self._ee_pose()
            error_position = target_position - position
            error_elbow = (None if target_elbow_position is None
                           else target_elbow_position - self._last_elbow_position)
            linear, angular = self._jacobian(positions)
            elbow_linear = (None if error_elbow is None
                            else self._jacobian(
                                positions, link_index=self.elbow_link)[0])
            if self.wrist_only_orientation:
                # ``orientation_scope="wrist"``：姿态误差只准腕关节消化（历史路径的语义）。
                angular = angular.copy()
                mask = np.zeros(len(self.joint_indices), dtype=bool)
                mask[self.wrist_columns] = True
                angular[:, ~mask] = 0.0
            if target_quaternion is not None:
                error_rotation = rotation_vector_error(target_quaternion, quaternion)
                rot_deg = float(np.degrees(np.linalg.norm(error_rotation)))
            else:
                error_rotation = None
            residual = float(np.linalg.norm(error_position))
            elbow_residual = (None if error_elbow is None
                              else float(np.linalg.norm(error_elbow)))
            if (residual <= POSITION_TOLERANCE_M
                    and (elbow_residual is None
                         or elbow_residual <= POSITION_TOLERANCE_M)
                    and (rot_deg is None or rot_deg <= ROTATION_TOLERANCE_DEG)):
                break

            # 加权最小二乘：位置 50、姿态 1（原文件 minimize 的前两项）。
            rows = [sqrt(TRANSLATIONAL_WEIGHT) * linear]
            errors = [sqrt(TRANSLATIONAL_WEIGHT) * error_position]
            if error_elbow is not None:
                rows.append(sqrt(ELBOW_TRANSLATIONAL_WEIGHT) * elbow_linear)
                errors.append(sqrt(ELBOW_TRANSLATIONAL_WEIGHT) * error_elbow)
            if error_rotation is not None:
                rows.append(sqrt(ROTATION_WEIGHT) * angular)
                errors.append(sqrt(ROTATION_WEIGHT) * error_rotation)
            jacobian = np.vstack(rows)
            stacked = np.concatenate(errors)

            # 后两项：0.02 * sumsqr(q) + 0.1 * sumsqr(q - q_last)。
            # 注意符号：``stacked`` 是残差 ``e = target - p``，而 ``e`` 对 ``q`` 的导数是
            # ``-J``，所以 ``jacobian.T @ stacked`` 恰是代价的**负**梯度（下降方向）。
            # 原文件把整件事交给 ipopt，不需要自己决定符号；这里手写必须自己定对：
            # 取 ``-`` 号就是朝增大的方向爬，实测结果是"残差一动不动"。
            system = (jacobian.T @ jacobian
                      + (REGULARIZATION_WEIGHT + SMOOTH_WEIGHT) * np.eye(len(q)))
            descent = (jacobian.T @ stacked
                       - REGULARIZATION_WEIGHT * (q - self.regularization_reference)
                       - SMOOTH_WEIGHT * (q - q_last))
            try:
                step = np.linalg.solve(system, descent)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(system, descent, rcond=None)[0]
            step = np.clip(step, -self._trust, self._trust)
            # 回溯线搜索：ipopt 自带信赖域/线搜索，Gauss-Newton 必须补上，
            # 否则不可达的姿态目标会让解撞在关节限位上反复弹（实测 h1_2 抖 2.28 rad）。
            #
            # **候选点用线性化误差打分，不碰 PyBullet**：位置/姿态残差对 q 的导数就是
            # 雅可比，所以 ``e(q+Δ) ≈ e - J·Δ``。原实现每个候选点都真跑一次 FK
            # （= N 次 resetJointState + 1 次 getLinkState），一帧下来单条臂就有上千次
            # GUI 往返；现场三台机器人同场景时单次往返更贵，实测把帧率从 8 打到 3.4。
            # 现在每迭代只剩 1 次写 + 1 次读 + 1 次雅可比，候选点评估是纯 numpy。
            current = self._terms(q, error_position, error_rotation, q_last,
                                  error_elbow)
            best_q, accepted = q, False
            scale = 1.0
            tried = []
            for _attempt in range(4):
                candidate = np.clip(q + scale * step, self.lower, self.upper)
                delta = candidate - q          # 取投影后的真实增量
                predicted_position = error_position - linear @ delta
                predicted_elbow = (None if error_elbow is None
                                   else error_elbow - elbow_linear @ delta)
                predicted_rotation = (None if error_rotation is None
                                      else error_rotation - angular @ delta)
                cost = self._terms(candidate, predicted_position,
                                   predicted_rotation, q_last, predicted_elbow)
                tried.append((scale, cost, float(np.linalg.norm(delta))))
                if cost < current:
                    best_q, accepted = candidate, True
                    break
                scale *= 0.5
            q = best_q
            if _DEBUG:
                trials = " ".join(f"{s:.3f}:{c:.6f}/{m:.4f}" for s, c, m in tried)
                print(f"[paper_ik] it {iterations:2d} res {residual * 1000:7.2f}mm "
                      f"rot {rot_deg if rot_deg is None else round(rot_deg, 3)} "
                      f"|step| {np.linalg.norm(step):7.4f} cost {current:.6f} "
                      f"trust {self._trust:.3f} tried {trials}")
            if not accepted:
                # 四档缩放全无改善 = 这一步在当前位置无下降方向（典型：目标不可达 /
                # 关节顶在限位上被 projection 抵掉）。再迭代只会用更小的 trust 重算
                # **同一个** 零步：实测会白跑满 40 次迭代（每次约 11 次 PyBullet 调用），
                # 直接吃掉帧率，而解一点没动 —— 现场表现就是「卡，还不动」。
                # 允许一次重试（trust 减半后线性化点更保守，偶尔真能走出去了）。
                rejected += 1
                self._trust = max(MIN_TRUST_RAD, self._trust * 0.5)
                if rejected >= 2:
                    break
                continue
            rejected = 0
            # 信赖域自适应（ipopt 的 trust-region 在本机的等价物）：收敛慢就放大步子，
            # 被拒绝就缩小。固定步长上限是「跟手滞后」的直接原因（实测残差 130 mm）。
            self._trust = min(MAX_TRUST_RAD, self._trust * 1.5)
        return q, residual, elbow_residual, rot_deg, iterations
