"""关节角的常速度卡尔曼滤波，用于实时命令和离线序列。

每维独立维护角度与角速度。process_noise控制跟随速度，measurement_noise
控制观测平滑程度。创新门控gate默认关闭；快速连续动作不应被当作孤立尖峰。
在线使用reset(n_dims)和step(z)，离线使用filter_sequence(seq)。
"""
import numpy as np


class KalmanSmoother:
    """每维独立的常速度卡尔曼滤波（含创新门控）

    Args:
        dt: 采样间隔（秒）。30 FPS 用 1/30。
        process_noise: 过程噪声谱密度 q（默认 1.0）
        measurement_noise: 观测噪声方差 r（默认 0.01）
        gate: 创新门控倍数（默认 0.0；<=0 表示关闭门控，即纯卡尔曼）
        max_velocity: 角速度限幅（rad/s），防止滤波后速度爆炸（默认 20）
    """

    def __init__(self, dt=1.0 / 30.0, process_noise=1.0,
                 measurement_noise=0.01, gate=0.0, max_velocity=20.0):
        self.dt = float(dt)
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.gate = float(gate)
        self.max_velocity = float(max_velocity)

        self.n_dims = 0
        self.x = None          # (D, 2)  状态 [θ, ω]
        self.P = None          # (D, 2, 2) 协方差

    # ------------------------------------------------------------------
    def _F(self):
        dt = self.dt
        return np.array([[1.0, dt], [0.0, 1.0]])

    def _Q(self):
        dt, q = self.dt, self.q
        return q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                             [dt ** 2 / 2.0, dt]])

    # ------------------------------------------------------------------
    def reset(self, n_dims):
        """重置滤波器（开始一个新的序列时调用）"""
        self.n_dims = int(n_dims)
        self.x = np.zeros((self.n_dims, 2))
        self.P = np.tile(np.eye(2) * 1e3, (self.n_dims, 1, 1))
        self.initialized = False

    # ------------------------------------------------------------------
    def step(self, z):
        """推进一帧

        Args:
            z: (D,) 观测角度；NaN 表示该帧无效（只做预测，不做更新）

        Returns:
            (D,) 滤波后的角度
        """
        z = np.asarray(z, dtype=np.float64)
        if self.x is None or len(z) != self.n_dims:
            self.reset(len(z))

        # ---------- 首次：直接用观测初始化 ----------
        if not self.initialized:
            first = np.nan_to_num(z, nan=0.0)
            self.x[:, 0] = first
            self.x[:, 1] = 0.0
            self.initialized = True
            return self.x[:, 0].copy()

        F = self._F()
        Q = self._Q()

        # ---------- 预测 ----------
        x_pred = self.x @ F.T                              # (D, 2)
        P_pred = F @ self.P @ F.T + Q                      # (D, 2, 2)
        # 角速度限幅
        x_pred[:, 1] = np.clip(x_pred[:, 1],
                               -self.max_velocity, self.max_velocity)

        # ---------- 更新（逐维） ----------
        S = P_pred[:, 0, 0] + self.r                       # (D,) 创新方差
        innov = z - x_pred[:, 0]                           # (D,) 残差
        valid = ~np.isnan(innov)

        # 创新门控：残差过大的帧判为离群点，跳过更新（只用预测）
        if self.gate > 0:
            outlier = np.abs(innov) > self.gate * np.sqrt(S)
        else:
            outlier = np.zeros_like(valid)
        do_update = valid & (~outlier)

        K = np.zeros((self.n_dims, 2))
        K[:, 0] = P_pred[:, 0, 0] / S                      # 只观测角度
        K[:, 1] = P_pred[:, 1, 0] / S

        m = do_update
        self.x = x_pred.copy()
        self.x[m, 0] = x_pred[m, 0] + K[m, 0] * innov[m]
        self.x[m, 1] = x_pred[m, 1] + K[m, 1] * innov[m]

        # 协方差更新：P = (I - K H) P_pred（仅对做了更新的维度）
        I_KH = np.tile(np.eye(2), (self.n_dims, 1, 1)).astype(np.float64)
        I_KH[m, 0, 0] = 1.0 - K[m, 0]
        I_KH[m, 0, 1] = 0.0
        I_KH[m, 1, 0] = -K[m, 1]
        I_KH[m, 1, 1] = 1.0
        self.P = I_KH @ P_pred
        self.P[~m] = P_pred[~m]                            # 未更新的维度保持预测
        # 数值稳定：保持对称正定
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

        return self.x[:, 0].copy()

    # ------------------------------------------------------------------
    def filter_sequence(self, seq, dt=None, return_stats=False):
        """对整段序列滤波

        Args:
            seq: (T, D) 观测序列（可含 NaN）
            dt:  覆盖构造时的 dt（通常用数据真实帧间隔）
            return_stats: 是否返回统计（离群点数量等）

        Returns:
            (T, D) 滤波结果；若 return_stats 则返回 (result, stats)
        """
        seq = np.asarray(seq, dtype=np.float64)
        if seq.ndim != 2:
            raise ValueError(f'seq 应为二维 (T, D)，实际 {seq.shape}')
        if dt is not None:
            self.dt = float(dt)

        self.reset(seq.shape[1])
        out = np.empty_like(seq)
        n_outlier = 0
        n_nan = 0
        for i in range(seq.shape[0]):
            z = seq[i]
            if np.any(np.isnan(z)):
                n_nan += 1
            # 统计离群点（与 step 内的门控逻辑一致）
            if (self.gate > 0 and self.initialized and self.x is not None):
                dt_ = self.dt
                pred = self.x[:, 0] + self.x[:, 1] * dt_
                s = np.sqrt(self.P[:, 0, 0] + self.P[:, 0, 1] * dt_
                            + self.P[:, 0, 0] + self.r)
                n_outlier += int(np.sum(
                    np.abs(z - pred) > self.gate * s))
            out[i] = self.step(z)

        if return_stats:
            return out, {'n_outlier': n_outlier, 'n_nan': n_nan}
        return out
