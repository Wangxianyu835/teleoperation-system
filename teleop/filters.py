"""信号平滑滤波 —— 卡尔曼滤波

论文 TeleOpBench §3.2.1 提到：
    "Finally, we implement a Kalman filter to smooth the robot's DoF,
     significantly reducing jitter-related instability in motion execution."

本模块就是这句话的实现，用于压掉重定向输出里的**单帧跳变**。

设计要点
--------
1. **每维独立**的常速度（CV）卡尔曼滤波：
      状态  x = [角度 θ, 角速度 ω]^T
      预测  x <- F x,  F = [[1, dt], [0, 1]]
      观测  z = H x + v,  H = [1, 0]
2. **连续白噪声加速度模型** 的 Q 矩阵：
      Q = q · [[dt^3/3, dt^2/2],
               [dt^2/2, dt   ]]
3. **创新门控（innovation gating）**：若某帧的残差超过 gate 倍标准差，
   判定为离群点（outlier），**跳过该帧的测量更新**、只用预测值 ——
   这是应对"单帧突然抽搐"的关键（34/511 帧存在 >0.3 rad 的跳变，最大 83.8°）。

用法
----
    from teleop.filters import KalmanSmoother

    # ① 序列模式（离线回放，最常用）
    kf = KalmanSmoother(dt=1/30.0)
    smoothed = kf.filter_sequence(angles)        # (T, D) -> (T, D)

    # ② 在线模式（实时遥操作用）
    kf.reset(n_dims)
    for z in angles:
        s = kf.step(z)

参数调优
--------
    process_noise (q)     越大 -> 越"信任数据"、跟随快、但平滑弱
    measurement_noise (r) 越大 -> 越"信任模型"、平滑强、但有滞后
    gate                  创新门控倍数；**默认 0（关闭）**

注意 关于默认 gate=0（重要，来自实测经验）
----------------------------------------
    2026-09-12 用队友真实数据实测发现：把 gate 设为 3 时，
    512 帧里有 **436 帧**被判为离群点 —— 因为这份数据里存在
    **成片的真实快速运动**（33/34 个"跳变"其实是真实动作，
    集中在某 9 秒内），门控会把合法快动作误杀，导致：
        max_delta 从 1.46 恶化到 5.22
    因此**默认关闭门控**。仅当你确认数据里有孤立尖峰
    （而非成片快动作）时才开启，并用 `classify_jumps()` 先诊断。

实测调参参考（队友数据，512 帧，30 FPS）
----------------------------------------
    q=1   r=0.1   -> maxd 0.256  jumps 0   accel 11.46  err 0.126（平滑强、偏差大）
    q=1   r=0.01  -> maxd 0.422  jumps 21  accel 18.95  err 0.087（<-- 推荐默认）
    q=1000 r=0.001-> maxd 1.407  jumps 34  accel 68.56  err 0.003（几乎无平滑）
"""
import numpy as np


class KalmanSmoother:
    """每维独立的常速度卡尔曼滤波（含创新门控）

    Args:
        dt: 采样间隔（秒）。30 FPS 用 1/30。
        process_noise: 过程噪声谱密度 q（默认 1.0）
        measurement_noise: 观测噪声方差 r（默认 0.01）
        gate: 创新门控倍数（默认 3.0；<=0 表示关闭门控，即纯卡尔曼）
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


# ----------------------------------------------------------------------
def classify_jumps(seq, threshold=0.3, neighbor_ratio=0.5):
    """诊断「超阈值的帧」是【孤立尖峰】还是【成片真实快速运动】

    ★ 这个诊断非常重要：
       - 若多为【孤立尖峰】 -> 是噪声，应该滤波
       - 若多为【成片快动作】-> 是真实运动，**不要滤波**（会破坏动作）

    Args:
        seq: (T, D) 角度序列
        threshold: 判定跳变的阈值（rad）
        neighbor_ratio: 相邻帧变化量低于 threshold*该比例时，视为「小」

    Returns:
        dict: {
            n_frames, n_jumps, n_isolated, n_sustained,
            jump_ratio, isolated_frames, sustained_frames,
            verdict: 'noise' | 'real_motion' | 'mixed' | 'clean',
            timeline: [(start, end, count), ...]  # 按 50 帧分段统计
        }
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] < 3:
        return {}

    step = np.abs(np.diff(seq, axis=0)).max(axis=1)
    jump_idx = np.where(step > threshold)[0]

    low = threshold * neighbor_ratio
    iso, sus = [], []
    for i in jump_idx:
        prev = step[i - 1] if i > 0 else 0.0
        nxt = step[i + 1] if i + 1 < len(step) else 0.0
        # 前值/后值绝对值也看：真尖峰会在下一帧"弹回去"
        col = int(np.argmax(np.abs(np.diff(seq, axis=0)[i])))
        v0, v1 = seq[i, col], seq[i + 1, col]
        v2 = seq[i + 2, col] if i + 2 < len(seq) else np.nan
        bounced = (not np.isnan(v2)) and (
            abs(v2 - v0) < abs(v1 - v0) * 0.5)
        if prev < low and nxt < low and bounced:
            iso.append(int(i))
        else:
            sus.append(int(i))

    if len(jump_idx) == 0:
        verdict = 'clean'
    elif len(sus) == 0:
        verdict = 'noise'
    elif len(iso) == 0:
        verdict = 'real_motion'
    else:
        verdict = 'mixed'

    timeline = []
    for s in range(0, len(step), 50):
        cnt = int(np.sum((jump_idx >= s) & (jump_idx < s + 50)))
        if cnt:
            timeline.append((s, min(s + 49, len(step) - 1), cnt))

    return {
        'n_frames': int(seq.shape[0]),
        'n_jumps': int(len(jump_idx)),
        'n_isolated': len(iso),
        'n_sustained': len(sus),
        'jump_ratio': float(len(jump_idx) / len(step)),
        'isolated_frames': iso,
        'sustained_frames': sus,
        'verdict': verdict,
        'timeline': timeline,
    }


def print_jump_report(info, title='跳变性质诊断'):
    """把 classify_jumps 的结果打印成报告"""
    if not info:
        print('（数据不足，无法诊断）')
        return
    print()
    print('=' * 92)
    print(title)
    print('=' * 92)
    print(f"  帧数 = {info['n_frames']}   超阈值帧 = {info['n_jumps']} "
          f"({info['jump_ratio']*100:.1f}%)")
    print(f"  ★ 孤立尖峰（疑似噪声）   = {info['n_isolated']}")
    print(f"  ★ 成片快速运动（真实动作）= {info['n_sustained']}")

    v = info['verdict']
    concl = {
        'clean': '没有跳变 —— 数据很干净，无需滤波',
        'noise': '以孤立尖峰为主 —— 是噪声，【建议】滤波',
        'real_motion': '以成片快动作为主 —— 是真实运动，【不要】滤波！',
        'mixed': '两者都有 —— 建议只对孤立尖峰做轻度处理',
    }[v]
    print(f'  -> 结论：{concl}')

    if info['timeline']:
        print()
        print('  跳变的分布（按 50 帧分段）：')
        for s, e, c in info['timeline']:
            print(f'    帧 {s:>4d}~{e:>4d}: {c:>3d}  {"#" * c}')

    if info['isolated_frames']:
        print()
        print(f"  孤立尖峰的帧号：{info['isolated_frames'][:30]}")
    print('=' * 92)


# ----------------------------------------------------------------------
def detect_bad_frames(seq, valid=None, near_zero=0.05, min_dims=10,
                      neighbor_max_dims=4):
    """检测「整帧塌零」坏帧，返回【绝对下标】

    ★ 2026-09-12 重写（重要，不要改回旧判据）
    ----------------------------------------
    旧判据是「当前帧相对前后帧均值，偏离 > 0.4 rad 的维度 >= 3 个」。
    用队友真实数据（557 帧）实测发现：**该判据在快速运动上必然误报** ——
        - 全序列「单帧最大变化」中位数只有 0.0304 rad（1.7 度）
        - 但真实快动作帧可达 1.48 rad（84.9 度）
        - 数据里成片的真实快动作会成批被判为坏帧
        - 决定性反例：432->433（84.9 度）没被命中，
          434->435（73.5 度）却被命中 —— 纯属偶然，不是异常检测
    结论：旧判据把「文档注释里描述的典型场景」当成了判据名，
    **实际上根本没有检测「塌零」**。

    新判据（「孤立塌陷」，必须同时满足）：
      1. 本帧「接近 0」（|v| < near_zero）的维度数 >= min_dims
      2. 前后最近的【有效】帧「接近 0」的维度数 <= neighbor_max_dims
    即「邻居正常，只有本段塌了」。真实快动作的邻居也不接近 0，因此不会误判。
    若塌零连续多帧，整段一并返回。

    Args:
        seq: (T, D) 角度序列 —— **必须是完整序列**，不要传 seq[valid]
        valid: (T,) bool 有效帧标记；None 表示全部有效
        near_zero: 「接近 0」阈值（rad）
        min_dims: 本帧至少多少个维度接近 0 才算「塌陷」
        neighbor_max_dims: 邻居最多多少个维度接近 0 才算「正常」

    Returns:
        list[int]: 坏帧的【绝对下标】（升序）

    注意（踩过的坑）：**必须传完整数组 + valid**。
    如果传 seq[valid]（压缩数组），返回的是【压缩下标】，
    一旦当成绝对下标去引用就会指向完全无关的帧 ——
    2026-09-12 就是这样把 left[522] 误报成了 left[497]。
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] < 3:
        return []

    T = seq.shape[0]
    ok = (np.ones(T, bool) if valid is None
          else np.asarray(valid).astype(bool))
    n_zero = np.sum(np.abs(seq) < near_zero, axis=1)
    collapsed = ok & (n_zero >= min_dims)

    def nearest_valid(j, step):
        while 0 <= j < T:
            if ok[j]:
                return j
            j += step
        return None

    bad = []
    i = 0
    while i < T:
        if not collapsed[i]:
            i += 1
            continue
        j = i
        while j + 1 < T and collapsed[j + 1]:
            j += 1
        p = nearest_valid(i - 1, -1)
        n = nearest_valid(j + 1, +1)
        if ((p is not None and n_zero[p] <= neighbor_max_dims)
                or (n is not None and n_zero[n] <= neighbor_max_dims)):
            bad.extend(range(i, j + 1))       # 整段塌陷
        i = j + 1
    return [int(k) for k in bad]


def repair_bad_frames(seq, bad_frames):
    """把坏帧用「前后帧线性插值」替换（只用于真正的塌零帧）

    Args:
        seq: (T, D)
        bad_frames: detect_bad_frames 返回的【绝对下标】列表

    Returns:
        (T, D) 修复后的序列（不修改原数组）

    注意：**不要**把它用在「快速运动」帧上 —— 插值会把真实快动作抹掉。
    只有 detect_bad_frames 判定的「孤立塌陷」才该修复。
    """
    seq = np.asarray(seq, dtype=np.float64).copy()
    T = seq.shape[0]
    for i in bad_frames:
        lo = i - 1
        hi = i + 1
        # 向两侧找最近的好帧
        while lo >= 0 and lo in bad_frames:
            lo -= 1
        while hi < T and hi in bad_frames:
            hi += 1
        if lo >= 0 and hi < T:
            w = (i - lo) / float(hi - lo)
            seq[i] = (1.0 - w) * seq[lo] + w * seq[hi]      # 线性插值
        elif lo >= 0:
            seq[i] = seq[lo]
        elif hi < T:
            seq[i] = seq[hi]
    return seq


def print_bad_frames_report(bad_frames, n_frames, title='坏帧检测'):
    """打印坏帧报告（bad_frames 为【绝对下标】）"""
    print()
    print('=' * 92)
    print(title)
    print('=' * 92)
    if not bad_frames:
        print(f'  共 {n_frames} 帧，未发现「整帧塌零」坏帧  [OK]')
        print('  说明：数据里成片的快速运动是【真实动作】，不算坏帧。')
    else:
        print(f'  共 {n_frames} 帧，发现 {len(bad_frames)} 个坏帧（整帧塌零）:')
        print(f'    【绝对下标】{bad_frames}')
        print('  [!] 建议：')
        print('     1) 回放端会用 repair_bad_frames() 插值修复（--no-repair 可关）')
        print('     2) 若能定位成因，请在导出侧修正（但需先核对原始数据）')
    print('=' * 92)


# ----------------------------------------------------------------------
def moving_average(seq, win):
    """简单滑动平均（对照用；边界采用 edge padding）

    Args:
        seq: (T, D)
        win: 窗口大小（<=1 时原样返回）
    """
    seq = np.asarray(seq, dtype=np.float64)
    if win <= 1:
        return seq
    win = win if win % 2 == 1 else win + 1
    pad = win // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(win) / win
    out = np.empty_like(seq)
    for j in range(seq.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode='valid')
    return out


# ----------------------------------------------------------------------
def motion_metrics(seq, jump_threshold=0.3, fps=None):
    """量化一段角度序列的「抖动程度」，用于对比滤波前后

    指标说明：
        max_delta   单帧最大变化量（rad）—— 越小越好
        n_jumps     单帧变化 > jump_threshold 的帧数 —— 越少越好
        rms_vel     角速度有效值（rad/s 或 rad/帧）—— 反映整体动得多快
        rms_accel   角加速度有效值 —— **核心指标：越小越平滑**
        noise_ratio rms_accel / rms_vel —— 归一化的抖动比例（跨序列可比）

    Args:
        seq: (T, D) 角度序列（可含 NaN）
        jump_threshold: 判定「跳变」的阈值（rad）
        fps: 若给出，速度/加速度换算成每秒单位
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] < 3:
        return {}

    scale = float(fps) if fps else 1.0

    delta = np.diff(seq, axis=0)                       # (T-1, D)
    accel = np.diff(delta, axis=0)                     # (T-2, D)

    # 每帧取「所有维度里最大的变化量」
    step_max = np.nanmax(np.abs(delta), axis=1)
    rms_vel = float(np.sqrt(np.nanmean(delta ** 2))) * scale
    rms_accel = float(np.sqrt(np.nanmean(accel ** 2))) * (scale ** 2)

    return {
        'max_delta': float(np.nanmax(step_max)),
        'n_jumps': int(np.nansum(step_max > jump_threshold)),
        'jump_ratio': float(np.nansum(step_max > jump_threshold)
                            / len(step_max)),
        'rms_vel': rms_vel,
        'rms_accel': rms_accel,
        'noise_ratio': (rms_accel / rms_vel) if rms_vel > 1e-9 else 0.0,
    }


# ----------------------------------------------------------------------
def compare_methods(seq, methods, jump_threshold=0.3, fps=None,
                    labels=None):
    """对比多种平滑方法，返回指标表（离线评估用）

    Args:
        seq: (T, D) 原始序列
        methods: dict {名称: 处理后的序列}
        labels: 可选的显示名映射

    Returns:
        list of dict（每行含各指标）
    """
    rows = [('原始 (raw)', motion_metrics(seq, jump_threshold, fps))]
    for name, out in methods.items():
        disp = labels.get(name, name) if labels else name
        rows.append((disp, motion_metrics(out, jump_threshold, fps)))
    return rows


def print_metrics_table(rows, title='平滑效果对比'):
    """把 compare_methods 的结果打印成表格"""
    keys = ['max_delta', 'n_jumps', 'jump_ratio', 'rms_vel',
            'rms_accel', 'noise_ratio']
    head = ['方法', '最大单帧变化', '跳变帧数', '跳变比例',
            'rms速度', 'rms加速度', '噪声比']
    print()
    print('=' * 96)
    print(title)
    print('=' * 96)
    print(f'{head[0]:<22s} {head[1]:>13s} {head[2]:>10s} {head[3]:>10s} '
          f'{head[4]:>10s} {head[5]:>11s} {head[6]:>10s}')
    print('-' * 96)
    for name, m in rows:
        if not m:
            continue
        print(f'{name:<22s} {m["max_delta"]:>13.4f} {m["n_jumps"]:>10d} '
              f'{m["jump_ratio"]*100:>9.2f}% {m["rms_vel"]:>10.4f} '
              f'{m["rms_accel"]:>11.4f} {m["noise_ratio"]:>10.4f}')
    print('=' * 96)
    print('说明：最大单帧变化 / 跳变帧数 越少越好；'
          'rms加速度 越小越平滑；噪声比 = rms加速度/rms速度')

