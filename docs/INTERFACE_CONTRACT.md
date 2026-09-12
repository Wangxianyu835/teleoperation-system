# 接口契约（INTERFACE CONTRACT）

> **本文件是团队协作的「技术合同」** —— 三人各自开发时必须遵守这里的定义。
> 任何修改都需要三人确认后同步更新本文件。
>
> **最后更新**：2026-09-12　|　**状态**：v1 初稿（待三人评审冻结）

---

## 0. 一句话总览

```
┌───────────────┐   action (np.ndarray)    ┌──────────────────┐   obs (dict)   ┌──────────────┐
│ 重定向算法     │ ───────────────────────> │  SimulationEnv    │ ─────────────> │ 数据采集      │
│ (队友 A)      │                          │  (王宪雨)         │                │ (队友 B)     │
│               │ <─────────────────────── │                  │                │              │
└───────────────┘        obs (dict)        └──────────────────┘                └──────────────┘
                                                     │
                                               写入 HDF5
```

**契约有三条：A 动作空间 / B 观测空间 / C 控制器接口。** 下面逐条定义。

---

## 契约 A：动作空间（重定向算法的**输出**）

### A.1 核心规则

动作是 **一维 float 数组**，按**关节名顺序**排列：

```
action = [ 左臂 7 个 ] + [ 右臂 7 个 ] + [ 左手 N_L 个 ] + [ 右手 N_R 个 ]
          索引 0~6       索引 7~13      索引 14 ~ 14+N_L-1    其余
```

### A.2 ⚠️ 维度**不是固定的**，由机器人决定

| robot_type | 机器人 | 左臂 | 右臂 | 左手 | 右手 | **动作维度** |
|---|---|---|---|---|---|---|
| `h1_2`   | Unitree H1-2   | 7 | 7 | **12** | **12** | **38** |
| `gr1_t2` | Fourier GR1-T2 | 7 | 7 | **11** | **11** | **36** |
| `g1`     | Unitree G1     | 7 | 7 | **7**  | **7**  | **28** |

**获取方式（不要硬编码！）：**
```python
dim = env.action_dim                     # 推荐
dim = len(loader.action_joint_indices)   # 等价
```

### A.3 🔴 最重要的一条铁律

> **绝不能假设「动作向量第 i 个元素 → 关节索引 i」！**

**原因**：URDF 里前 12 个关节是**腿部**：
```
索引 0~5   : left_hip_yaw / left_hip_pitch / left_hip_roll / left_knee / left_ankle_pitch / left_ankle_roll
索引 6~11  : right_hip_* / right_knee / right_ankle_*
索引 12    : torso_joint 或 waist_*  （躯干）
索引 13+   : 才是手臂（left_shoulder_pitch_joint 开始）
```

**如果按索引直接映射** → `action[0]`（本意左肩）会被送到 `left_hip_yaw_joint`（左髋＝腿）
→ **表现为「手臂不动、腿乱动」**（这是本项目修过的一个真实 bug）

**正确写法：必须使用映射表**
```python
# 仿真环境内部已经这样做（envs/simulation_env.py 的 step()）
self.task.apply_action(action, joint_indices=env.action_joint_indices)
```

### A.4 实测映射表（2026-09-12）

| robot_type | `action[0]` 对应的关节索引 | 关节名 |
|---|---|---|
| h1_2   | **13** | `left_shoulder_pitch_joint` |
| gr1_t2 | **16** | `left_shoulder_pitch_joint` |
| g1     | **22** | `left_shoulder_pitch_joint` |

> 完整映射表可在运行期打印：
> ```python
> loader.describe_action_space()
> ```

### A.5 手臂关节顺序（三种机器人**完全一致**的语义顺序）

```
action[0] shoulder_pitch
action[1] shoulder_roll
action[2] shoulder_yaw
action[3] elbow            （H1-2/GR1-T2 叫 elbow_pitch_joint，G1 叫 elbow_joint）
action[4] wrist_a          （各机器人具体轴序略有差异，见 action_joint_names）
action[5] wrist_b
action[6] wrist_c
```
> ⚠️ 后 3 个腕关节各机器人的轴序不完全相同，**必须用 `env.action_joint_names` 取实际名字**，不要假设。

### A.6 角度单位与范围

| 项目 | 规定 |
|---|---|
| 单位 | **弧度（rad）** |
| 控制模式 | PyBullet `POSITION_CONTROL` |
| 力矩上限 | 默认 `force=100`（可在 `apply_action(..., force=)` 调整）|

---

## 契约 B：观测空间（`SimulationEnv.step()` 的**输出**）

### B.1 返回值签名

```python
obs, success, done, elapsed = env.step(action)
```

| 返回 | 类型 | 含义 |
|---|---|---|
| `obs` | `dict` | 观测（见 B.2）|
| `success` | `bool` | 本步是否完成任务（由任务的 `check_success()` 判定）|
| `done` | `bool` | episode 是否结束（成功 **或** 超时）|
| `elapsed` | `float` | 距 episode 开始的秒数 |

### B.2 obs 的键（当前实现）

```python
{
  'task_name':  str,          # 如 'pushcube'
  'robot_type': str,          # 'h1_2' / 'gr1_t2' / 'g1'
  'table_pos':  (3,),  'table_orn':  (4,),   # 每个任务物体一对
  'cube_pos':   (3,),  'cube_orn':   (4,),   # 键名 = 任务里注册的物体名
  'target_pos': (3,),  'target_orn': (4,),
}
```

### B.3 ⚠️ 与论文的差距（**待补齐 —— 队友 B 的 P0 任务**）

论文要求三类观测：

| 论文要求 | 当前状态 | 负责 |
|---|---|---|
| (a) 机器人状态向量（关节位置/角度/速度）| ⚠️ **不在 obs 里**（只在 HDF5 里记录）| 队友 B 补进 obs |
| (b) 相机流（头戴第一人称 + 固定第三人称）| ⚠️ 只在 HDF5 里记首尾帧，第一人称 eye 是硬编码 | 队友 B |
| (c) 物体位姿元数据 | ✅ 已有 | — |

---

## 契约 C：控制器接口（重定向算法**接入**仿真的方式）

### C.1 签名

```python
controller: Callable[[dict], np.ndarray]
   输入: obs（契约 B 的 dict）
   输出: action（契约 A 的 np.ndarray，长度 = env.action_dim）
```

### C.2 使用方式

```python
from envs import SimulationEnv

env = SimulationEnv(robot_type='h1_2', task_name='pushcube',
                    render=False, record=True)

def my_controller(obs):
    # ← 队友 A 的重定向算法在这里
    action = retarget(obs)
    assert len(action) == env.action_dim
    return action

env.run_episode(controller=my_controller, max_steps=10000)
```

### C.3 开发建议（可并行）

> **队友 A 不需要等仿真环境完全就绪** —— 先用假数据写单元测试：
> ```python
> action = retarget(fake_input)          # 假输入
> assert action.shape == (38,)           # H1-2
> assert np.all(np.isfinite(action))
> ```

---

## 契约 D：数据格式（HDF5，队友 B 负责）

### D.1 文件命名

```
data/ep_0001.h5, data/ep_0002.h5, ...
```

### D.2 结构

```
ep_0001.h5
├── joint_positions          (T, num_joints)  float64
├── joint_velocities         (T, num_joints)  float64
├── timestamps               (T,)             float64
├── object_trajectories/                (HDF5 group)
│   ├── cube/
│   │   ├── positions        (T, 3)
│   │   └── orientations     (T, 4)
│   └── target/...
├── camera_rgb/                         (HDF5 group)
│   ├── first_rgb                         ⚠️ 待改为逐帧 (T, H, W, 3)
│   └── last_rgb
├── head_camera_rgb/
│   ├── first_rgb                         ⚠️ 待改为逐帧
│   └── last_rgb
└── attrs:
    ├── episode_id           int
    ├── success              bool
    ├── completion_time      float
    └── num_steps            int
```

### D.3 ⚠️ 已知待改（队友 B 的 P0）

1. **相机改为逐帧存储**（现在只存首尾帧，无法用于模仿学习）
2. **第一人称相机跟随机器人头部**（现在 `eye=[0,0,1.5]` 是硬编码）
3. 增加 `action` 序列的记录（论文要求，便于模仿学习）
4. 增加 `camera` 内参/外参

---

## 契约 E：命名与目录约定

| 项目 | 约定 |
|---|---|
| 机器人标识 | `h1_2` / `gr1_t2` / `g1`（小写下划线）|
| 任务名 | 全小写无空格，如 `pushcube` / `ballbimanual` |
| 分支命名 | `dev/simulation`（王宪雨）/ `alg/retarget-*`（A）/ `data/recorder-*`（B）|
| 提交信息 | `feat:` / `fix:` / `docs:` / `refactor:` / `test:` + 空格 + 说明 |
| 中文 | 允许（已实测 UTF-8 提交信息正常）|

---

## 契约 F：变更流程

1. 任何**接口变更**（动作维度、obs 键名、HDF5 结构）必须先在本文件更新
2. 在群里 @ 三人确认
3. 更新后同步 `PROJECT_CONTEXT.md`
4. 提交信息用 `docs: 更新接口契约 - xxx`

> 🔴 **冻结后的契约是「合同」** —— 一个人擅自改会导致另外两人的代码全部失效。

---

## 契约 G：人类手部数据文件（队友B → 队友A）

> 📄 **完整定义见 [`OFFLINE_PIPELINE.md`](OFFLINE_PIPELINE.md) 第 2 节**

**格式：`.h5` 或 `.npz`**

```
human_hand.h5
├── keypoints_3d      (T, 21, 3)  float32   ⭐ MediaPipe 21 关键点世界坐标(米)
├── keypoints_2d      (T, 21, 2)  float32   可选：像素坐标（调试用）
├── wrist_pose        (T, 7)      float32   可选：[x,y,z,qx,qy,qz,qw]（阶段2）
├── timestamps        (T,)        float64   每帧时间戳(秒)
└── attrs: hand_side / fps / source / mediapipe_version
```

**三条硬性约定**：
1. 坐标系：**右手系、Z 轴向上、单位米**
2. 21 个关键点顺序**必须遵循 MediaPipe 官方 `HAND_21_LANDMARKS`**，不可自定义
3. 检测失败的帧**用 NaN 填充**，不要丢帧（否则 timestamps 错位）

**参考实现**：`python scripts/make_sample_data.py --kind hand`
→ 生成 `datasets/samples/human_hand_demo_right.h5` 供格式对照

---

## 契约 H：动作序列文件（队友A → 仿真平台）

> 📄 **完整定义见 [`OFFLINE_PIPELINE.md`](OFFLINE_PIPELINE.md) 第 3 节**

**格式：`.h5` 或 `.npz`**

```
actions.h5
├── actions      (T, action_dim)  float32   ⭐ 契约A 定义的动作向量序列
├── timestamps   (T,)             float64   可选，缺省按 fps 生成
└── attrs: robot_type / task_name / fps / source / algo
```

**维度必须匹配**（回放脚本会自动校验并给出明确报错）：

| robot_type | action_dim |
|---|---|
| `h1_2` | **38** |
| `gr1_t2` | **36** |
| `g1` | **28** |

**参考实现**：
```powershell
# 生成示例文件
python scripts/replay_actions.py --dummy --save-actions datasets/samples/actions_demo_h1_2.h5
# 回放
python scripts/replay_actions.py --file datasets/samples/actions_demo_h1_2.h5
# 查看动作空间定义
python scripts/replay_actions.py --describe --robot h1_2
```

---

## 契约总览表

| 契约 | 内容 | 谁 → 谁 | 载体 |
|---|---|---|---|
| **A** | 动作空间（内存）| 队友A → SimulationEnv | `np.ndarray(action_dim)` |
| **B** | 观测空间（内存）| SimulationEnv → 所有人 | `dict` |
| **C** | 控制器接口 | 队友A → env.run_episode | `Callable[[dict], ndarray]` |
| **D** | HDF5 数据记录格式 | 队友B | `ep_xxxx.h5` |
| **E** | 命名与目录约定 | 全体 | — |
| **F** | 变更流程 | 全体 | — |
| **G** | **人类手部数据文件** | **队友B → 队友A** | `human_hand.h5` |
| **H** | **动作序列文件** | **队友A → 仿真平台** | `actions.h5` |
