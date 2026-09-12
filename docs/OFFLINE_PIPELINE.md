# 离线 Vision-based 流水线方案（OFFLINE PIPELINE）

> **本文件是「离线测试」的技术规范** —— 定义数据格式与三方分工。
> 与 `INTERFACE_CONTRACT.md` 配套使用（那份定义**内存接口**，这份定义**文件接口**）。
>
> **最后更新**：2026-09-12　|　**状态**：v1 待三人评审

---

## 0. 为什么先做「离线」而不是实时

| 实时（论文做法）| **离线（本项目做法）** |
|---|---|
| 需要摄像头 + 低延迟（>30FPS）| ✅ 录好视频**慢慢处理** |
| 重定向必须跑在帧率以上 | ✅ **无性能压力** |
| 调试困难（数据流过即逝）| ✅ **数据落盘，可对同一片段反复调试** |
| 改算法就要重新采集 | ✅ **改算法不用重新采集** ← 最大优势 |

> 论文的评测协议本身就是**先录制、后评测**，所以离线**符合论文思路**，不是简化版。

---

## 1. 三方流水线

```
【阶段 1】采集           【阶段 2】重定向          【阶段 3】仿真回放
  队友 B                  队友 A                    王宪雨
─────────────────────────────────────────────────────────────────────
📹 录制视频               📂 读 human_hand.h5      📂 读 actions.h5
      ↓                        ↓                        ↓
🤖 MediaPipe              🔧 dex-retargeting       🎮 SimulationEnv
   手部 21 关键点             向量优化                 逐步 replay
      ↓                        ↓                        ↓
💾 human_hand.h5    →     💾 actions.h5      →   💾 episode_result.h5
   （契约 G）                （契约 H）                （回放结果）
```

---

## 2. 契约 G：人类手部数据（队友B 输出 → 队友A 输入）

**格式：`.h5` 或 `.npz`**

```
human_hand.h5
├── keypoints_3d        (T, 21, 3)  float32   ⭐ 核心：MediaPipe 21 关键点世界坐标(米)
├── keypoints_2d        (T, 21, 2)  float32   可选：原始像素坐标（调试可视化用）
├── wrist_pose          (T, 7)      float32   可选：手腕位姿 [x,y,z,qx,qy,qz,qw]（阶段2）
├── timestamps          (T,)        float64   每帧时间戳(秒)
└── attrs:
    ├── hand_side       str                   'left' / 'right'
    ├── fps             float                 采集帧率（如 30）
    ├── source          str                   来源视频文件名
    └── mediapipe_version str
```

### ⚠️ 三条硬性约定

1. **坐标系**：右手系、**Z 轴向上**、**单位米**
2. **21 个关键点顺序**必须遵循 **MediaPipe 官方 `HAND_21_LANDMARKS`** 定义，顺序**不可自定义**
3. **检测失败的帧用 NaN 填充**，不要丢帧（丢帧会导致时间戳/timestamps 错位）

> 💡 **参考实现**：运行 `python scripts/make_sample_data.py --kind hand`
> 会生成一个符合本契约的示例文件，可直接用文件结构对照。

---

## 3. 契约 H：动作序列（队友A 输出 → 仿真平台输入）

**这就是「重定向的输出」**，用于仿真回放。

```
actions.h5
├── actions       (T, action_dim)  float32   ⭐ 契约A 定义的动作向量序列
├── timestamps    (T,)             float64   可选，缺省按 fps 生成
└── attrs:
    ├── robot_type  str   必需：'h1_2' | 'gr1_t2' | 'g1'（决定 action_dim）
    ├── task_name   str   必需：'pushcube' 等 30 个任务之一
    ├── fps         float 默认 240（与仿真步长一致）
    ├── source      str   来源的人类数据文件名（便于溯源）
    └── algo        str   算法标识，如 'dex-retargeting+mediapipe'
```

### ⚠️ 维度校验

`actions.shape[1]` **必须等于** `robot_type` 对应的动作维度：

| robot_type | action_dim | 左臂 | 右臂 | 左手 | 右手 |
|---|---|---|---|---|---|
| `h1_2` | **38** | 7 | 7 | 12 | 12 |
| `gr1_t2` | **36** | 7 | 7 | 11 | 11 |
| `g1` | **28** | 7 | 7 | 7 | 7 |

> 回放脚本会自动校验，维度不符会**明确报错并提示期望值**。

---

## 4. 分阶段实施

### ✅ 阶段 1：只做手部（最小闭环，建议先做）

| 谁 | 任务 | 输出 |
|---|---|---|
| B | MediaPipe 采集手部 21 关键点 | `human_hand.h5`（契约 G）|
| A | dex-retargeting：关键点 → **手部关节角** | 只需输出**手部部分**的动作 |
| 你 | 回放：只驱动手部关节，看能否抓握 | 可视化 / 指标 |

**为什么先做手部**：手臂需要 手腕6DoF → IK(PINK) → 7 关节，链路长、易出错。
**先打通「术」（手部），再加「臂部」。**

### 🔜 阶段 2：加上手臂

- B 增加采集 `wrist_pose`（手腕 6DoF）
- A 增加：手腕位姿 → IK → 7 个手臂关节
- 输出完整 `action_dim`

### 🎯 阶段 3：对齐论文

- 用 **SMPLer-X** 替代 MediaPipe 做手腕位姿（论文做法）
- 用 **PINK** 做 IK
- 手部加**卡尔曼滤波**平滑

---

## 5. 数据交换方式

| 数据类型 | 典型大小 | 交换方式 |
|---|---|---|
| `human_hand.h5` | < 10 MB | ✅ **git commit 到 `datasets/`** |
| `actions.h5` | < 5 MB | ✅ **git commit** |
| 原始视频 `.mp4` | 几十~几百 MB | ⚠️ 网盘/移动硬盘（**不进 Git**）|

> 💡 **用 Git 交换的好处**：有版本记录，能追溯「这个结果是用哪版数据跑出来的」。

### 分支与提交约定

```powershell
git checkout -b data/vision-record-xxx     # 队友B
git add datasets/ && git commit -m "data: 采集 xxx 手部数据（契约G）"
git push -u origin data/vision-record-xxx
# 然后开 PR 让队友A 评审
```

---

## 6. 执行顺序（行动清单）

| # | 步骤 | 负责 | 状态 |
|---|---|---|---|
| 1 | **三人确认契约 G / H 格式** | 全体 | ⬜ |
| 2 | 采集一小段（3~5 秒）手部数据 | B | ⬜ |
| 3 | 跑 dex-retargeting，输出 `actions.h5` | A | ⬜ |
| 4 | 回放脚本灌进仿真，出结果 | 你 | ✅ **脚本已就绪** |
| 5 | 三人看结果，迭代 | 全体 | ⬜ |

---

## 7. 回放脚本用法（已实现）

```powershell
# ① 先用假数据验证链路（不需要真实数据）
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render

# ② 顺便导出一份「契约H 示例文件」，给队友A 照着写
python scripts/replay_actions.py --dummy --save-actions datasets/samples/actions_demo_h1_2.h5

# ③ 回放真实数据
python scripts/replay_actions.py --file datasets/actions/pushcube_h1_2.h5

# ④ 带可视化（会弹 pybullet 窗口）
python scripts/replay_actions.py --file xxx.h5 --render

# ⑤ 查看某机器人的动作空间定义
python scripts/replay_actions.py --describe --robot h1_2
```
