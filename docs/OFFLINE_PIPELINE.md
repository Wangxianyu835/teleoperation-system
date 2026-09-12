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

---

# 8. 队友的重定向输出已接入（★ 目标手是 l21）

> 本节记录 2026-09-12 收到队友数据后的分析与映射推导，**队友A 必须确认**。

## 8.1 收到的数据

```
datasets/raw/retarget_twohand_153542.h5      ← 项目内路径（原始文件在微信目录，已复制）
  left_angles / right_angles  (557, 18)  float32
  left_valid  / right_valid   (557,)     bool     ← 无效帧 = 全零帧
  timestamps                  (557,)     float64  0.174 ~ 18.99 s（30.3 FPS）
  attrs: checkpoint=mytrans/.../linker/none_warmstart/model_best.pth
         input_file=visual_hand_data_20260912_153542.h5
         output_shape=[18]
```

**数据质量**：有效帧 512/557（91.9%），**有效范围 = 第 25 ~ 536 帧**（首尾各约 25 帧无效）。

## 8.2 ★ 关键结论：目标手是 **`l21`**，不是 `l7`

用「每个维度的实测范围」与「每个关节的限位」做**二分图最优匹配**，遍历全部 15 个型号：

| 型号 | 关节数 | 可行匹配 | 越界 |
|---|---|---|---|
| **l21** | **17** | **17/18** | **1** ← **最吻合** |
| g20 / l25 | 21 | 17/18 | 1 |
| l7 | 17 | 11/18 | 7 |

**决定性证据**：
```
dim 13 [-0.5997, -0.5875] ↔ thumb_cmc_roll  (±0.600)  精确到 0.0003
dim 14 [+0.0067, +1.5953] ↔ thumb_cmc_yaw   (0~1.600) 精确到 0.002
```

> 🔴 **`hand_interface.py` / `show_hand.py` 用的是 `l7`（无 `*_mcp_roll`），
> 而队友模型的目标是 `l21`（多 4 个侧摆关节）。两边必须统一！**

## 8.3 映射表（18 维 → l21 的 17 关节）

```
dim 0                → 占位（恒 0）
dim 1,2,3            → index:  mcp_roll, mcp_pitch, pip
dim 4,5,6            → middle: mcp_roll, mcp_pitch, pip
dim 7,8,9            → ring:   mcp_roll, mcp_pitch, pip
dim 10,11,12         → pinky:  mcp_roll, mcp_pitch, pip
dim 13,14,15,16,17   → thumb:  cmc_roll, cmc_yaw, cmc_pitch, mcp, ip
```
**顺序 = l21 URDF 顺序**，交叉验证：加载 l21 URDF 后「可驱动的动作维度 17/17」全部命中。

## 8.4 回放脚本：`scripts/replay_hand_angles.py`

```powershell
# 静态检查（维度/范围/越界/跳变）
python scripts/replay_hand_angles.py --check --hand right

# GUI 可视化（能看见手指动）
python scripts/replay_hand_angles.py --hand right --render
python scripts/replay_hand_angles.py --hand both --render --smooth 5

# 无头回放（自动验证）
python scripts/replay_hand_angles.py --hand both --headless-replay
```

## 8.5 给队友A 的反馈

| # | 反馈 | 建议 |
|---|---|---|
| 1 | ✅ **有效性标记设计规范** | `*_valid` 与全零帧完全对应，很好 |
| 2 | ✅ **目标手已识别为 l21** | 请确认这个理解是否正确 |
| 3 | ⚠️ **有 34/511 帧跳变 > 0.3 rad（最大 83.8°）** | 建议按论文加**卡尔曼滤波**平滑 |
| 4 | ⚠️ **首尾各约 25 帧全零** | 是检测失败还是手张开？请在导出时说明 |
| 5 | ❓ **18 维的顺序定义** | 我按「范围 vs 限位」推导如上，**请确认** |
| 6 | ❓ **只有手指、没有手臂** | 契约H 需要完整 `action`（含双臂各 7），后续如何补？ |
| 7 | ❓ **dim 0 恒 0** | 是占位、手腕、还是没用上？ |

## 8.6 下一步建议

**短期（可立即做）**：
```powershell
# 让重定向的师兄跑一下 GUI 回放，肉眼确认动作是否合理
python scripts/replay_hand_angles.py --hand both --render
```

**中期**：写一个「适配器」把 18 维 → 目标机器人的完整 `action`（38/36/28 维），
再走 `replay_actions.py` 灌进仿真环境。

## 8.7 ✅ 视觉验证通过（2026-09-12）

**用 `--render` 打开 GUI 窗口肉眼观察 `l21_right` 回放，结果与预测完全一致：**

| 部位 | 预测（基于数据统计）| 实际观察 | 结论 |
|---|---|---|---|
| 四根手指 | 反复「握紧→张开」，幅度接近 90° | ✅ 一致 | 正确 |
| 四指侧摆 | 几乎不动（±0.18 rad）| ✅ 一致 | 正确 |
| 大拇指 | 基本固定姿势，轻微动作 | ✅ 一致 | 正确 |
| 弯曲方向 | 向掌心 | ✅ 一致 | 正确 |

**这一次验证同时坐实了 4 件事：**
1. ✅ 映射推导 **100% 正确**（「维度范围 vs 关节限位」二分图匹配的结果是对的）
2. ✅ 目标手型号确实是 **`l21`**
3. ✅ **队友的重定向算法输出有效**（能驱动真实 l21 做合理抓握）
4. ✅ 回放脚本端到端工作正常

> 📌 **手部链路已可作为后续工作的可靠基础。**

**复现方式**：
```powershell
cd F:\simulation_platform
$env:PYTHONPATH = 'F:\simulation_platform\lib'
python scripts/replay_hand_angles.py --hand right --render
```

## 8.8 更新后的问题清单（给队友A）

> ✅ = 已通过视觉验证 / ❓ = 仍需你确认

| # | 问题 | 状态 |
|---|---|---|
| 1 | 目标手是 LinkerHand **l21**（17 关节）吗？ | ✅ 视觉验证一致（仍欢迎你确认）|
| 2 | 18 维顺序 = `dim0占位 + 4指×3 + 拇指×5`？ | ✅ 视觉验证一致 |
| 3 | 跳变 34/511 帧（最大 83.8°）要不要加卡尔曼滤波？ | ❓ |
| 4 | 首尾各约 25 帧全零是「检测失败」还是「手张开」？ | ❓ |
| 5 | **手臂部分**怎么补？（契约H 需要含双臂各 7 关节）| ❓ **最关键** |
| 6 | dim 0 恒 0 是什么？ | ❓ |
| 7 | 项目里手接口是 **l7**（无 `*_mcp_roll`），你目标是 **l21**，要不要统一？ | ❓ |
