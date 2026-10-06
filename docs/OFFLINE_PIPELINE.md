# 离线 Vision-based 流水线方案（OFFLINE PIPELINE）

> **PR1 更新（2026-10-06）**：本文件保留早期系统离线规划与契约 G/H。当前正式 hand
> retargeting 已在本仓库统一，使用 Hand25、三帧、18D L21 输出与严格 alignment metadata。
> 当前完整架构、各消费协议和人工验收见 [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)。
> 当前 hand H5 schema 和 CLI 以 [HAND_RETARGETING.md](HAND_RETARGETING.md) 为准。
> 下文单手 `keypoints_3d` / 系统 `actions.h5` 不能直接等同于 canonical 两手输入/角度输出。

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

## 8.8 给队友A 的反馈清单（2026-09-12 第 2 次修订）

> ✅ = 已验证 / 🐛 = 需要你修 / ❓ = 需要确认 / ⏸️ = 暂缓
>
> **★ 重要更正**：原清单第 1 项（「`*_valid` 漏标 3 个坏帧」）**下标报错了，
> 结论也错了**。队友核对后指出 `left[497]` 范数 4.3940、数据正常 —— **他是对的**。

### ★ 更正说明：坏帧报告错在哪

| | 我原先的说法 | 更正后的事实 |
|---|---|---|
| 下标 | `left[497]`、`right[497, 510]` | 那是**压缩下标**（剔掉无效帧后的下标）；**绝对下标**是 `left[522]`、`right[522, 535]` |
| `522` | 坏帧 | ❌ **不是坏帧** —— 是手快速张开的过渡帧，我的判据误报 |
| `left` | 有 1 个坏帧 | ❌ **没有任何塌零帧** |
| `right[535]` | 没提过 | ⚠️ **唯一疑似真坏帧**，待用原始数据核对 |

**为什么会报错下标**：我把 `detect_bad_frames(arr[valid])` 的返回值
（压缩数组下标）直接当成绝对下标写进了报告。队友去查绝对下标 497/510，
数据当然是好的。**SHA256 已核对：双方看的是同一个文件**
（`65851D74...75A342`），不存在版本问题。

**为什么结论也错**：旧判据是「当前帧相对前后帧均值偏离 > 0.4 rad 的维度 >= 3 个」，
这在**快速运动上必然误报**。决定性反例：

```
left 全体单帧最大变化：中位数 0.0304 rad(1.7度)，最大 1.4823 rad(84.9度)

  432 -> 433 : 1.4823 rad (84.9度)   <-- 没被命中
  434 -> 435 : 1.2825 rad (73.5度)   <-- 被命中
```

同一量级、命中与否纯属偶然 —— 说明这个判据**不是异常检测**。
旧版把函数文档注释里描述的「典型场景」当成了判据名，**实际根本没检测「塌零」**。

### 已修复（本仓库，无需队友处理）

`teleop/filters.py` 的 `detect_bad_frames` 已**重写**：

- 新判据（「孤立塌陷」，须同时满足）：
  1. 本帧 `|v| < 0.05` 的维度数 >= 10
  2. 前后最近的**有效**帧该计数 <= 4

  即「邻居正常、只有本段塌了」—— 真实快动作的邻居也不接近 0，因此不会误判
- **必须传完整数组 + `valid`**，返回**绝对下标**（不再有下标混淆）
- 3 处调用点（`replay_hand_angles.py` / `replay_hand_on_robot.py` /
  `verify_hand_pipeline.py`）全部改为传完整数组
- `verify_hand_pipeline.py` 的验收项改成**检测器自检**：
  ① 注入一个塌零帧必须精确检出；② 真实快动作帧必须零误报

### 【请复核】1 项

| # | 内容 |
|---|---|
| 1 | ⚠️ **`right[535]`（绝对下标）疑似单帧异常**，请用原始 MediaPipe 数据核对 |

**证据**（`right_angles[535]`，`valid=True`）：

```
范数 0.8125，|v|<0.05 的维度数 = 10/18
前一帧 534 范数 2.1213（只有 4 维接近 0）   <-- 正常
后一帧 536 范数 2.6652（只有 4 维接近 0）   <-- 正常
534 -> 535 = 1.4546 rad (83.3 度)
535 -> 536 = 1.4634 rad (83.8 度)          <-- 掉下去又弹回（孤立单帧）

三个侧摆关节（限位 ±0.18）在一帧内翻到反侧：
  534: middle_roll=+0.1785  ring_roll=+0.1786  pinky_roll=+0.1790
  535: middle_roll=-0.1126  ring_roll=-0.1620  pinky_roll=-0.1027
  536: middle_roll=+0.1796  ring_roll=+0.1798  pinky_roll=+0.1798
```

- `frame_id = 535`，`t = 18.2880 s`（相对起点 +18.114 s）
- **83 度/帧 = 2500 度/秒**，人手指做不到；且侧摆从正限位瞬间翻到负侧无物理意义
- 若原始 MediaPipe 在该时刻正常 → 是**模型端**单帧失效；
  若原始数据本身就有问题 → 是**采集端**
- 提示：帧 537 起 `valid=False`（录制末尾），535 正好在这段之前

### 【需要你处理】1 项

| # | 内容 | 状态 |
|---|---|---|
| 2 | 🔧 **手型号统一到 LinkerHand L21** | ✅ 你已确认，本仓库**立即执行** |

**澄清一个误会**：你说的 `hand_interface.py` / `show_hand.py` 在**我们这份仿真仓库**
（`F:\simulation_platform`），不在 `mytrans` —— 所以**不是你要改的，我们这边改**。
同意你的方案：模型训练 / H5 定义 / 离线仿真 / GUI 回放**全部统一到 L21**；
若将来要用真实 L7，再单独实现明确的 `L21 -> L7` 降维映射。

### 【已确认】2 项（队友已答复，全部确认）

| # | 内容 | 结论 |
|---|---|---|
| 3 | 目标手 = LinkerHand **L21** | ✅ 已确认。<br>**更准确的表述（采用队友的）**：`L21 URDF = 17 个可动关节 + 1 个 hand_base_link 固定占位 + 5 个固定指尖节点 = 23 维`；模型输出 18 维，其中真正可动 17 维。<br>我们原先的推断依据：15 个型号二分图匹配中 l21 最吻合（零越界）|
| 4 | 18 维顺序 | ✅ **队友逐维确认，与我们的推断完全一致**，且确认了 dim13/dim14 的限位判断 |

**18 维顺序（队友确认版）**：

```
dim  0  hand_base_link     固定占位，恒 0

dim  1  index_mcp_roll      dim  4  middle_mcp_roll     dim  7  ring_mcp_roll      dim 10  pinky_mcp_roll
dim  2  index_mcp_pitch     dim  5  middle_mcp_pitch    dim  8  ring_mcp_pitch     dim 11  pinky_mcp_pitch
dim  3  index_pip           dim  6  middle_pip          dim  9  ring_pip           dim 12  pinky_pip

dim 13  thumb_cmc_roll      dim 14  thumb_cmc_yaw       dim 15  thumb_cmc_pitch
dim 16  thumb_mcp           dim 17  thumb_ip
```

**★ 新信息（我们原先不知道）**：做 **18 -> 23** 维转换时，末尾补的是
**五个固定指尖节点**：`index_tip, middle_tip, ring_tip, pinky_tip, thumb_tip`。
→ 这个转换在**仿真适配层**做即可，模型输出保持 18 维不变。

### 【待你提供】

| # | 内容 | 用途 |
|---|---|---|
| 5 | 原始采集数据 `visual_hand_data_20260912_153542.h5`（队友说在 `D:\2026\code\mytrans\input\`）| 核对上面第 1 项（`right[535]`），并区分责任在采集端还是模型端 |

### 【已自行查明，无需询问】

| 原问题 | 结论 |
|---|---|
| 跳变 34/511 要不要滤波？ | ✅ **不是噪声** —— 全部是**成片真实快速运动**（集中后 9 秒），孤立尖峰 = 0 → **不需要滤波** |
| `dim 0` 恒 0 是什么？ | ✅ 占位（队友确认 = `hand_base_link` 固定占位），对回放无影响 |
| 是否同一个文件？ | ✅ SHA256 `65851D74...75A342`，双方一致 |
| 18 -> 23 转换在哪做？ | ✅ 我们这边（仿真适配层），模型输出保持 18 维 |

### 【暂缓（因决定先只做手部）】

| # | 内容 |
|---|---|
| — | 手臂部分怎么补？（契约H 需含双臂各 7 关节）→ **待手部阶段完成后再议** |

---

## 8.9 ⚠️ 阶段限制声明（写给报告用）

**当前阶段 = 手部链路验证（hand-only）**，必须明确声明以下限制：

| 能做 ✅ | 不能做 ❌ |
|---|---|
| 证明「数据 → 手部模型」链路通 | **论文的 30 个任务**（`pushcube`/`pickcube`…）|
| 手部动作可视化与量化 | 「伸手 → 抓 → 放」完整流程 |
| 组件级验证 | 论文的「成功率 + 完成时间」任务级评测 |

> 📌 **必须写进报告**：论文的评测指标要求**手移动到物体处**，
> **缺少手臂数据则无法进行任务级评测** —— 本阶段完成手部链路验证，
> 任务级评测待手臂数据接入后进行。

### 与论文的核心差异（必须主动声明）

| 项 | 论文 | 本项目 |
|---|---|---|
| 重定向算法 | **dex-retargeting**（向量优化）| **自训练模型**（`mytrans` + `model_best.pth`）|
| 目标手 | Inspire / Fourier / Unitree 手 | **LinkerHand l21** |
| 仿真器 | **NVIDIA Isaac Sim**（PhysX + 照片级渲染）| **PyBullet** |
| 数据获取 | 实时遥操作 | **离线**（录制后处理）|

> 这些差异**本身不是缺点**，但**必须主动说明** —— 说清了是加分，被问出来是减分。

---

## 8.10 ✅ 方案B：l21 已装到机器人腕部（2026-09-12）

**状态：已完成并验证通过（6 / 6 组合精确对齐）。**

### 做了什么

新增 `scripts/replay_hand_on_robot.py`：画面里现在是
**「论文机器人（H1-2 / GR1-T2 / G1） + LinkerHand l21」**，
比孤立的一只手更接近真实场景，同时也**为阶段 2（加手臂）铺好了路** ——
手是用 pybullet 固定约束锁在腕部的，**将来驱动手臂时，手会自动跟着走**。

| 机器人 | 手基座 link（挂载点）|
|---|---|
| `h1_2` | `L_hand_base_link` / `R_hand_base_link` |
| `gr1_t2` | `l_hand_base_link` / `r_hand_base_link` |
| `g1` | `left_hand_palm_link` / `right_hand_palm_link` |

机器人自带的手会被自动隐藏（h1_2 26 个 link / gr1_t2 40 个 / g1 16 个）。

### 安装朝向是「算」出来的，不是「试」出来的

1. 用正运动学量出**机器人自带手**的坐标系 `M_robot`（手指方向 + 拇指侧）
2. 同法量出 **l21** 的坐标系 `M_l21`
3. 安装旋转 `R = M_robot * M_l21^T`

脚本会打印算出的 rpy，并**在安装后自动复测比对**，输出：

```
自检：手指方向差 0.00deg / 拇指方向差 0.00deg  [OK]
```

实测 6 个组合的角度差全为 `0.000deg`，矩阵最大分量差 `~1e-07`
（纯 float32 精度），`det(R) = +1.0000`（**无镜像问题**）。

### ⚠️ 过程中踩到一个 pybullet 大坑（已修，务必记住）

`p.loadURDF()` 对**自由刚体**是按 **base link 的质心（COM）** 摆放
`basePosition` 的，**不是 link 原点**：

```python
p.loadURDF(手.urdf, [0.286, 0.2095, 0.095], orn)
p.getBasePositionAndOrientation(手)     # 实际 (0.2773, 0.2372, 0.1713)
#                                              偏移 0.0816 m
```

手的位置差 8 cm，会导致「测量它的坐标系时参考系错位」，
**手指方向被算歪约 15 度** —— 表现是「朝向不对但看不出哪里不对」。
修复：测量时用 `getBasePositionAndOrientation()` 的**实际**值作参考，
并在 `loadURDF` 后立即 `resetBasePositionAndOrientation()` 到目标位姿。

> 详细排查过程见 `PROJECT_CONTEXT.md` 第 17 节。

### 用法

```powershell
# 任意机器人 + 自动朝向
python scripts/replay_hand_on_robot.py --robot h1_2   --hand both --render
python scripts/replay_hand_on_robot.py --robot gr1_t2 --hand both --render
python scripts/replay_hand_on_robot.py --robot g1     --hand both --render

# 无头验证（不弹窗）
python scripts/replay_hand_on_robot.py --robot h1_2 --hand both

# 手动覆盖（非零时优先于自动值）
python scripts/replay_hand_on_robot.py --robot h1_2 --hand right --render `
    --mount-offset 0 0 0.05 --mount-rpy 0 0 1.5708
```

### 这对阶段 2 意味着什么

- **手 -> 机器人的挂载已经解决**，接口固定（手基座 link + 固定约束）
- 阶段 2 只需要把**契约 H 里的双臂 7 关节**喂给机器人即可，
  **手会自动跟随手臂运动**，不需要额外改动
- 目前手臂保持不动（还没手臂数据），所以画面是「静止站立的机器人 + 活动的手指」

---

## 8.11 ★ 决策变更：仿真改用机器人【原装手】（2026-09-12）

> 本节记录一次**方案变更**，请以本节为准；8.10 的「把 l21 装到腕部」保留为备用方案。

### 最终采用：原装手 + 有损降维映射

| | **采用** | 备用 |
|---|---|---|
| 方案 | **用机器人自带的灵巧手** | 把 l21 装到腕部 |
| 脚本 | `scripts/replay_hand_native.py` | `scripts/replay_hand_on_robot.py` |
| 映射模块 | `teleop/native_hand.py` | —— |

**原因**：
1. 机器人保持**完全原装**，与论文的 H1-2 基线更可比
2. 使用者实测反馈：换成 LinkerHand **一直"手抖"**
3. 原装手的动力学参数是厂商调过的，不用额外治理

### ⚠️ 必须声明的代价：**有损映射**

数据是 L21 的 17 个自由度，原装手装不下：

| 机器人 | 可表达 | 覆盖率 | 丢弃 |
|---|---|---|---|
| **H1-2** | **12/17** | 71% | 4 个 `*_mcp_roll`（侧摆）+ 拇指 `cmc_roll` |
| GR1-T2 | 11/17 | 65% | 再丢拇指 1 个 |
| G1 | 7/17 | 41% | 另丢无名指/小指整根 |

**映射原则：语义 1:1，不做无依据的"合并"**

```
L21 mcp_pitch  <->  原装手 proximal        （近端指节屈曲）
L21 pip        <->  原装手 intermediate    （远端指节屈曲）
L21 *_mcp_roll ->   丢弃
拇指 cmc_yaw/pitch -> proximal_yaw/pitch
拇指 mcp/ip        -> intermediate/distal
```

### ★ 符号方向自动判定（关键）

```
H1-2   [0.000, +1.700]   屈曲为正  -> sign = +1
GR1-T2 [-1.570, 0.000]   屈曲为负  -> sign = -1
G1     left 负 / right 正（左右手还不一样）
规则：sign = +1 if |hi| >= |lo| else -1
```
不处理的话手指会**反着弯**。

### 「手抖」根因（已查明）

l21 的 URDF 在 pybullet 里动力学参数退化：
`base link 质量 1.5785e-07 kg`、`惯量对角 (0,0,0)`、`各指节 0.0004~0.003 kg`、`damping=friction=0`。
→ 位置控制增益相对这么小的惯量过大 → 过冲 / 数值发散。

修法（备用方案脚本已默认开启）：
`p.changeDynamics(..., mass=0.02, localInertiaDiagonal=[1e-6]*3)` + `p.setTimeStep(1/1000)`
→ 稳态误差 0.4380 → **0.0007**。

### ★★ 另一个普遍问题：仿真时间 ≠ 数据时间

原来所有回放脚本**每帧只调一次 `stepSimulation()`**：
`dt=1/240` 时每帧只推进 4.17 ms，而数据是 33 ms/帧 → **慢放 1/8，关节必然滞后**。

现在两个脚本都用 `--substeps`（默认自动匹配数据时间，约 8 步）。
验收实测：**跟踪误差 0.0018 rad**（不给足步数则 0.1878 rad）。

### 用法

```powershell
# 原装手（★ 本项目主用）
python scripts/replay_hand_native.py --robot h1_2   --hand both --render
python scripts/replay_hand_native.py --robot gr1_t2 --hand both --render
python scripts/replay_hand_native.py --robot g1     --hand both --render

# 只看映射报告（不开仿真）
python scripts/replay_hand_native.py --robot h1_2 --hand both --report
```

> 详细记录见 `PROJECT_CONTEXT.md` 第 18 节。

### 对任务级评测的影响（写报告要注意）

用原装手后，**手部动作是有损的**（丢侧摆/对掌），因此：

- ✅ 可以用来展示「数据 -> 手部动作」的因果链
- ⚠️ **不适合**断言"手部动作与真机一致"
- ⚠️ 若论文对比需要精确的手部行为，应回到 L21 方案（无损）并声明换手

**但无论哪种方案，手臂都不会动** —— 数据里没有手臂关节角，也没有手基座位姿。
这与用哪只手无关。

---

## 8.12 ★★ 坏帧 `right[535]` 的根因已查明：**MediaPipe 左右手身份切换**

> **这一节请直接转给队友（队友A / 数据提供方）。**
> 队友自己核对原始关键点后给出了根因，我们用角度数据独立验证通过。

### 结论：不是模型输出坏值，是**输入端身份标签错了**

`right_angles[535]` 单帧塌陷（范数 0.8125，10/18 维接近 0），前后帧正常。

**队友从原始关键点找到的证据**：

```
frame 534:  left 中心 x ≈ 0.731    right 中心 x ≈ 0.304
frame 535:  left 全零（消失）       right 中心 x ≈ 0.747

R535 vs L534 的 RMSE = 0.0216   <- 非常像
R535 vs R534 的 RMSE = 0.2654   <- 不像
```

### 我们独立验证（只用角度 + valid，不需要原始关键点）

| 验证项 | 结果 |
|---|---|
| `left_valid[535]` 是否为 False | ✅ **是**（`[536]` 也是；537 起两侧都无效 = 录制结束）|
| `right[535]` 最像谁 | ✅ 除自己外**第二像就是 `L[534]`（RMSE 0.0441）**，前 5 名里 4 个是 `L[53x]` |
| 对照 `right[534]` 的邻居 | 全是 `R[53x]`（0.0366/0.0455/0.0618）—— 正常右手轨迹 |
| `right[536]` 也可疑吗 | ✅ **是**（与 `R[534]` 的 RMSE 0.316，正常相邻帧只有 ~0.03）|

**→ `right[535]` 和 `[536]` 实际都在跟踪之前的左手。**

### 为什么原来的 valid 检查抓不到

采集侧只判断：

```python
np.isfinite(points).all() and not np.all(points == 0)
```

只要关键点非零且无 NaN 就是 valid —— **身份跳变完全检不出来**。

### 仿真侧的修复（已完成）

`teleop/filters.py` 新增 `detect_identity_swaps` / `hold_last_valid` / `repair_stream`：

- **身份切换帧 → 保持上一有效姿态**（不用插值：`536` 同样受污染，插值会被拉偏）
- **真塌零帧 → 插值**
- 两个回放脚本 + 一键验收都已接入，验收 **7/7 PASS**

判据（不需要原始关键点）：

```
d_same  = RMSE(X_s[t], X_s[t-1])
d_cross = RMSE(X_s[t], X_other[t-1])
条件：d_cross < 0.5 * d_same  且  d_same > 0.05  且  另一侧本帧【消失】
```

> ★ 只用 RMSE 比值会**误报 7 处**（345/346/353/378/383/429 —— 都是**真实快速运动**）；
> 加上「另一侧消失」这个门控后**只命中 535，0 误报**。

### ★ 建议队友在采集/加载侧根治

1. 检测**手掌中心瞬移** + **左右身份交换**
2. 发生跳变后**清空 3 帧历史**，等重新积满 3 帧再恢复 `valid=True`
3. 按此规则，`535`、`536` 都会保持上一有效姿态

> 详细记录见 `PROJECT_CONTEXT.md` 第 18.8 节。


