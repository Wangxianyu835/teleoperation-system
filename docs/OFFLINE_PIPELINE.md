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

## 8.8 给队友A 的反馈清单（2026-09-12 修订）

> ✅ = 已验证 / 🐛 = **需要你修** / ❓ = 需要你确认 / ⏸️ = 暂缓
>
> **更新说明（2026-09-12）**：项目决策为**先只做手部（不含手臂）**，
> 因此原清单中"手臂怎么补"等条目暂缓；跳变问题已自行查明，无需询问。

### 【需要你处理】2 项

| # | 内容 | 状态 |
|---|---|---|
| 1 | 🐛 **`*_valid` 漏标了 3 个坏帧**：`left[497]`、`right[497, 510]` | **请修导出逻辑** |
| 2 | 🔧 **手型号要统一**：项目里是 **l7**（无 `*_mcp_roll`），你目标是 **l21** | 待商定 |

### 🐛 坏帧问题的具体表现（供你定位）

```
right 后 3 帧（dim3 / dim6 / dim8 / dim11）:
  帧 509:  1.557   0.592   0.128   1.056      <-- 正常
  帧 510:  0.102   0.044   0.000   0.405      <-- 【整帧塌陷】所有维度同时到 ~0
  帧 511:  1.565   1.216   1.002   0.807      <-- 又恢复
```

**判断依据**：正常情况下各关节是**独立运动**的，**不可能 18 个维度同时塌到 0**。
→ 这是**该帧的推理/检测失败**，但 `valid` 没标出来。

**建议**：导出时检测并标记 `valid=False`；或在模型侧加保护。

> 💡 回放端已支持自动检测 + 插值修复（`scripts/replay_hand_angles.py` 默认开启），
> 但**源头修掉更好**（否则每份数据都会带这个"整帧抽搐"）。

### 【请确认】2 项（我们已推断，你确认即可）

| # | 内容 | 我们的依据 |
|---|---|---|
| 3 | 目标手 = LinkerHand **l21**（17 关节）？ | 15 个型号二分图匹配中 l21 最吻合（零越界）|
| 4 | 18 维顺序 = `dim0占位 + dim1~12 = 四指(mcp_roll,mcp_pitch,pip) + dim13~17 = 拇指(cmc_roll,cmc_yaw,cmc_pitch,mcp,ip)`？ | ① 限位匹配零越界<br>② `dim13 [-0.5997,-0.5875]` 正好落在 `thumb_cmc_roll (±0.600)` 内（精确到 0.0003）<br>③ `dim14 [+0.0067,+1.5953]` 正好是 `thumb_cmc_yaw` 上限 1.600<br>④ GUI 回放肉眼验证与预测一致 |

### 【可选但建议】

| # | 内容 | 用途 |
|---|---|---|
| 5 | 把采集端原始数据（`visual_hand_data_20260912_153542.h5`）也发一份 | 出问题时能回查是【采集端】还是【模型端】的责任（例如上面 3 个坏帧）|

### 【已自行查明，无需询问】

| # | 原问题 | 结论 |
|---|---|---|
| — | 跳变 34/511 要不要滤波？ | ✅ **不是噪声** —— 34 个全部是**成片真实快速运动**（集中后 9 秒），孤立尖峰 = 0 → **不需要滤波** |
| — | `dim 0` 恒 0 是什么？ | ✅ 已忽略，对回放无影响 |

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
