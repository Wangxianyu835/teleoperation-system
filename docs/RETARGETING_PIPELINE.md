# 手部重定向链路怎么跑、怎么复核（current-state 布局）

> **适用版本**：`origin/current-state`（包 `src/teleoperation`，CLI `python -m teleoperation`）
> **最后更新**：2026-10-10 · **相关**：[`CONVENTIONS.md`](CONVENTIONS.md)（规则与解释器）·
> [`contracts.md`](contracts.md)（H5 / 命令契约，字段以它为准）· [`architecture.md`](architecture.md)（模块层次）·
> [`archive/`](archive/)（旧布局手册，只读参考）
>
> 本文只讲**怎么用现有代码跑通并复核**；算法/契约本身由队友维护，**不要为跑通而改 `src/teleoperation`**（CONVENTIONS 约定 3）。

---

## 0. 一次性准备

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"                                  # 本机没有 conda 时必须
E:\python3.11.7\python.exe -m teleoperation --help             # 期望退出码 0
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1  # 环境自检，期望 RESULT [OK]
```

顶层子命令（`--help` 实测 2026-10-10）：

| 子命令 | 管什么 |
|---|---|
| `hand {align,train,export,inspect,realtime}` | 手部角度对齐 / 训练 / 导出 / 检查 / 实时 |
| `replay {actions,native-hand,l21-hand,mounted-hand}` | 回放：动作序列 / 原生手 / l21 手 / 装到机器人手上 |
| `sim {run,demo-joints}` | 任务仿真（`pushcube`/`pickcube` 等）与关节演示 |
| `dual {export,realtime,replay}` | 双臂（离线导出 / 实时 / 回放）|
| `tools {...}` | 一堆自检/可视化工具（见 §4）|

---

## 1. 离线链路：原始关键点 -> 手部角度 -> 关节命令

```powershell
$py='E:\python3.11.7\python.exe'

# (1) 原始关键点 H5 -> 归一化手部角度 H5（对齐左/右，每侧 18 维）
& $py -m teleoperation hand align --input  datasets\raw\retarget_twohand_153542.h5 `
                                   --output outputs\tmp_hand\angles.h5

# (2) 复核角度文件（帧数 / 维度 / 有效帧 / 越限）
& $py -m teleoperation hand inspect --angle-h5 outputs\tmp_hand\angles.h5

# (3) 训练（可选；不训练就用已有 checkpoint 或 hand align 的直出结果）
& $py -m teleoperation hand train --input outputs\tmp_hand\angles.h5 --run-name my_run --device cpu

# (4) 用 checkpoint 导出角度（把模型输出写回 H5）
& $py -m teleoperation hand export --input datasets\raw\retarget_twohand_153542.h5 `
                                   --checkpoint <checkpoint 路径>.pt `
                                   --output outputs\tmp_hand\angles_from_ckpt.h5 --device cpu
```

- `hand align` 只有两个参数：`--input` / `--output`（实测）。
- `hand export` 还有 `--batch-size` / `--scale-factor` / `--disable-identity-tracking`
  / `--max-center-displacement` / `--max-shape-rmse` / `--device {auto,cpu,cuda}`。
- 产出文件一律写 `outputs\...`（`.gitignore` 已忽略；CONVENTIONS 约定 7）。
- **H5 字段与语义以 [`contracts.md`](contracts.md) 为准**，本文不重复定义。

## 2. 回放与可视化（每步都能单独验）

```powershell
$py='E:\python3.11.7\python.exe'

# l21 手：角度 -> l21 关节（可选 --check 做检查、--render 开 GUI、--analyze 出统计）
& $py -m teleoperation replay l21-hand --file outputs\tmp_hand\angles.h5 --check

# 原生手：把角度直接喂给机器人自带手（--robot h1_2/gr1_t2/g1，可 --render --view left-hand）
& $py -m teleoperation replay native-hand --file outputs\tmp_hand\angles.h5 --robot h1_2 --render

# 把手装到机器人手腕上（检查安装位姿/抖动）
& $py -m teleoperation replay mounted-hand --file outputs\tmp_hand\angles.h5 --robot h1_2 --render

# 动作序列（契约 H 的 .npz/.h5）或假数据，无头跑，最省事的冒烟
& $py -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render
& $py -m teleoperation replay actions --file datasets\...\actions.npz --robot h1_2 --no-render
```

## 3. 任务与整机

```powershell
$py='E:\python3.11.7\python.exe'
& $py -m teleoperation sim run --task pushcube --robot h1_2 --no-render --demo    # 有头去掉 --no-render
& $py -m teleoperation sim demo-joints                                            # GUI 里看关节
& $py -m teleoperation dual export --observations <obs> --angle-h5 <angles.h5> --calibration <file> --output outputs\tmp_dual\cmd.h5
& $py -m teleoperation dual replay --command-h5 outputs\tmp_dual\cmd.h5 --calibration <file>
```

`sim run` 的动作空间实测为 **38 维** = 左臂 7 + 右臂 7 + 左手 12 + 右手 12（`tools check-environment` 会打印）。

---

## 4. 一键复核（改动前后都要跑同一条）

| 命令 | 判据 | 说明 |
|---|---|---|
| `python -m teleoperation tools verify-hand-pipeline` | 退出码 `0`，末行 `结论：... [PASS]` | **整条离线链路的验收器**：读真实 H5（557 帧 × 18 维）→ 逐项检查 → 仿真回放。旧布局的 `scripts/run_retargeting_pipeline.py` 对应物 |
| `python -m teleoperation tools check-environment` | 退出码 `0` | 项目自带环境自检：6 项，含 300 步无头端到端（`[OK] 端到端跑通（robot_id=1, action_dim=38）`）|
| `python -m unittest discover -s tests -v` | 见 [`CONVENTIONS.md`](CONVENTIONS.md) 约定 5 / §3.2 | 本机回落下（**必须 `PYTHONUTF8=1`**，否则多 6 项编码假失败）实测 `Ran 197 tests / failures=0 / errors=17 / skipped=1`；判据是**不劣化** |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full` | 退出码 `0` | 上面三条 + 4 条冒烟，共 8 项一次跑完（含 GBK 两条、架构边界、C 盘零写）|

`tools` 里还有一批单项工具（`--help` 实测）：`validate-retarget-input`、`check-gbk-safe`、
`check-camera`、`show-robots`、`show-hand`、`show-all-hands`、`visualize-hand-keypoints`、
`visualize-l21-fk`、`compare-training-hand-pose`、`diagnose-hand-coordinates`、`make-sample-data`、`verify-p0`。
排查问题时优先用它们，**不要**先改代码。

---

## 5. 旧布局 → 新布局对照表（合并前的习惯怎么迁过来）

| 旧（`3e4e763` 及之前）| 新（`origin/current-state`）|
|---|---|
| `python -m retargeting ...` | `python -m teleoperation hand ...` |
| `scripts/run_retargeting_pipeline.py`（我写的一键流水线）| `python -m teleoperation tools verify-hand-pipeline` |
| `inspect_angle_h5.py` | `python -m teleoperation hand inspect --angle-h5 <angles.h5>` |
| `main_offline_dual_teleop.py` | `python -m teleoperation dual export` / `dual replay` |
| `scripts/replay_hand_native.py` | `python -m teleoperation replay native-hand` |
| `retargeting/**`、`input_adapters/**`、`config/**` | `src/teleoperation/**`（同一个包内，路径变了）|
| `pytest tests -q`（40 passed / 3 skipped）| `python -m unittest discover -s tests -v`（197 tests）|
| `robots/from_teleopbench/**` | `assets/robots/**` |
| `data/**`、`outputs/**` | `datasets/**`、`outputs/**` |
| 仓库根 `.venv` 空壳、`E:\python3.11.7` 全路径解释器 | 正式环境是 conda `teleoperation`；本机仍回落 `E:\python3.11.7` + `PYTHONPATH=src` |

**迁移时踩到的两个小坑**（已写进 CONVENTIONS）：
`src/` 布局下必须带 `$env:PYTHONPATH="$pwd\src"`（包没装进回落解释器）；
`tmp_*` 在本仓库**不会**被 `.gitignore` 忽略（只有 `outputs/` 会），临时文件要放 `outputs\tmp_*\`。