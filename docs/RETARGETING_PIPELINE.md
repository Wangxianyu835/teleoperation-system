# 重定向流水线跑通指南（RETARGETING PIPELINE）

> **用途**：把「原始双手关键点 → 18 维 L21 角度 → 三台机器人仿真回放」这条链路
> **在本机端到端跑通**，并记录每一步的**实测输出**与**已知缺口**。
> 这是队友 retargeting 分支合并进主仓库后的对接口径。
>
> **配套文档**
> - 环境配置 / 解释器 / GBK 约定 → [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)
> - 契约 G / H（数据文件格式）→ [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md)
> - 离线回放流水线（本平台侧）→ [`OFFLINE_PIPELINE.md`](OFFLINE_PIPELINE.md)
> - 队友从零上手 → [`TEAM_ONBOARDING.md`](TEAM_ONBOARDING.md)
>
> **文件位置**：`F:\simulation_platform\docs\RETARGETING_PIPELINE.md`
> **最后更新**：2026-09-30（实测环境：`E:\python3.11.7\python.exe`）

---

## 1. 这条流水线在做什么

```
原始双手关键点 H5          对齐到 L21 坐标系         自监督训练              导出                 仿真回放
(T,21,3) 未对齐     ->   (T,25,3) + frame=l21  ->  角->FK->对比关键点  ->  18 维角度 H5  ->  三台机器人原装手
datasets/raw/*.h5        tmp_motion/*_aligned.h5     model_best.pth        left/right_angles
```

| 阶段 | 代码 | 说明 |
|---|---|---|
| 对齐 | `scripts/align_h5_coordinates.py` → `retargeting/data.py` | 原始数据必须显式对齐；`coordinate_frame` 属性是训练/导出的强制前置条件 |
| 训练 | `retargeting/training.py` | **自监督**：网络预测 18 维角度 → 正运动学算回关键点 → 与输入关键点对比。**不需要真实采集设备**，只要有原始关键点就能自跑 |
| 导出 | `retargeting/inference.py` | 用 checkpoint 推理出 `left_angles` / `right_angles`（各 18 维）+ `*_valid` |
| 校验 | `retargeting/inspect.py` | 帧数 / 有效帧 / 非有限值 / 越限统计 + 关键属性 |
| 回放 | `scripts/show_hands_all.py`、`scripts/replay_hand_native.py` | 把角度文件灌给 H1-2 / GR1-T2 / G1，原装灵巧手按同一份数据同步屈伸 |

> **为什么这条链路能自己跑通**：训练是自监督的，损失函数里没有真实机器人标签，
> 所以「造一份原始关键点 → 训练 → 导出 → 回放」闭环成立，不依赖队友的采集设备。

---

## 2. 先决条件

### 2.1 解释器与依赖

跑项目用的是**全局 Python `E:\python3.11.7\python.exe`**（项目内 `.venv` 是空的，见 `ENVIRONMENT_SETUP.md` §2.2）。

```powershell
# 手部流水线必需的依赖（版本区间已实测：numpy 2.4.6 / h5py 3.16.0 / scipy 1.17.1）
E:\python3.11.7\python.exe -m pip install -r requirements-retargeting.txt

# 机械臂链路（可选，且还需要第三方 TRON2A 描述包，见第 6 节）
E:\python3.11.7\python.exe -m pip install roboticstoolbox-python
```

### 2.2 原始数据从哪来

三种都可以：

| 来源 | 说明 |
|---|---|
| 队友采集的原始数据 | 放在 `datasets/raw/` 的**未对齐**双手关键点 H5（训练前必须先跑第 3.1 步对齐）|
| 队友已导出的角度文件 | `datasets/raw/retarget_twohand_*.h5`（18 维，**可直接回放**，跳过训练）|
| 本机自造示例 | `python scripts/make_twohand_raw_sample.py --output tmp_motion/raw_twohand.h5 --frames 240` |

---

## 3. 五步命令（含实测输出）

### 3.0 一键跑通（推荐先跑这个）

```bash
python scripts/run_retargeting_pipeline.py                 # 全链路：对齐 -> 训练 -> 导出 -> 校验 -> 回放
python scripts/run_retargeting_pipeline.py --no-replay     # 不启动仿真，只要角度文件
python scripts/run_retargeting_pipeline.py --epochs 30 --device cpu
```

产物落在 `tmp_motion/pipeline/`；脚本**不只看退出码**，还会独立用 h5py 复核
`left/right_angles` 的形状与有效帧，最后打印 PASS/FAIL 汇总。

### 3.1 对齐坐标系

```bash
python scripts/align_h5_coordinates.py \
    --input  tmp_motion/raw_twohand.h5 \
    --output tmp_motion/aligned_twohand.h5
```

作用：原始 `(T,21,3)` → 补成 25 点并转到 **L21 固定坐标系**，写入属性
`coordinate_frame=l21`、`coordinate_alignment=source_to_l21_xyz`、`source_file=...`。
**没有这个属性，训练/导出会直接报错**（这是有意的硬门控）。

### 3.2 训练

```bash
python -m retargeting train \
    --input tmp_motion/aligned_twohand.h5 \
    --run-name smoke_e2e --epochs 3 --batch-size 16 \
    --device cpu --checkpoint-root tmp_motion/ckpt
```

实测（240 帧示例数据、CPU、3 轮）：

```
raw_frames=240
train_raw_range=[0, 192)      validation_raw_range=[194, 240)
train_windows=190             validation_windows=44
coordinate_alignment=source_to_l21_xyz
epoch=1 train_total=1060.803828 val_total=744.902729
epoch=2 train_total=749.448767  val_total=719.169975
epoch=3 train_total=725.100178  val_total=711.822732
saved_best=tmp_motion/ckpt/models/twohand_h5/linker/smoke_e2e/model_best.pth
training_time=11.848s
```

> checkpoint 路径规则：`<checkpoint-root>/models/twohand_h5/<robot>/<run-name>/model_best.pth`。
> 训练用 CPU 也很快（示例数据 3 轮约 12 秒），GPU 不是必需条件。

### 3.3 导出角度

```bash
python -m retargeting export \
    --input tmp_motion/aligned_twohand.h5 \
    --checkpoint tmp_motion/ckpt/models/twohand_h5/linker/smoke_e2e/model_best.pth \
    --output tmp_motion/angles_smoke_e2e.h5 --device cpu
```

实测：`frames=240`、`left_shape=(240, 18)`、`right_shape=(240, 18)`、双侧各 `238` 帧有预测。

### 3.4 校验角度文件

```bash
python -m retargeting inspect --angle-h5 tmp_motion/angles_smoke_e2e.h5
# 等价入口（根目录包装脚本，两种写法都行）
python inspect_angle_h5.py tmp_motion/angles_smoke_e2e.h5
```

实测：`left_valid=238 left_invalid=2 left_nonfinite=0 left_out_of_limits=0`
（右侧相同），并打印 `attr.checkpoint` / `attr.coordinate_alignment` / `attr.invalid_angle_policy=hold_previous` 等。

### 3.5 仿真回放

```bash
python scripts/show_hands_all.py --file tmp_motion/angles_smoke_e2e.h5      # 三台机器人并排
python scripts/replay_hand_native.py --robot h1_2 --hand both --render      # 单台 + 可视化
python scripts/check_native_hand_motion.py --file tmp_motion/angles_smoke_e2e.h5 --quiet
```

实测（240 帧 × 3 台机器人，原装手映射）：

| 机器人 | 会动的映射关节 | 幅度偏小 | 没动 | 最大跟踪误差 | 结论 |
|---|---|---|---|---|---|
| H1-2 | 24（左右各 12）| 0 | 0 | 0.0000 rad | PASS |
| GR1-T2 | 22（左右各 11）| 0 | 0 | 0.0000 rad | PASS |
| G1 | 14（左右各 7）| 0 | 0 | 0.0000 rad | PASS |

> 有损映射（丢弃原装手没有的自由度）与覆盖率见 `OFFLINE_PIPELINE.md` 第 8.11 节。

### 3.6 队友数据的回归验收

```bash
python scripts/verify_hand_pipeline.py       # 默认用 datasets/raw/retarget_twohand_153542.h5
```

实测：**7 / 7 PASS**（557 帧 × 18 维，left 92% / right 92% 有效帧，0 越限，三台机器人回放全部映射成功）。
这条命令是合并不回归的证据。

---

## 4. 文件契约（照着这个写就不会接不上）

| 文件 | 必需内容 | 形状 / 类型 |
|---|---|---|
| 原始关键点 H5 | `left_hand_keypoints`、`right_hand_keypoints`、`frame_ids`、`timestamps` | `(T,21,3) float32`、`(T,) int64`、`(T,) float64` |
| 对齐后关键点 H5 | 同上 + 属性 `coordinate_frame=l21` | 点数补成 25：`(T,25,3) float32` |
| 角度 H5（契约 H） | `left_angles`、`right_angles`、`left_valid`、`right_valid`、`frame_ids`、`timestamps` | 角度 `(T,18) float32`、valid `(T,) bool` |

`18` 维 = 17 个手指关节 + 1 个掌心/朝向分量；下游回放只认这个布局。

---

## 5. 本次合并后的实测记录（2026-09-30）

| 项目 | 命令 | 结果 |
|---|---|---|
| 全量测试 | `python -m pytest tests -q` | **40 passed, 3 skipped**（跳过的是缺 URDF 的机械臂用例）|
| 手部流水线端到端 | 第 3 节五步 | 全链路跑通（对齐 240 帧 → 训练 3 轮 11.8s → 导出 240×18 → 回放三台机器人）|
| 队友数据回归验收 | `python scripts/verify_hand_pipeline.py` | **7 / 7 PASS** |
| 原装手运动量 | `python scripts/check_native_hand_motion.py --file <角度.h5> --quiet` | H1-2 24 / GR1-T2 22 / G1 14 个关节会动，0 卡死 |
| 三台机器人回放 | `python scripts/show_hands_all.py --file <角度.h5>` | 240 帧完成，退出码 0 |
| 一键脚本 | `python scripts/run_retargeting_pipeline.py` | **7/7 PASS，exit 0**（本轮新脚本复查，产物在 `tmp_motion/pipeline/`）|
| 训练前后对比图 | `python scripts/compare_training_hand_pose.py --input <对齐.h5> --frame 120 --before-checkpoint <ckpt> --after-checkpoint <ckpt> --device cpu` | 生成 690 KB `compare_hand_pose.png`，exit 0（确认该队友脚本可正常导入并运行）|

> **提示**：用 PowerShell 的 `| Select-Object -Last N` 截断这些脚本的输出时，
> Python 收到 BrokenPipe 会**返回退出码 1**，这是管道行为、不是脚本失败。
> 要判断成败请重定向到文件：`python xxx.py ... > out.log 2>&1; $LASTEXITCODE`。

---

## 6. 已知缺口（不影响手部流水线）

### 6.1 第三方 TRON2A 机械臂描述包不在仓库里（★ 唯一硬缺口）

- 代码默认路径：`third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf`
- 现状：`third_party/` **目录不存在**，`.gitmodules` 也没有，历史上从未入库过（不是漏提交）。
- 受影响：`retargeting/arm.py`、`retargeting/dual_teleop.py`、`retargeting/realtime_dual_teleop.py`，
  入口 `main_offline_dual_teleop.py` / `main_realtime_dual_teleop.py` / `main_pybullet_dual_teleop.py`，
  以及 `tests/test_dual_arm.py`（3 个用例）。
- 表现：`FileNotFoundError: TRON2A URDF was not found: third_party\tron2-robot-description\tron2a\DACH_TRON2A\urdf\robot.urdf`
- 补齐方式：从 `tron2-robot-description` 仓库取 `tron2a/DACH_TRON2A/urdf/`（含 `robot.urdf` 与 mesh），
  按上面的相对路径放到项目根目录即可，无需改代码。

### 6.2 其它缺口

| 缺口 | 影响 | 现状 / 处理 |
|---|---|---|
| 预训练 checkpoint 不在仓库 | 直接用 `datasets/raw/*.h5` 回放没问题；想「重新推理」得先自己训练 | 本机用 `python -m retargeting train` 自监督训练即可（示例数据 12 秒）|
| 原始采集数据不在仓库 | 只有队友导出的 18 维角度文件 | 用 `scripts/make_twohand_raw_sample.py` 造示例原始数据先跑通链路 |
| `roboticstoolbox-python` 是可选依赖 | 不装时 `retargeting/arm.py` 顶层 import 会连累整个仓库（**已修**：改成延迟导入 + 给安装命令）| 手部流水线不需要它 |

---

## 7. 排错表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ImportError: cannot import name 'main' from 'retargeting.inspect'` | 根目录包装脚本 `inspect_angle_h5.py` 指向不存在的函数（**已修**）| 直接用修好的 `python inspect_angle_h5.py <角度.h5>` |
| `ModuleNotFoundError: No module named 'input_adapters'` | `input_adapters/` 包与 `hand_keypoints.py` 从未入库（**已修**：补齐转发层 + 补 `NPY_REPLAY_SOURCE` 常量）| 重新拉最新代码 |
| `ModuleNotFoundError: roboticstoolbox` | 机械臂链路可选依赖没装 | `python -m pip install roboticstoolbox-python`（只跑手部可忽略）|
| `KeyError: 'coordinate_frame'` / 训练拒绝开始 | 用了未对齐的原始数据 | 先跑第 3.1 步对齐 |
| `FileNotFoundError: TRON2A URDF was not found` | 缺第三方描述包 | 见 6.1 |
| 控制台中文乱码 / `UnicodeEncodeError` | 中文 Windows 控制台是 cp936 | `.py` 里禁止非 GBK 字符，提交前跑 `python scripts/check_gbk_safe.py --strict` |
| 脚本"失败"退出码 1 但输出完整 | PowerShell `Select-Object` 截断管道 | 重定向到文件再判断退出码（见第 5 节提示）|

---

## 8. 相关文件

| 文件 | 作用 |
|---|---|
| `scripts/run_retargeting_pipeline.py` | ★ 一键跑通全链路（本文件第 3.0 节）|
| `scripts/make_twohand_raw_sample.py` | 造未对齐的原始双手关键点 H5 |
| `scripts/align_h5_coordinates.py` | 原始 → L21 固定坐标系 |
| `scripts/show_hands_all.py` / `scripts/replay_hand_native.py` | 三台机器人 / 单台回放 |
| `scripts/check_native_hand_motion.py` | 量化"每个关节实际转了多少弧度" |
| `scripts/verify_hand_pipeline.py` | 平台侧一键验收（7 项）|
| `scripts/compare_training_hand_pose.py` | 训练前后对比图（同一帧的 FK 结果 vs 关键点目标，需 2 个 checkpoint）|
| `inspect_angle_h5.py` | 根目录便捷入口（**已修**：不再导入不存在的 `main`）|
| `retargeting/{training,inference,inspect,data,arm,dual_teleop}.py` | 重定向算法主体 |
| `input_adapters/hand_keypoints.py` | 输入适配转发层（指向 `retargeting/tracking.py`）|
| `requirements-retargeting.txt` | 重定向依赖（含可选机械臂依赖说明）|

---

## 9. 合并后修复清单（本次）

| # | 文件 | 问题 | 处理 | 复核方式 |
|---|---|---|---|---|
| 1 | `retargeting/arm.py` | 顶层 `import roboticstoolbox`，没装就**连累整个仓库**（连手部链路都跑不了）| 改为延迟导入 `require_roboticstoolbox()`，缺依赖时只在用到机械臂时报错并给出安装命令 | 卸载/不装 RTB 也能跑完第 3 节五步 + 全量测试 |
| 2 | `tests/test_dual_arm.py` | 3 个用例因缺 TRON2A URDF 直接 **ERROR**（把测试套件搞成红的）| 加 `@unittest.skipUnless(URDF.is_file(), ...)` | `pytest tests -q` → 40 passed / 3 skipped |
| 3 | `inspect_angle_h5.py` | `from retargeting.inspect import main`，而该函数**不存在** → 入口直接 ImportError | 重写为独立入口，支持位置参数 `<角度.h5>` 与 `--angle-h5` 两种写法 | `python inspect_angle_h5.py tmp_motion/angles_twohand.h5` → exit 0 |
| 4 | `input_adapters/` | 包与其 `hand_keypoints.py` **从未入库**，`ModuleNotFoundError` | 补齐转发层（本文件已入库）| `pytest tests -q` 全绿 |
| 5 | `retargeting/tracking.py` | 缺 `NPY_REPLAY_SOURCE` 常量，npy 回放路径直接报错 | 补常量 | 同上 |
| 6 | `scripts/run_retargeting_pipeline.py` | 原链路要手敲 5 条命令、checkpoint 路径还得自己拼 | 新增一键脚本（自带 h5py 独立复核，不依赖 `retargeting` 包）| 本机 `python scripts/run_retargeting_pipeline.py` → **7/7 PASS, exit 0** |

> 机械臂链路（`dual_teleop` 等）本身**未改逻辑**，只处理了「缺资产/缺依赖不应拖垮仓库」这一层。


