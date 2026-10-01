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
> 训练用 CPU 也很快（示例数据 3 轮约 7 ~ 12 秒，实测 6.97s / 11.85s），GPU 不是必需条件。

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
```

> 根目录还留着队友的便捷入口 `inspect_angle_h5.py`，但它导入的是 `retargeting.inspect` 里
> **不存在**的 `main`（实测 `ImportError: cannot import name 'main'`）。本轮**未修改该文件**
> （属于队友代码），请统一用上面的 `python -m retargeting inspect`。修法见第 9 节末尾。

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
| 手部流水线端到端 | 第 3 节五步 | 全链路跑通（对齐 240 帧 → 训练 3 轮 6.97s / 11.85s → 导出 240×18 → 回放三台机器人）|
| 队友数据回归验收 | `python scripts/verify_hand_pipeline.py` | **7 / 7 PASS** |
| 原装手运动量 | `python scripts/check_native_hand_motion.py --file <角度.h5> --quiet` | **「没动」= 0，三台全 `[PASS]`**；可映射关节 H1-2 24 / GR1-T2 22 / G1 14。会动数随数据波动：自造 240 帧示例 = 24/22/14 全动；队友 557 帧真实数据 = 22（+2 项「幅度偏小」）/ 22 / 14 |
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
| `roboticstoolbox-python` 没装（瘦环境）| `retargeting/arm.py` 顶层就 import 它，于是 `import retargeting.dual_teleop` / `tests/test_dual_arm.py` 会 `ModuleNotFoundError`（**队友代码如此，本轮有意未改**）| 手部流水线根本不导入该模块，不受影响；要跑机械臂就装一次：`python -m pip install roboticstoolbox-python`（参考环境已装 1.4.4）|

---

## 7. 排错表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ImportError: cannot import name 'main' from 'retargeting.inspect'` | 根目录包装脚本 `inspect_angle_h5.py` 指向不存在的函数（**队友脚本自身问题，本轮未改**）| 改用 `python -m retargeting inspect --angle-h5 <角度.h5>`；想让这个入口能用，见第 9 节末尾的 5 行修法 |
| `ModuleNotFoundError: No module named 'input_adapters'` | `input_adapters/` 包与 `hand_keypoints.py` 从未入库（**已修**：新增转发层，并在转发层内补齐 `NPY_REPLAY_SOURCE`，未改队友文件）| 重新拉最新代码 |
| `ModuleNotFoundError: roboticstoolbox` | 机械臂链路依赖没装（`retargeting/arm.py` 顶层 import，队友代码原样保留）| `python -m pip install roboticstoolbox-python`；只跑手部可忽略 |
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
| `inspect_angle_h5.py` | 根目录便捷入口（**队友脚本，当前 import 不存在的 `main`，本轮未改**；请用 `python -m retargeting inspect`）|
| `retargeting/{training,inference,inspect,data,arm,dual_teleop}.py` | 重定向算法主体（**本轮零改动**）|
| `tests/conftest.py` | 缺失第三方资产/依赖时跳过机械臂用例（不改队友测试文件）|
| `input_adapters/hand_keypoints.py` | 输入适配转发层（指向 `retargeting/tracking.py`，并在此定义队友代码缺的 `NPY_REPLAY_SOURCE`）|
| `requirements-retargeting.txt` | 重定向依赖（含可选机械臂依赖说明）|

---

## 9. 本轮改动清单（原则：**不修改队友的算法代码**）

**先给结论（可自查）**：

```bash
git diff 3e4e763..HEAD --stat -- retargeting/ tests/test_dual_arm.py inspect_angle_h5.py \
    input_adapters/npy_replay_adapter.py main_offline_dual_teleop.py
# → 输出为空：队友已入库的源码一行未改
```

改动全部落在**新增文件**上：

| # | 文件 | 为什么需要 | 复核方式 |
|---|---|---|---|
| 1 | `input_adapters/__init__.py`、`input_adapters/hand_keypoints.py`（**新增**）| 队友的 `npy_replay_adapter.py` 从 `input_adapters.hand_keypoints` 取 `HandWindowBuffer` / `NPY_REPLAY_SOURCE`，而该模块**从未入库**（任何提交都搜不到），一 import 就 `ModuleNotFoundError`。转发层把 `retargeting/tracking.py` 的实现按旧路径暴露出来；队友 `tracking.py` 里没有的 `NPY_REPLAY_SOURCE` 也在此定义（只是「来源」字符串标识，`tracking.py` / `hand_core.py` 都不校验取值）| `python scripts/validate_retarget_input.py tmp_motion/npy_replay_sample.npy` → `payloads=10 result=ok`, exit 0 |
| 2 | `tests/conftest.py`（**新增**）| 队友的 `tests/test_dual_arm.py` 缺 TRON2A URDF 时 3 个用例直接 ERROR，把测试套件染红、掩盖真实失败。用 pytest 钩子标记 skip，**不动队友测试文件** | `pytest tests -q` → 40 passed / 3 skipped；`-rs` 打印原因 |
| 3 | `scripts/run_retargeting_pipeline.py`（**新增**）| 原链路要手敲 5 条命令、checkpoint 路径还得自己拼 | `python scripts/run_retargeting_pipeline.py` → **7/7 PASS, exit 0** |
| 4 | `scripts/make_twohand_raw_sample.py`（**新增**）| 原始采集数据不在仓库，全链路没法自测 | 第 3 节步骤 0 → 造出 240 帧未对齐数据 |
| 5 | `requirements-retargeting.txt`（补注释）| 讲清手部必需依赖与机械臂依赖的边界 | `pip install -r requirements-retargeting.txt` 一次装齐 |
| 6 | `README.md`、`docs/RETARGETING_PIPELINE.md`（新增/补充）| 命令、实测数字、已知缺口、自检清单 | 见本文件第 10 节 |

### 9.1 已知但**有意未修**的队友代码问题（只做规避，不动源码）

| 位置 | 问题 | 我们的规避方式 | 建议队友的修法 |
|---|---|---|---|
| `inspect_angle_h5.py` | `from retargeting.inspect import main`，但 `retargeting/inspect.py` 里只有 `inspect_angle_h5` / `run` / `configure_parser`（`grep 'def main'` 无输出）→ 该入口必然 `ImportError` | 统一用 `python -m retargeting inspect --angle-h5 <角度.h5>` | 见下方 5 行 |
| `retargeting/tracking.py` | 只定义了 `VISIONPRO_SOURCE` / `MEDIAPIPE_APPROX_SOURCE`，没有 npy 回放需要的 `NPY_REPLAY_SOURCE` | 在 `input_adapters/hand_keypoints.py` 里定义该常量 | 把 `NPY_REPLAY_SOURCE = "npy_replay"` 加到 `tracking.py` 第 37 行旁即可 |
| `retargeting/arm.py` | 顶层 `import roboticstoolbox`，瘦环境下 `import retargeting.dual_teleop` 会失败 | 手部链路根本不导入该模块；参考环境已装 RTB 1.4.4 | 可选：改成函数内延迟导入 |

`inspect_angle_h5.py` 的建议修法（复用队友自己的 `configure_parser` + `run`，不引入新逻辑）：

```python
"""Command-line entry point for exported angle H5 inspection."""

import argparse

from retargeting.inspect import configure_parser, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="inspect_angle_h5.py")
    configure_parser(parser)
    raise SystemExit(run(parser.parse_args()))
```

> 实测依据：把 `arm.py` 还原成队友原版（顶层 `import roboticstoolbox`）后，
> 手部链路 `import`、`python -m retargeting inspect --angle-h5 ...`、
> `pytest tests -q`（40 passed / 3 skipped）**全部照常通过**。
> 说明手部流水线并不经过该模块，先前设想的「缺 RTB 会连累整个仓库」不成立，
> 因此本轮**不再改它**（相关误判已在本节更正）。

---

## 10. 自检清单（★ 不信任任何人，自己复现）

原则：**每一步都看「退出码 + 不变量」，不看漂亮输出**。
所有命令都在项目根目录执行，`python` 均指 `E:\python3.11.7\python.exe`（§2.1）。

### 10.1 七条命令（按顺序，全绿就是对的）

| # | 命令 | 期望 | 不对的话说明 |
|---|---|---|---|
| 0 | `python -c "import sys;print(sys.executable)"` | 打印 `E:\python3.11.7\python.exe` | 若打印 `.venv\Scripts\python.exe` → 你在用空壳环境，后面必然 `No module named 'torch'` |
| 1 | `python -m pytest tests -q` | `40 passed, 3 skipped`，退出码 0 | 出现 `ERROR`（而不是 `skipped`）→ 环境/资产有问题；`failed` → 真回归 |
| 2 | `python scripts/run_retargeting_pipeline.py` | 末尾 `7/7 PASS`、`结论：全链路跑通`、退出码 0 | 任一步 FAIL，该步日志会打印失败命令，可单独重跑 |
| 3 | `python -m retargeting inspect --angle-h5 tmp_motion/angles_twohand.h5` | 打印 `attr.output_shape=[18]` 等属性，退出码 0 | 文件缺属性 → 用的是未对齐数据或别的导出器 |
| 4 | `python scripts/check_native_hand_motion.py --file tmp_motion/angles_twohand.h5` | 表格三行全 `[PASS]`，**「没动」= 0** | 「没动 > 0」= 映射或回放坏了，不能拿去答辩；「幅度偏小」随数据波动（阈值 0.05 rad），**不判失败** |
| 5 | `python scripts/verify_hand_pipeline.py` | `7 / 7 PASS`，退出码 0 | 这是队友数据的回归验收，红了说明合并破了手部契约 |
| 6 | `python scripts/check_gbk_safe.py --strict` | `[OK]`，退出码 0 | 改了 `.py` 里有非 GBK 字符，中文 Windows 控制台会崩 |

> 完整"README 里逐字粘贴"版本（步骤 0→5：造数据 → 对齐 → 训练 → 导出 → inspect）见第 3 节，
> 本机已逐条复现通过（`saved_best=tmp_motion\ckpt\models\twohand_h5\linker\demo\model_best.pth` 确实存在）。

### 10.2 不看代码、直接验产物（推荐：最独立的一种）

**判断标准（18 维契约的不变量）**：
`left/right_angles` 都是 `(T, 18) float32`、**dim0 恒 0**、无 NaN/Inf、每维范围落在
`retargeting/config.py::ANGLE_LIMITS` 之内、`*_valid` 数量 = 有效帧数（首尾全零帧会被判无效，属正常）。

把下面这段存成 `tmp_motion/h5_dump.py`（本机已有这个文件）再跑：

```python
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    print("keys:", list(f.keys()))
    print("attrs:", {k: f.attrs[k] for k in f.attrs})
    for side in ("left", "right"):
        a = np.asarray(f[f"{side}_angles"][...])
        print(side, a.shape, "nonfinite=", int((~np.isfinite(a)).sum()),
              "dim0全0=", bool(np.allclose(a[:, 0], 0.0)))
        print("  每维 min/max:", [(round(float(a[:, i].min()), 3), round(float(a[:, i].max()), 3))
                                for i in range(a.shape[1])])
    for k in ("left_valid", "right_valid"):
        v = np.asarray(f[k][...]); print(k, int(v.sum()), "/", v.size)
```

本机 2026-09-30 实测输出（`tmp_motion/angles_twohand.h5`）：
`(240, 18) float32`、`nonfinite=0`、`dim0全0=True`、`left_valid 238/240`；各维 max 分别为
`0.00/0.14/1.567/1.457/0.128/1.567/1.564/0.164/1.567/1.561/0.170/1.567/1.556/-0.449/1.204/0.818/0.249/1.564`，
对照 `ANGLE_LIMITS`（`(-0.18,0.18)`、`(0,1.57)`、`(-0.6,0.6)`、`(0,1.6)`、`(0,1.0)`）**零越界**。

### 10.3 三个"假失败"陷阱（别被误导）

| 现象 | 真相 | 怎么证实 |
|---|---|---|
| 脚本明明跑完，退出码却是 `1` | PowerShell `\| Select-Object -Last N` 提前关管道，Python 收到 BrokenPipe | 改成 `python xxx.py > out.log 2>&1; $LASTEXITCODE`，再看 `out.log` |
| 控制台输出成 `鏈哄櫒浜?` 乱码 | 日志是 UTF-8，控制台按 GBK 解码；**脚本没坏** | `Get-Content out.log -Encoding UTF8` |
| `3 skipped` | 缺 TRON2A URDF 的机械臂用例（§6.1），**不是失败** | `pytest tests -q -rs` 会打印 skip 原因 |

### 10.4 "这算对了吗？"——可以写进答辩的判定依据

1. **结构对**：角度文件就是契约 H（`(T,18)` × 左右 + `*_valid` + 属性齐全）。
2. **数值对**：dim0 恒 0、无坏值、零越限（§10.2）。
3. **动作真的发生**：三台机器人可映射 24 / 22 / 14 个手部关节，「没动」= 0、最大跟踪误差 0.0000 ~ 0.0032 rad（§10.1 第 4 条）。
   （会动/幅度偏小的**具体个数随数据变化**，只有「没动 = 0」才是判据。）
4. **可回归**：`pytest` 40 passed / `verify_hand_pipeline.py` 7/7，说明合并没破坏既有契约。
5. **边界诚实**：手臂关节不动（数据只有手，§1）；机械臂 IK 因缺第三方 URDF 而 skip（§6.1）——
   这两条是**已知且有据的范围**，不是"跑坏了"。


