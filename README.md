# 遥操作与重定向项目

项目包含 L21 手部重定向、TRON2A 双臂 IK，以及 H1-2、GR1-T2、G1 的 PyBullet 任务和回放。生产代码统一位于 `src/teleoperation`。

**阶段汇报与答辩材料**：[present 成果包](present/README.md)，完整汇报正文见 [项目阶段成果](present/项目阶段成果.md)。

**2026-10-11 合并更新**：已集成实时三机器人手部展示、无需 checkpoint 的几何重定向，以及手部与肩肘腕的米制 world 数据采集。新增入口、数据差异与验证记录见 [分支集成说明](docs/integration-2026-10-11.md)。

## 安装

沿用现有 Conda 环境和 requirements 版本，不升级依赖。在已有环境中执行：

```powershell
conda activate teleoperation
python -m pip install -e . --no-deps
python -m teleoperation --help
```

要求 Python 3.10 或更新版本。若当前环境已安装 setuptools，可加 `--no-build-isolation` 完全使用现有构建依赖。

未安装时可临时使用源码：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m teleoperation --help
```

安装后 `teleoperation ...` 和 `python -m teleoperation ...` 使用同一个 CLI。帮助页仅解析参数，不启动设备、网络或仿真。

## 摄像头录制

`hand record` 使用摄像头和 MediaPipe，无需重定向 checkpoint。录制入口是 `applications/offline_h5_record.py`，流式写入由 `data/hand_recording.py` 负责。
原始 H5 保存左右手 `(T,21,3)` 关键点、连续帧号和 Unix 秒时间戳；未检测到的手保存为零，不丢弃该帧。正常结束或 Ctrl+C 后，程序自动生成 palm-local `(T,25,3)` 的对齐文件，
将这个文件作为 `hand train --input` 和 `hand export --input`。

`--hand-side left/right/both` 选择录制左手、右手或双手，默认 `both`；单手录制时另一侧保存为零，保持双手 H5 格式兼容。自动文件名沿用 mytrans 的年月日与时分秒格式，
例如 `visual_hand_data_left_20261009_201530.h5` 和 `aligned_visual_hand_data_left_20261009_201530.h5`。默认目录为 `datasets/raw/`，
用 `--output-dir` 更换目录；显式传入 `--output` 或 `--aligned-output` 时保留指定的完整文件名。程序拒绝覆盖已有输出；摄像头读取失败时保留已录制的原始帧并返回失败，
可用 `hand align` 单独对齐。训练前应确认有足够的连续有效手部帧，单手文件的训练使用对应的 `hand train --hand-side left/right`。

```powershell
# 摄像头录制：保存原始 21 点，并自动生成供训练/导出的 palm-local 25 点 H5
python -m teleoperation hand record --hand-side left 
python -m teleoperation hand record --hand-side right --output-dir datasets/raw 
python -m teleoperation hand record --hand-side both --frames 900
# 默认弹出摄像头与关键点预览；q/Esc、关闭窗口或 Ctrl+C 停止并保存
# 省略 --frames 持续录制；--no-preview 可关闭预览窗口

# H5 对齐：另存输出，保留原文件
python -m teleoperation hand align --input raw.h5 --output aligned.h5

# shared模型训练、权重初始化和导出（原有用法）
python -m teleoperation hand train --input aligned.h5 --run-name experiment --device cuda
python -m teleoperation hand train --input aligned.h5 --run-name experiment2 --init-checkpoint model_best.pth
python -m teleoperation hand export --input aligned.h5 --checkpoint model_best.pth --output angles.h5 --device cuda
python -m teleoperation hand inspect --angle-h5 angles.h5


```
```powershell
# 训练示例：左手独立模型，CUDA，随机初始化
python -m teleoperation hand train `
  --input datasets/raw/aligned_visual_hand_data_left_20261009_200808.h5 `
  --run-name separate_left_v1 --hand-side left --device cuda `
  --epochs 100 --batch-size 64 --learning-rate 0.0001 `
  --val-ratio 0.2 --early-stopping-patience 20 --seed 1234
```


`hand train` 默认从头初始化 PoseTransformer；省略 `--hand-side` 时使用 `shared`，左右手共用一份网络权重。指定 `--hand-side left` 或 `right` 时，
只用所选手的有效数据、对应 URDF 和 FK 训练一个独立模型。`--init-checkpoint` 用于加载初始网络权重，优化器、学习率调度器和训练轮次仍重新开始。
训练输入必须是已声明坐标对齐方式的 H5，摄像头录制生成的 `aligned_visual_hand_data_*.h5` 可直接使用。每个输入窗口包含连续 3 帧，每帧 25 个三维关键点，
即 `(3,25,3)`；网络输出 18 个关节角，其中第一个是固定为零的 root 占位，实际可执行关节为 17 个。输入不需要机器人角度标签：训练把预测角送入可微正向运动学（FK），再比较机器人和人手的关键点几何关系。
数据按时间顺序切分：默认前 80% 的原始帧用于训练，后段用于验证，并在边界留出 2 帧间隔，避免相邻三帧窗口跨越训练/验证边界。训练批次打乱，验证批次保持顺序；缺手或无效窗口由有效性 mask 排除，不把零帧作为有效目标。每轮完成训练和验证后，用验证集总损失调整学习率、判断早停并选择最佳权重。
常用参数可直接在训练命令中覆盖；下表是未指定这些选项时的实际默认值：

| 命令行参数 | 默认值 | 含义 |
|---|---|---|
| `--hand-side` | `shared` | 左右手共享模型；`left`/`right` 训练独立单手模型 |
| `--epochs` | `100` | 最多训练 100 轮，可能因早停提前结束 |
| `--batch-size` | `64` | 每批的时间窗口数量 |
| `--learning-rate` | `0.0001`（`1e-4`） | AdamW 的初始学习率 |
| `--val-ratio` | `0.2` | 按原始帧划分验证段的比例，另保留边界间隔 |
| `--early-stopping-patience` | `20` | 连续 20 轮验证损失未改善时停止 |
| `--seed` | `1234` | 随机种子 |
| `--device` | `auto` | CUDA 可用时选择 CUDA，否则选择 CPU |
| `--init-checkpoint` | 无 | 默认从头训练；指定后只加载模型权重 |
| `--checkpoint-root` | `outputs/hand_retargeting/checkpoint` | 模型和日志的根目录 |




要永久修改默认参数，或修改尚未开放命令行选项的设置，查看以下具体文件：

| 文件 | 设置或实现内容 |
|---|---|
| [src/teleoperation/retargeting/hand/config.py](src/teleoperation/retargeting/hand/config.py) | `TrainingConfig` 集中定义训练默认值、优化器数值、调度器数值、损失权重、碰撞阈值和尺度；`ModelConfig` 定义网络结构；`RuntimeConfig.device` 定义默认设备 |
| [src/teleoperation/cli/schemas.py](src/teleoperation/cli/schemas.py) | `hand_train()` 定义训练命令参数，从配置中读取默认值；命令行显式值覆盖这些默认值 |
| [src/teleoperation/applications/training.py](src/teleoperation/applications/training.py) | `_train()` 组装 Dataset、网络、AdamW、学习率调度器、验证和早停，保存 checkpoint |
| [src/teleoperation/learning/dataset.py](src/teleoperation/learning/dataset.py) | `TwoHandH5Dataset` 构建连续三帧样本、目标与有效性 mask；`TwoHandH5ChunkedGenerator` 组装和打乱批次 |
| [src/teleoperation/learning/trainer.py](src/teleoperation/learning/trainer.py) | `_run_epoch()` 计算损失、反向传播和梯度裁剪；`_split_frame_ranges()` 实现按时间切分及边界间隔 |
| [src/teleoperation/learning/losses.py](src/teleoperation/learning/losses.py) | `hand_loss()` 及各分项损失的具体公式 |
| [src/teleoperation/retargeting/hand/transformer.py](src/teleoperation/retargeting/hand/transformer.py) | `PoseTransformer` 网络实现；训练通过 `L21.model_kwargs()` 传入配置值，因此修改网络默认结构应优先修改 `config.py` 的 `ModelConfig` |
| [src/teleoperation/contracts/constants.py](src/teleoperation/contracts/constants.py) | 三帧窗口、25 个输入点、18 维输出等共享常量，模型配置引用这些值 |
| [src/teleoperation/paths.py](src/teleoperation/paths.py) | checkpoint 根目录、默认初始化权重路径等路径设置 |

`TrainingConfig` 中未提供训练命令行选项的数值如下：

| 配置字段 | 默认值 | 用途 |
|---|---|---|
| `weight_decay` / `optimizer_eps` | `0.01` / `1e-6` | AdamW 权重衰减与数值稳定项 |
| `scheduler_factor` / `scheduler_patience` | `0.5` / `8` | ReduceLROnPlateau 的学习率缩减倍数与耐心参数 |
| `minimum_learning_rate` | `1e-6` | 调度器的最低学习率 |
| `gradient_clip_norm` | `10.0` | 梯度范数裁剪上限 |
| `loss_weights` | `(500,500,10,500,0.01,500)` | 依次对应 `vec`、`pos`、`collision`、`thumb`、`tip_distance`、`thumb2` |
| `collision_threshold` | `0.010` | 碰撞损失中的点间距离阈值 |
| `source_scale` / `robot_scale` | `1.0` / `1.0` | 输入手部关键点与机器人 FK 的尺度参数 |

六项损失分别约束指骨方向、指尖方向、碰撞、拇指到手掌平面的距离、指尖间距离和拇指夹角；其中日志名 `pos` 对应代码里的归一化指尖方向比较。`ModelConfig` 的主要默认值为 `embed_dim_ratio=32`、`spatial_depth=6`、`temporal_depth=4`、`num_heads=8`、`drop_path_rate=0.1`，空间/时间 MLP 比例为 `4.0/1.0`。改变网络结构后，加载旧权重仍需满足现有 checkpoint 的结构校验。

默认模型目录为 `outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/<run-name>/`；独立模型在其下增加 `left/` 或 `right/`。`model_best.pth` 对应验证集总损失最低的一轮，`model_last.pth` 对应最后完成的一轮；同目录的 `epoch_metrics.csv` 保存逐轮指标。
日志在 `outputs/hand_retargeting/checkpoint/logs/twohand_h5/linker/<run-name>/[left或right]/`。

```powershell
# 左右手独立训练：同一个双手 H5，分别优化各自的模型
python -m teleoperation hand train --input aligned.h5 --run-name separate_v2 --hand-side left --device cuda
python -m teleoperation hand train --input aligned.h5 --run-name separate_v2 --hand-side right --device cuda
# 左右手使用各自的模型进行推理，输出为angle.h5
$leftCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/separate_v1/left/model_best.pth'
$rightCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/separate_v1/right/model_best.pth'
python -m teleoperation hand export --input aligned.h5 --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint --output angles_separate.h5 --device cuda

# mediapipe实时推理
python -m teleoperation hand realtime --frames 300
python -m teleoperation hand realtime --visualize --frames 300

#先把这两行输入进去再执行下面的命令
$leftCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/separate_v1/left/model_best.pth'
$rightCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/separate_v1/right/model_best.pth'
#实时推理，这里是视觉输入和fk的对比画面
python -m teleoperation hand realtime --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint --visualize --frames 0

# 原生机器人动作文件，以及 L21 角度到原装手的回放
# action.h5是任务动作的录制文件，angle.h5是输出手部角度的文件
python -m teleoperation replay actions --file actions.h5 --robot h1_2 --no-render
python -m teleoperation replay native-hand --file angles.h5 --robot h1_2 --hand both
python -m teleoperation replay l21-hand --file angles.h5 --headless-replay
#这一条暂时用不上
python -m teleoperation replay mounted-hand --file angles.h5 --robot g1 --hand both

# 仿真任务与关节演示
python -m teleoperation sim run --task pushcube --robot h1_2 --no-render --no-record
python -m teleoperation sim demo-joints
```

`--hand-side left/right` 仅使用所选手的有效时间窗口和对应 URDF/FK、碰撞损失；学习率调度、早停和最佳权重按该手验证损失判断。模型、CSV 和日志在实验目录的 `left/`、`right/` 下分别保存。省略参数或选择 `shared` 沿用共享训练及原输出目录。默认从头训练；`--init-checkpoint` 只初始化模型权重，可使用同侧或旧共享模型，不恢复优化器和训练轮次。单侧权重带有手侧标记，不能接到另一侧或作为双手共享模型。

推理时使用 `--checkpoint` 或同时提供 `--left-checkpoint`、`--right-checkpoint`，两种方式不能混用。双模型参数会关闭默认共享权重路径。摄像头实时需要两份均与 palm-local 坐标契约匹配的权重；用于训练和导出的 H5 坐标声明也必须与权重一致。独立训练保留左右手各自的输入坐标，不增加镜像变换。正式训练后应在相同验证帧上分别比较两手的损失和 FK 姿态，确认效果。

双臂链路使用明确的 TRON2A 标定和 URDF：

```powershell
python -m teleoperation dual export --observations observation.h5 --angle-h5 angles.h5 --calibration configs/tron2a_dach_calibration.example.json --output commands.h5 --urdf robot.urdf
python -m teleoperation dual realtime --adapter my_input:factory --calibration calibration.json --urdf robot.urdf
python -m teleoperation dual realtime --adapter my_input:factory --calibration calibration.json --hand-preprocessing palm-local --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint
python -m teleoperation dual replay --command-h5 commands.h5 --calibration calibration.json --urdf robot.urdf
```

`dual realtime` 的 factory 返回 `InputSource` 或原始组合观测迭代器，具体字段见 [contracts](docs/contracts.md)。示例标定文件需要结合实际设备填写，不代表已经完成实体标定。
MediaPipe normalized 输入使用 `--hand-preprocessing palm-local --checkpoint <palm-local checkpoint>`；默认 `legacy` 继续配对 legacy checkpoint。adapter 必须明确声明原始输入空间或已对齐坐标，详见上述 contracts。

检查与展示统一使用 `tools`：

```powershell
python -m teleoperation tools --help
python -m teleoperation tools show-robots
python -m teleoperation tools show-all-hands --file angles.h5
python -m teleoperation tools check-camera
python -m teleoperation tools check-environment

# 读取双模型并验证左右手 FK、损失和梯度（不训练、不修改权重）
python -m teleoperation tools verify-p0 --input aligned.h5 --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint

# 在同一帧比较共享模型和左右手独立模型的 FK
python -m teleoperation tools compare-training-hand-pose --input aligned.h5 --frame 100 --before-checkpoint shared_model.pth --after-left-checkpoint $leftCheckpoint --after-right-checkpoint $rightCheckpoint --output picture/separate_compare.png
```

`compare-training-hand-pose` 的 Before 和 After 分别支持 `--before/after-checkpoint` 或对应的 `--before/after-left-checkpoint`、`--before/after-right-checkpoint`。各组参数独立选择共享或双模型。

## 代码导航

| 要修改的内容 | 位置 |
|---|---|
| 观测、结果、动作类型与校验 | `contracts/` |
| 相机、MediaPipe、Vision Pro、逐帧回放 | `inputs/` |
| 坐标、拓扑、跟踪、窗口、手部模型与 FK | `retargeting/hand/` |
| 标定、双臂目标、IK、安全控制 | `retargeting/arm/` |
| 原生手映射、机器人规格和动作编码 | `robots/` |
| 物理环境、加载、状态采集、任务 | `simulation/` |
| Dataset、loss、优化计算与训练报告 | `learning/` |
| H5/NPY/NPZ、checkpoint、标定、事务与记录 | `data/` |
| 完整工作流、循环、资源释放、同步 | `apps/` |
| 绘图、诊断报告显示 | `tools/` |

阅读主链路从 `apps/hand_processing.py`、`apps/hand_realtime.py`、`apps/dual.py` 和 `apps/simulation.py` 开始。[架构说明](docs/architecture.md)解释依赖方向，[contracts](docs/contracts.md)解释数据含义。

## 资源与当前能力

机器人资产位于 `assets/robots/from_teleopbench`，L21 位于 `assets/robots/l21/{left,right}`，标定示例位于 `configs`。默认路径由 `teleoperation.paths` 定位，用户传入的相对路径按当前工作目录解释。历史数据、checkpoint 和输出保持原位置，不自动迁移或重写。

MediaPipe 需要相应依赖、摄像头与模型文件，录制和实时命令默认使用项目内的 `datasets/hand_landmarker.task`；也可通过 `--model-asset-path` 指定其他路径。`show-hand` 沿用外部 `linkerhand_sdk` 的展示资源。TRON2A 的外部 URDF/mesh 沿用 `third_party/tron2-robot-description/...` 默认位置，也可显式指定 `--urdf`。

手部网络仍输出 18D，去除 root placeholder 后才是 L21 的 17 个执行关节。TRON2A/L21 命令是 48D，H1/GR1/G1 原生动作分别为 38/36/28D。当前没有经过验证的跨机器人手臂转换；原生手回放保持手臂中性姿态。完整实时整机闭环、硬件驱动与录制性能优化仍属于后续功能工作。

## 手动验证

目录迁移时只执行了语法、导入、CLI 帮助、包构建和架构边界检查。后续已执行 current-state 回归与实时坐标修复，实测结果和缺失资源见 [2026-10-09 验收报告](docs/current-state-validation-2026-10-09.md)。

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.architecture.test_boundaries -v
python -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render
python -m teleoperation replay actions --dummy --steps 120 --robot gr1_t2 --no-render
python -m teleoperation replay actions --dummy --steps 120 --robot g1 --no-render
```

完整测试中的 TRON2A 测试需要外部 URDF。可用环境变量 `TRON2A_TEST_URDF` 指定位置；资源缺失仍明确失败，不新增跳过规则。然后检查 CPU 小规模训练、初始化恢复、导出、H5 对齐事务、相机失效恢复和实际回放。

