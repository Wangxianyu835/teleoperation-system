# 统一手部重定向：训练、控制与结果文件

本页描述当前实现。手型与算法分别选择：`--hand-model` 指机器人，`--backend nn|dex` 指方法；二者输出相同目标手的原生独立关节角。L21 是登记手型之一，旧模型内部的 18 维布局仅在历史权重、文件及诊断边界出现。

## 手型与坐标

执行 `python -m teleoperation hand list-models` 查看登记信息。当前规格为：

| 手型 | 每侧独立坐标 D | 展开关节数 | NN | dex | 几何 |
|---|---:|---:|---|---|---|
| l21 | 17 | 17 | 需目标侧权重 | 已接入 | 旧 L21 几何算法 |
| h1_2 | 6 | 12 | 需目标侧权重 | 已接入 | 未适配 |
| gr1_t2 | 6 | 11 | 需目标侧权重 | 已接入 | 未适配 |
| g1 | 7 | 7 | 需目标侧权重 | 已接入 | 未适配 |

维度取决于独立关节数；从动关节由 URDF 耦合关系展开。L21 的 17 个有名坐标中含 URDF 零范围坐标，其输出受限于零，不增加旧 root 占位。21D 的扩展测试只验证动态输出头；仓库尚无真实 21D 新手资产。添加真实手型必须登记 URDF、关节顺序、耦合、有效限位与速度、掌部参考点和人体任务点对应。

## 掌长归一化与换操作者

人体采集和对齐 H5 始终保存米制数据。先逐帧以腕点、食指 MCP、中指 MCP 建立掌部基，规范为右手表示，再扩展为 hand25。相机训练和新控制都在形成三帧窗口后执行同一函数：

1. 每帧减去其腕点 0；计算腕点到中指 MCP 11 的距离。
2. 取当前三帧距离的中位数，作为该侧整个窗口共用的分母。
3. 输出 `(3,25,3)` 无量纲点，训练目标取窗口最新帧。

新 profile 是 `camera_palm_length_window3_v1`，坐标声明是 `mediapipe_world_wrist_index_middle_canonical_right_palm_length_window3_v1`，`coordinate_frame=hand_canonical_right`，`units=length_unit=1`。分母另存为米制诊断 `palm_length_m`。不使用整段统计、未来帧、首个张开手姿态或操作者永久标定。非有限、退化掌部或任一帧掌长不大于 `1e-6 m` 时，该侧无效并清空历史；连续三帧有效后恢复。两侧独立。

换操作者不要求先校准掌长，但仍依赖检测质量及训练覆盖范围。归一化去除了整体手大小这一因素，不能消除指长比例、遮挡和不同人的动作差异；跨人效果需要独立操作者数据验收。

## 公共结果与调用顺序

```text
MediaPipeCameraWorkflow / TwoHandH5Dataset
  -> MediaPipePalmLocalProcessor -> TemporalBuffer
  -> normalize_hand_window
  -> HandWindow
  -> HandRetargeter.retarget
  -> dict[(hand_model,side), HandJointResult]
  -> JointCommandPipeline.update
  -> HandJointCommand
  -> HandExecutor.apply -> scene.apply_native -> PyBullet
```

`HandJointResult.positions[D]` 是 rad，顺序为 `spec.independent_joint_names`；字段还包括 hand_model、side、spec_hash、timestamp、valid、error、solve_ms。无效结果不伪装成零角度。`HandJointCommand` 表示已接受、耦合展开、具名的原生执行命令，与原始解分开。

`HandSpec` 从目标资产生成身份哈希，折回从动关节的限位与速度约束，并在有效界内零位计算固定参考掌长。人体归一化任务向量转换到目标手根坐标后乘机器人参考掌长，得到米制目标。新 NN 用可微 FK 计算任务向量；dex 使用同一目标。

## 训练与保存路径

先录制 world 数据，再对齐；不要给已对齐 H5 再做对齐：

```powershell
$py = 'D:/Anaconda/envs/teleoperation/python.exe'
$env:PYTHONPATH = "$pwd/src"
& $py -m teleoperation hand align --input raw_world.h5 --output aligned_world.h5 --landmark-space world
& $py -m teleoperation hand train --input aligned_world.h5 `
  --hand-model l21 g1 --run-name experiment01 `
  --checkpoint-root outputs/hand_retargeting/models --device auto --epochs 100
```

同一人体 H5 与数据划分用于所有目标，依次训练，每个目标手型生成一份 shared 权重，同时汇合左、右有效窗口。模型在 canonical right 角度语义下训练，运行时为左手转换原生关节方向；`--checkpoint-root` 指向新版目标目录根，`--input-profile` 明确选择掌长规范。`--init-checkpoint` 只接收同手型、shared 输出、同输入和输出定义的新权重，只初始化参数，不恢复优化器；与 `--resume` 互斥，架构从 checkpoint 严格重建。输入仅接受机器人无关的 canonical world 米制 H5，旧图像坐标或机器人根坐标文件需要从原始 world 数据重新对齐，不能改标签冒充新格式。

每个模型路径固定为：

```text
  outputs/hand_retargeting/models/<hand_model>/shared/nn/
  camera_palm_length_window3_v1/<run_name>/
    model_best.pth
    model_last.pth
    config.json
    dataset.json
    metrics.jsonl
    training.log
```

目录存在时拒绝覆盖。数据哈希、原始帧划分及有效窗口数存 dataset.json；模型结构、关节语义、预处理、优化配置存 config.json。checkpoint 原子替换，schema=`hand_joint_nn_v1`，必须严格匹配目标手、侧、规格哈希、有序关节、输出单位、模型结构和掌长规范。不会自动寻找“最新权重”，也不会将旧 18D 权重标记成新版。

新主损失是归一化指尖任务向量距离的 SmoothL1，delta=0.02；弱中性姿态项为非零范围坐标的 `((q-q_reference)/range)^2` 均值，权重 `1e-4`。保留 AdamW、ReduceLROnPlateau 和梯度裁剪配置。原六项 L21 loss 仅用于历史诊断，不是新训练损失。

断点续训必须是单手型、单目标侧，同目录、同 run/config/data：

```powershell
& $py -m teleoperation hand train --input aligned_world.h5 --hand-model l21 `
  --run-name experiment01 --epochs 150 `
  --resume outputs/hand_retargeting/models/l21/shared/nn/camera_palm_length_window3_v1/experiment01/model_last.pth
```

epochs 表示总轮数，需超过已完成轮数。恢复网络、优化器、调度器、随机状态及早停状态；epoch 排列由 seed 和 epoch 决定。短训练产物只用于流程检查，不能宣称是可用的跨人模型。

## 实时选择方法

```powershell
$shared = 'outputs/hand_retargeting/models/l21/shared/nn/camera_palm_length_window3_v1/experiment01/model_best.pth'
& $py -m teleoperation hand control --hand-model l21 --backend nn --hand both --checkpoint $shared `
  --camera-index 0 --model-asset-path datasets/hand_landmarker.task

# 同一手型改用 dex，无需权重或操作者掌长标定
. ./devtools/dex_native_env.ps1
& $dexPython -m teleoperation hand control --hand-model l21 --backend dex --hand right
```

双侧新 NN 使用一个 shared `--checkpoint`。控制在打开相机/GUI 前校验目标与权重，不回退到另一算法。Ctrl+C 或关闭 PyBullet 窗口停止；`--frames N --headless` 可用于有限运行。可用 `--record-results <new.h5>` 记录原始解与命令。

所有方法共用独立坐标滤波、有效限位、`min(CLI,URDF折回速度)` 限速，之后展开耦合。L21 上限为 1.0 rad/s，CLI 默认 1.5 不会覆盖该约束。丢手保持接受命令；恢复首步的时间预算不超过 1/30 s，若距上一条保持命令的实际间隔更短，则使用该实际间隔。L21 仿真保留网格路径检查；其他手没有等价运行时检查。默认低通 alpha=0.2；`--joint-filter none` 仍有限速。`--no-collision-guard` 仅用于明确关闭 L21 诊断检查。

## 导出、检查、回放、比较

```powershell
& $py -m teleoperation hand export --input aligned_world.h5 --output outputs/nn_results.h5 `
  --hand-model l21 --backend nn --hand right --checkpoint $right
& $dexPython -m teleoperation hand export --input aligned_world.h5 --output outputs/dex_results.h5 `
  --hand-model l21 --backend dex --hand right
& $py -m teleoperation hand inspect --angle-h5 outputs/nn_results.h5
& $py -m teleoperation replay hand-joints --input outputs/nn_results.h5 --hand-model l21 --hand right
& $py devtools/compare_hand_methods.py --nn outputs/nn_results.h5 --dex outputs/dex_results.h5 `
  --hand-model l21 --hand right --output outputs/method_comparison.json
```

新 schema `hand_joint_results_v1` 保留完整时间线，包括前两帧预热和缺手。每个 `<hand_model>/<side>` 组保存 raw_positions[D]、command_positions[展开数]、valid、error、solve_ms、归一化及米制任务向量和分母诊断。回放验证规格、关节顺序、限位与耦合，直接执行 accepted commands，不再次滤波。写入采用独占创建，不覆盖已有文件。

比较必须使用同一输入哈希、profile、命令策略和目标向量，按时间戳及共同有效帧对齐。分别报告原始解/命令的归一化与米制指尖误差、捏合距离误差、限位、命令速度、无效比例和耗时。无效比例包括预热/缺手，不能直接称为数值求解失败率。该工具不证明跨操作者泛化；应按人划分训练/验证/测试，再做真人操作验收。

## 历史链路边界

旧 camera L21 NN 仍按真实权重规范选择 `--input-profile legacy_l21_world` 或 `legacy_l21_normalized`，只允许 L21。单侧显式 `--checkpoint`；单个权重用于双侧时必须明确声明 shared。几何方法仅支持 L21 与 legacy_l21_normalized，不自动套用新 NN 预处理。

旧 `hand realtime`、三机器人 L21 映射显示、TRON2A 双臂工作流仍为历史配置入口，不能加载新版 D 维权重。它们输出也已迁入公共结果类型；L21 到其他原装手的旧映射仍是有损诊断，并非新 dex/NN 原生重定向。

VR 保持独立 `hand vr` 入口和实际上游预处理；不对 VR 套用相机掌长规范。上游右手语义输出在边界转换为目标侧原生 L21 17D，单手左侧与双手左侧一致；`last_raw_angles` 和历史 VR H5 保留原始 18D 诊断。旧 VR dex 仍要求其明确校准。历史 L21/dex 文件由明确旧 schema 读取器处理，不复活旧公共结果包装。

数值移植来源、路径与哈希见 [计划执行记录](superpowers/plans/2026-10-29-hand-retargeting-unified.md) 和 [VR 说明](transhandr-vr.md)。
