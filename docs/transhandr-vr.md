# TransHandR 上游 Vision Pro 链路移植

更新：2026-10-15。TransHandR 是本工程的上游和重要参考来源。
本页将原版数值链路、现有接口适配和额外的执行防护分开记录。

## 上游行为与对应实现

| 上游位置 | 原版行为 | 当前实现 |
|---|---|---|
| `main_visionpro_realtime.py:1136` | `VisionProStreamer.get_latest()`，默认读取 `left_fingers` | 既有 `inputs.visionpro.VisionProInput` |
| 同文件第 1162 行起 | 从 25 个矩阵提取平移；左手 `(y,-z,x)`，右手注释分支 `(-y,z,-x)` | `applications.visionpro_processing.transhandr_vr_points` |
| 同文件第 1181 行起 | 减去腕部第 0 点 | 同上，无额外掌面旋转 |
| 同文件第 1187 行起 | 最近三帧 `np.stack(..., axis=1)` | `TransHandRVisionProPipeline` |
| 同文件第 1273 行起 | 转置为 `(1,3,25,3)`，固定缩放，转换 float32 | 同上，在栈叠之后缩放 |
| 同文件第 1285 行起 | `model.eval()`、`torch.no_grad()`、18 维原始角度 | `retargeting.hand.transhandr_vr.TransHandRVRRetargeter` |

默认保留原版的显式路由：**左侧传感器数据 -> 原版右侧 L21 模型/仿真手**。
使用 `--source-hand` 指定输入侧，`--hand` 指定目标机器人侧；两者含义不同。
右侧输入轴向来自原版相邻注释分支/离线转换，而完整数值对照验证的是原版默认左侧路线。
目标为左手时，单手与双手模式均在输出边界执行左手原生关节符号适配；模型原始 18D 输出保留作诊断，不改变上游模型或其输入。
新增同侧双手模式必须显式指定 `--hand both --hand-mapping same`，在输出边界适配左侧原生角度。

VR 输入不使用摄像头的 MediaPipe 21 点拓扑转换或三点掌面规范化。
双手输出边界复用 L21 关节符号适配，和摄像头输入处理无关。
既有摄像头权重不是这条上游 VR 链路的匹配权重。

## 与当前工程兼容的接口

- `VisionProInput(ip, side="left", streamer=None)` 保留原签名和默认行为；
  额外支持 `side="both"`，仍输出左右槽位明确的 `RawHandFrame`。
- `TransHandRVisionProPipeline.process_frame()` 返回 `(HandWindow | None, CanonicalHandFrame)`；
  窗口保留 `(3,25,3)`，来源仍声明 `visionpro`，坐标使用已有 `source_to_l21_xyz` 标识。
- `TransHandRVRRetargeter` 实现既有 `HandRetargeter.retarget()`，返回
  `dict[("l21",side), HandJointResult]`，每侧为原生 17D 独立角度；`last_raw_angles` 保留上游原始 18D。历史 VR H5 在序列化边界恢复其 18D 诊断布局。
- `VisionProReplayInput` 读取已交付 H5 的 `*_hand_transforms`、`*_valid` 和 `timestamps`；
  既有录制、裁剪和校验工具不需要改格式。

原版缺手/占位/非有限数据没有完整防护。本工程按现有接口要求清空缺失侧窗口，
恢复后需要三帧新数据；不做手侧身份重分配。这属于边界数据有效性适配。
进程共享字典和忙轮询改为当前工程的拉取接口，资源集中关闭，轮询上限通过 `--fps` 设置。
模型输入、变换公式、时间顺序与输出数值已与原函数做直接对照。

## 权重

上游原权重没有本工程的坐标元数据，因此在专用 VR 入口中显式选择上游 profile，
不修改原文件。必须包含 `model_pos`，参数严格加载；与摄像头坐标声明不匹配的权重会被拒绝。
原版 `strict=False` 改为 `strict=True`，避免参数缺失时以随机值继续运行。
本机现有 `thumb3/linker/model_final.pth` 已验证可以严格加载。

原始模型结果与最终仿真命令分层处理。此前新增的滤波、角速度限制和 L21 网格检查仍在
`scene.apply()` 之前执行；它们可能改变最终显示姿态，不能把其输出冒充原版原始模型输出。
具体范围见 [手部输出处理与防护](hand-command-safety.md)。
本移植不连接上游 CAN 控制进程，不启动真实机械手，也不从头显虚构肩肘数据。

## 使用

在仓库根目录、已配置 PYTHONPATH 与 D 盘缓存的终端运行：

```powershell
$py = 'D:\Anaconda\envs\teleoperation\python.exe'
$vrCheckpoint = 'D:\2026\code\TransHandR\TransHandR\checkpoint\models\thumb3\linker\model_final.pth'

# 先回放已有真机数据，检查原版 VR 模型输出
& $py -m teleoperation hand vr `
    --input-h5 datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 `
    --checkpoint $vrCheckpoint --source-hand left --hand right `
    --scene hands --device cpu --frames 120 `
    --report outputs/tmp_realtime/vr_replay.json

# 连接头显：打开 Tracking Streamer，使用它显示的地址
& $py -m teleoperation hand vr `
    --ip '<头显地址>' --checkpoint $vrCheckpoint `
    --source-hand left --hand right --scene hands --device cpu `
    --report outputs/tmp_realtime/vr_live.json
```

实时连接需要 `avp_stream`；H5 回放不需要它。`Ctrl+C` 或关闭 PyBullet 窗口停止。
默认 `--fps 30` 是拉取上限，并非对真实设备输出帧率的保证。
加 `--headless --fps 0` 可以执行无窗口离线验收。

## 同侧双手回放与实时输入

双手模式的对应关系固定为 **VR 左手 -> L21 左手，VR 右手 -> L21 右手**。
两侧使用一份严格加载的上游权重，分别推理；模型本身和 VR 数值预处理没有修改。
单手默认仍为左侧输入 -> 右侧输出，旧 `TransHandRVRRetargeter` 的签名和结果含义保持兼容。

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$env:PYTHONPATH = "$PWD\src"
$py = 'D:\Anaconda\envs\teleoperation\python.exe'
$vrCheckpoint = 'D:\2026\code\TransHandR\TransHandR\checkpoint\models\thumb3\linker\model_final.pth'
$session = Get-Date -Format 'yyyyMMdd_HHmmss'

# 弹出同一个 PyBullet 窗口，显示两只独立机械手；0 表示播放至文件结束。
& $py -m teleoperation hand vr `
    --input-h5 datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 `
    --checkpoint $vrCheckpoint --hand both --hand-mapping same `
    --scene hands --view front --device cpu --fps 30 --frames 0 `
    --report "outputs/tmp_realtime/vr_both_same_$session.json"
```

双手模式 `--source-hand` 省略时自动取 `both`；也可显式传 `--source-hand both`。
传入单侧来源或遗漏 `--hand-mapping same` 会在加载模型和创建窗口之前失败。
实时连接时，将 `--input-h5 ...` 替换为 `--ip '<头显地址>'`；其余双手参数相同。

`--view front` 将两只 L21 手整体绕世界 Z 轴旋转 -90°，掌面同向朝向位于 -Y 的相机，
手指朝上、左右并排；相机 yaw=0°、pitch=0°，观察中心位于掌部高度。
这是显示层刚体旋转，不修改左右手角度、VR 输入或 URDF。
基座重定位按根链接到惯性中心的变换计算，避免左手 URDF 的惯性偏移导致显示位置偏移。
`--view full` 保留原始基座朝向和斜视角；默认仍为 `full`。该视角也适用于摄像头的 L21 手部场景。

### 边界适配（不属于上游算法）

- 同一份观测同时携带左右侧，使用既有独立三帧队列。单侧缺失只重置该侧窗口，
  对应机械手保持上一条已接受命令；恢复后需要三帧新数据，另一侧继续输出。
- 上游 Linker 模型输出按右侧 L21 的原生关节顺序定义。核对实际上游 `hand_cfg`、
  右手 URDF 的轴/原点与本仓库右侧 FK 后，双手模式在输出边界调用
  `canonical_to_native_angles(raw, target_side)`：左手只翻转 1/4/7/10 四个 MCP roll 的符号，
  右手保持原值。这不修改输入、权重、checkpoint 元数据或原始预测。
- 分侧限位、滤波、速度限制与同手指间碰撞检查之后，写入两只手，再统一推进场景一次。
  手掌基座固定，不使用腕部绝对位姿驱动基座，也不覆盖两手互撞。
- 所有 H5 回放（包括旧单手模式）的执行层改用录制时间戳，要求有限且严格递增。
  `--fps` 仅限制播放循环速度，`--fps 0` 不再因程序运行更快而改变滤波/限速结果。
  旧单手模型原始角度不变，最终执行命令可能与过去使用墙钟的回放不同。
  实时输出仍使用单调时钟。

### 结果文件

`--report run.json` 默认同时生成 `run.h5`；可用 `--output-h5 another.h5` 指定逐帧结果路径，
也可单独使用 `--output-h5`。结果 H5 不覆盖已有文件，重复运行请使用新 session 名。
输入、权重、报告和结果路径不能重合。

结果采用独立诊断 schema `transhandr_vr_results_v1`，不是摄像头角度回放文件格式：

| 数据集 | 形状 | 含义 |
|---|---|---|
| `frame_ids` / `timestamps` | `(T,)` | 回放行索引/时间戳，时间单位秒 |
| `left/right_input_valid` | `(T,)` | 对应源侧本帧通过 VR 预处理有效性检查 |
| `left/right_model_valid` | `(T,)` | 本帧窗口已就绪并产生有效预测 |
| `left/right_raw_angles` | `(T,18)` | 原始上游模型输出，弧度 |
| `left/right_native_angles` | `(T,18)` | 目标侧适配后的角度，弧度 |
| `left/right_command_angles` | `(T,18)` | 最终仿真命令，弧度；缺手/预热保持上一姿态 |

每个输入帧都保留一行。缺手/预热的 raw/native 写 `NaN`，通过 model_valid 判断有效性，
不把保持姿态冒充新预测。结果按帧写入，不将完整录制累积在内存中；结束或异常时关闭资源。
单手结果只包含所选目标侧的数据集。

JSON 的 `routes` 为目标侧 -> 来源侧；`sides` 分侧统计输入有效、角度输出、缺手/预热帧数。
顶层 `angle_frames` 是至少一侧输出的输入帧数，不能将左右计数相加作为它的含义。
`hand_commands` 分侧记录限幅和碰撞受阻，`output_h5` 指向逐帧文件。

### 本机验证（2026-10-15）

- `devtools/verify_vr_both.py` 从实际上游 AST 提取左侧 worker；右侧仅激活相邻注释中的
  `right_fingers` 和 x/y/z 赋值，明确区别于上游已运行的左侧分支。
  从真机 H5 每侧抽取 5 个窗口、分别用 scale=1/1.7，合计每侧 10 组；
  两侧模型输入与原始输出最大差均为 0。
- 上游右手和本仓库右手的 FK 关节轴/原点逐项一致；左手输出符号适配后，
  17 个活动关节在公共参考系下的轴方向点积最小为 `0.9991503838621145`。
  参考系只用于验证，不进入 VR 输入处理。
- `visionpro_hand_arm_20261012_213812_live.h5` 前 120 帧通过实际双手 DIRECT 场景，
  左右各输出 118 帧；前两帧为窗口预热，结果与分侧统计成功保存。
- 测试覆盖不同两侧输入、目标侧符号适配、单侧缺手和恢复、保持命令、输出文件保护、
  时间戳失败时的资源关闭，以及 fps=0/30 下逐帧结果一致。
- `python -m unittest tests.test_transhandr_vr tests.test_coordinate_contracts -v`：28 项通过。
  `python -m unittest discover -s tests`：318 项，317 项通过、1 项跳过，含既有摄像头流程。
  本机日志与真机回放报告位于 `outputs/tmp_hand_command_guard/vr_both_*`。

数值一致性与 FK 检查不等同于已验证右侧输入的全部动作质量或头显现场效果。
当前仍复用原版单手训练权重；应结合逐帧 raw/native/command 对照检查实际手势，
不能以碰撞受阻后的静止姿态判断模型没有输出。

复现上游对照：

```powershell
& $py devtools/verify_vr_both.py `
    --input-h5 datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 `
    --checkpoint $vrCheckpoint `
    --report outputs/tmp_hand_command_guard/vr_both_upstream_parity.json
```

## 上游来源快照与验证

本机参考根目录：`D:/2026/code/TransHandR/TransHandR`。

| 文件 | SHA256 |
|---|---|
| `main_visionpro_realtime.py` | `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f` |
| `model/model_poseformer.py` | `9c80ba61d4465e63c7444347dbb4bb7f94e86fb976b884dc2b24c3155d94e8a3` |
| `dataset/data_collect/data_collect_visionpro.py` | `deb77448a16059a54e13d2a459c4d0805965614fefb263cf1235ffdbc80d8610` |
| `dataset/robot/l21_right/linkerhand_l21_right.urdf` | `7387e62b5f0e126b2a4915fff6c2230ecfcfd00e93f2f499e963095a082cbf01` |

本仓库目标 URDF 来源快照：

| 文件 | SHA256 |
|---|---|
| `assets/robots/l21/right/linkerhand_l21_right.urdf` | `4c7d580b7518cb58277020ca9a1d4ff69dafce694a76617795be4fab64b66ba7` |
| `assets/robots/l21/left/linkerhand_l21_left.urdf` | `cb265e5cad0e41bd02d9c60fe461c3b83645a7584cce175152328cc34bafc57a` |

直接抽取原版 `vision_pro_data_process` 函数，以现有真机录制中的三帧替代 streamer，
对照移植后的 `(1,3,25,3)` 输入，最大差为 0。
严格加载原版 `thumb3` 权重后，原版模型和移植模型的 18 维输出最大差为 0。
随机张量的模型对照也相同。该结果验证数值移植，不等同于已进行头显现场试验。

测试：`tests/test_transhandr_vr.py`，覆盖原轴向/腕原点、三帧顺序/缩放、缺手恢复、
现有输入接口、结果路由、严格权重检查、H5 回放和统一 CLI 的实际 L21 无头场景。

## 统一接口后的上游数值复核

来源根目录 `D:/2026/code/TransHandR/TransHandR`。实际执行上游源码模型与预处理函数，在录制 `visionpro_hand_arm_20261012_213812_live.h5` 上检查左右各10个窗口，尺度1和1.7：输入张量与原始输出最大差异均为0；右侧FK轴与offset逐项一致。左目标侧符号适配后的轴最小点积0.9991503838621145。结果为源码/数值一致性，不表示头显实时或动作质量验收。

| 来源 | SHA256 |
|---|---|
| main_visionpro_realtime.py | f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f |
| model/model_poseformer.py | 9c80ba61d4465e63c7444347dbb4bb7f94e86fb976b884dc2b24c3155d94e8a3 |
| checkpoint/models/thumb3/linker/model_final.pth | ffc4aeb4fd3f8c5b6687e7d5378da43f459480eee355e59a46eac65a34c8c895 |

相机新规范、训练和接口详见 [统一手部流程](hand-retargeting-unified.md)。VR不自动使用该相机归一化。
