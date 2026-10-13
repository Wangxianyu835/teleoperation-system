# TransHandR 上游 Vision Pro 链路移植

更新：2026-10-14。TransHandR 是本工程的上游和重要参考来源。
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
切换目标侧只改变结果槽位，不会自动重训、镜像或修正权重。

这里不使用摄像头的 MediaPipe 21 点拓扑转换、三点掌面规范化或左右规范角度适配。
既有摄像头权重不是这条上游 VR 链路的匹配权重。

## 与当前工程兼容的接口

- `VisionProInput(ip, side="left", streamer=None)` 保留原签名和默认行为；
  额外支持 `side="both"`，仍输出左右槽位明确的 `RawHandFrame`。
- `TransHandRVisionProPipeline.process_frame()` 返回 `(HandWindow | None, CanonicalHandFrame)`；
  窗口保留 `(3,25,3)`，来源仍声明 `visionpro`，坐标使用已有 `source_to_l21_xyz` 标识。
- `TransHandRVRRetargeter` 实现既有 `HandRetargeter.retarget()`，返回
  `HandRetargetResult` 和 18 维 `L21HandAngles`；不改变动作向量布局。
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

## 上游来源快照与验证

本机参考根目录：`D:/2026/code/TransHandR/TransHandR`。

| 文件 | SHA256 |
|---|---|
| `main_visionpro_realtime.py` | `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f` |
| `model/model_poseformer.py` | `9c80ba61d4465e63c7444347dbb4bb7f94e86fb976b884dc2b24c3155d94e8a3` |
| `dataset/data_collect/data_collect_visionpro.py` | `deb77448a16059a54e13d2a459c4d0805965614fefb263cf1235ffdbc80d8610` |

直接抽取原版 `vision_pro_data_process` 函数，以现有真机录制中的三帧替代 streamer，
对照移植后的 `(1,3,25,3)` 输入，最大差为 0。
严格加载原版 `thumb3` 权重后，原版模型和移植模型的 18 维输出最大差为 0。
随机张量的模型对照也相同。该结果验证数值移植，不等同于已进行头显现场试验。

测试：`tests/test_transhandr_vr.py`，覆盖原轴向/腕原点、三帧顺序/缩放、缺手恢复、
现有输入接口、结果路由、严格权重检查、H5 回放和统一 CLI 的实际 L21 无头场景。
