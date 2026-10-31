# SMPLer-X 相机机械臂链路

相机机械臂输入使用 SMPLer-X S32。MediaPipe Hands 只提供手指数据。
一次 RGB 采集同时交给 Hands 和独立工作进程。两者保留各自的采集时间。
身体结果通过跟踪器进入现有重定向、腕姿态对齐、IK 和安全控制。
机器人配置位于控制层，感知结果不绑定 TRON2A。

代码迁移与真实相机验收分开记录。遮挡、快速动作和现场抖动改善仍需实际视频确认。
本机 S32 的性能没有达到 10 Hz / p95 150 ms 的工程目标。

## 环境和资产

应用使用 `D:/Anaconda/envs/teleoperation/python.exe`。主环境和 dex 环境没有为模型安装升级依赖。
独立工作进程使用 Python 3.10、torch 1.13.0+cu117、torchvision 0.14.0+cu117、
mmcv-full 1.7.1。实际版本见 `external/smplerx_worker/environment.lock.yml`。

| 内容 | 本机默认目录 |
| --- | --- |
| 官方源码 | `.venv/smplerx-assets/SMPLer-X` |
| S32 权重 | `.venv/smplerx-assets/smpler_x_s32.pth.tar` |
| 人体模型 | `datasets/smpl_models` |
| 工作进程 | `.venv/smplerx-worker/Scripts/python.exe` |

源码和权重已从临时目录迁出。以上目录不进入 Git，重新 clone 后需单独准备。
四个路径可用 `--smplerx-root`、`--smplerx-checkpoint`、`--smplx-model-dir`、
`--smplerx-worker-python` 覆盖。
用户提供的 SMPL v1.1.0 原始文件保留原名；临时视图提供上游命名。
SMPLX 文件不能替代 SMPL 文件。

官方源码：[MotrixLab/SMPLer-X](https://github.com/MotrixLab/SMPLer-X)，版本
`064baef0e4ab5277a3297691bc1d46ea5412586f`。
S32 权重 SHA-256：
`1c78569f4e7b6e22e7166b55f322e5d5e019073b8d1a430a6926f93b4c86aa62`。

## 输入、时间和安全策略

- 身体位置使用 SMPL-X 根相对米制坐标。图像点为逆裁剪后的投影。
- 腕旋转来自完整父子旋转链，然后相对操作者中性姿态对齐机器人。
- 相机轴向映射是配置值，系统不自动估计外参或手眼标定。
- 体型取五个稳定预测的中位数，然后固定；原始预测仍保存在模型结果包中。
- 无检测、单侧异常、重复包和过期结果使对应侧保持。源过期阈值为 0.5 秒。
- 大幅跳变需三个不同且相互一致的新样本才能恢复；目标和关节仍受既有速率限制。
- 实际丢失后，即使回到原位置也需三个新样本恢复。等待异步结果不计入样本数，
  也不会清空尚未完成的稳定腕标定；真正失效或过期会清空该窗口。
- 人体检测分数阈值为 0.3。多框时直接选与上一帧操作者 IoU 最大的框，同分按置信度选择；
  初次启动按检测置信度选择。多框歧义不再触发保持或三帧身份恢复等待。
  离场超过两秒、检测框突然换位仍锁定身体输入，需结束运行、重开会话并重新标定。
  这是空间连续性选择，不是人体身份识别。
  规则与真实故障帧验证见 `docs/smplerx-operator-recovery.md`。
- 会话或模型身份改变会重置中性姿态与腕对齐。相机处理不应用于 VR 输入。

已移除 MediaPipe Pose 解析、旧机械臂 NPZ/H5 reader、旧中性标定读取及 Hands 到机械臂的回退。
旧文件需重新采集，磁盘上的用户旧数据没有被自动删除。
`--input-npz` 只支持手指调试；包含旧机械臂或腕字段时拒绝读取。

## 运行、录制和回放

在仓库根目录执行：

~~~powershell
$python = 'D:/Anaconda/envs/teleoperation/python.exe'
$env:PYTHONPATH = "$pwd/src"

# 三台机器人；可加 --robot g1 选择一台。
& $python -m teleoperation.simulation.realtime_teleop `
  --backend geometric --side both --arm-map posture --arm-calibrate-mode hold `
  --dump-body outputs/session/body.h5 --report outputs/session/live.json

# 先等待工作进程返回，再录制指定数量的相机样本。
& $python -m teleoperation hand record-world --frames 300 `
  --output outputs/session/capture.h5 --max-fps 5

# 无相机、无 S32 推理的回放。
& $python -m teleoperation.simulation.realtime_teleop `
  --input-body outputs/session/body.h5 --backend geometric `
  --headless --no-preview --report outputs/session/replay.json

& $python -m teleoperation.applications.arm_validation `
  --input-body outputs/session/body.h5 --output-dir outputs/session/validation
~~~

首次加载工作进程约需十余秒。期间手指可继续处理，机械臂保持。
手指也可使用已有 checkpoint 或 dex。`hand record-world --no-arms` 只录制 Hands。
TRON2A 的 `dual calibrate-camera` 和 `dual camera` 使用新感知源，仍要求显式安全关节、
工作空间及真实中性标定。合成标定不能用于实时控制。

组合 H5 使用 `smplerx_body_recording_h5_v1`。每个样本保存无损压缩 RGB、
新工作进程结果包、最终身体/手指/腕观测、序号、独立时间、源年龄、状态和 fresh 标志。
根层手部表供现有训练/对齐工具读取。手部对齐保留身体包，不重新解释身体坐标。
回放按原始采集间隔控制。有效率只计不同的新结果。
实时控制也使用相邻采集时间差，二者不使用主机循环耗时的滚动中位数。
录制器在保存前持有本次数据，长录制会占用内存和磁盘。

实时和回放共用 `apply_camera_arm_frame`。它连接 bridge、腕控制和现有 IK driver；
TRON2A 专用控制器继续使用原有安全接口。重定向方法没有改成向量优化。

## 上游适配及验证

使用官方网络、检测器、裁剪与 `SMPLer_X.get_model('test')`，保留 Demoer 权重键转换；
缺少学习参数时启动失败。跳过仅用于训练/评估的数据集导入。
先导入 MMDetection，再注册自定义 MMPose。
torchgeometry 0.1.2 布尔减法通过检查过的内存 shim 改为布尔取反。
上游源码和安装包没有被修改。

| 上游文件 | SHA-256 |
| --- | --- |
| `main/SMPLer_X.py` | `ad3215d700e917a9a6a818500bca853d1bd4d647778f24e78625a9554b5ebcf8` |
| `common/utils/preprocessing.py` | `a6aedd2a8fef22f8c0e3d12ed65b0af6b4d1aff30e187659e0e3a9fae5b60ebb` |
| `common/utils/human_models.py` | `343cf180b90e65e272817057ea04d0941d9d54eb6ec27450b96088384ad0cd3e` |

四个真实人体模型姿态的 FK 对照：位置误差 0 m，旋转矩阵分量最大误差 1.79e-7。
先前直接调用与子进程对照最大误差：位置 2.98e-8 m、旋转 7.90e-7、图像投影 5.96e-8。
这些检查验证接线，不能证明真实空间估计精度。

五分钟单模型静态图片测试见 `docs/smplerx-s32-endurance.json`：
1,604 个新结果、5.34 Hz、p50 187 ms、p95 234 ms；显存峰值 allocated 662 MiB、
reserved 728 MiB；测量期间 allocated 增长 0。没有相机、Hands 或机器人负载，性能目标未达到。

`devtools/verify_smplerx_runtime.py` 支持资产检查、图片 smoke、`--input-video`、
`--samples 300` 和 `--duration-s 300`。视频按序读取，不重复末帧补计。
`devtools/verify_smplerx_roundtrip.py` 检查真实进程、跟踪、H5 往返与图像投影。
现场仍需静止、单臂、双臂、交叉、遮挡、离场与恢复视频验收。

最终独立审查发现并修复了异步腕标定、丢失恢复和实时/回放时间差异。
另修复无手指输入的身体回放，以及姿态映射离线指标误用旧行程框的问题。
姿态模式检查测得的机器人臂长可达范围；位移模式继续检查原行程框，不放宽 IK 或关节安全限制。
40 个真实模型结果的 H5 往返误差为 0；正式三机器人回放完成，双腕进入 tracking；
对应离线诊断 15 项检查通过。输入是重复静态图片，不能用于证明动态或遮挡效果。
