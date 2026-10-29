# 独立手掌与腕旋转查看器

此入口显示放大 2.5 倍的独立 L21 五指手，不加载整机或机械臂。手指采用现有 dex-retargeting Vector IK，无训练 checkpoint、无几何手指求角。
各手指有独立颜色，橙色掌心与蓝色手背标记帮助辨认翻转；RGB 轴分别代表显示手根的 X/Y/Z。
位置固定，腕旋转幅度 1:1，无输出缩放或平滑；手指单独按 URDF 速度且最多 2 rad/s 下发。输入朝向估计本身的抖动会直接显示。

## 相机

只需要 Hands 模型，不需要 Pose 模型和训练 checkpoint：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
. ./devtools/dex_native_env.ps1
$python = $dexPython
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
& $python -m teleoperation.applications.wrist_viewer --source camera --side right --report "outputs/tmp_realtime/wrist_viewer_$session.json"
```

切换 `--side left` / `--side both` 检查另一侧或双手。
保持手掌稳定，10 个有效新样本后进入 tracking。无需肩肘入镜。
相机预览不镜像，避免把屏幕镜像与真实转动方向混淆。

## VR

复用已有带 `avp_stream` 的环境：

```powershell
. ./devtools/dex_native_env.ps1
& $dexPython -m teleoperation.applications.wrist_viewer --source visionpro --side both --ip '替换为头显地址或房间号'
```

相机和 VR 均使用已有 dex-native 环境，初始化 L21 Vector 优化器，不需要 dex 标定文件。
手指尺寸通过源掌长近三帧中位数与目标 URDF 掌长做比例匹配。放大的显示模型不参与 IK，优化仍使用物理尺寸。
VR 使用已有 `DexInputAdapter('visionpro')` 的专用左右侧处理；不将相机掌预处理用于 VR，也不修改 TransHandR 模型路线。
原始 VR 回放用 `--input-h5 <路径>` 替换 `--ip`。SDK 的 `(1,4,4)` 腕矩阵和 Z-up 定义沿用现有独立腕输入适配。
既有 TransHandR 手指三帧流程不被调用或修改。

相机 NPZ 回放：

```powershell
& $python -m teleoperation.applications.wrist_viewer --source camera --side right --input-npz outputs/tmp_realtime/wrist_20261029_223600.npz
```

回放以 `--fps` 指定的显示速率运行，不保证复现原始采样时长。

## 操作与判断

- 按 R：重新收集自然掌姿，以此作为零旋转。可在方块窗口或相机窗口操作。
- 按 Q / Esc：退出。关闭方块窗口或 Ctrl+C 也可结束。
- 平移手掌：方块中心应保持固定。
- 翻掌、俯仰、偏转：方块方向与 RGB 轴同步改变，无增益缩放。
- 回到对齐姿态：方块应回到初始朝向附近。
- 握拳/张手但不转掌：检查估计朝向是否明显受干扰。
- 遮挡手或 VR 停流：保持最后方向，终端显示 hold/waiting 原因。

报告保存输入参考系、初始四元数、显示四元数、相对初始旋转角和最近 6000 帧，`finger_ik` 保存后端、手侧求解状态及指关节输出。
显示角是三维相对旋转的总角度，不是欧拉角；不能拿它直接当某个单独轴角。
初始对齐与参考轴映射是为了方便观察，不是相机/头显到真实场景的外参标定。

## 撤销记录与验证

按用户要求，用 `a8c6087` revert 撤销上一轮 `8487354` 的标定保持、65%幅度、位置范围、碰撞拦截等修改，保留 `ece43b1` 的相机/VR腕观测。
这意味着旧机器人入口也恢复到上一步之前的状态；当前只使用本页的独立查看器，不用旧机器人入口验收。
撤销前提交 `8487354bfc0971d35b395eaa30227910a59dadef`，检查点 `checkpoint-20261029-2258-wrist-viewer`。
新增查看器前基线 `a8c6087`，检查点 `checkpoint-20261029-2258-wrist-viewer-base`。

本轮按用户要求不运行测试套件、不打开设备，不启动额外审查。只做语法和 `--help` 检查；真人显示效果待用户验收。

五指版本修改前提交 `4860b1d924d3fb9ef12056206f2e53bfb575d3ff`，检查点 `checkpoint-20261029-2306-dex-wrist-hand`。
现场修正：查看器的相机调用显式请求 `include_world=True`，避免公开接口默认省略 world_hands 导致 dex 拒绝输入；VR 和 NPZ 回放沿用各自的观测接口。修正前提交 `e394a5c`，检查点 `checkpoint-20261029-2310-viewer-world-input`。本轮只做语法检查，未运行设备或测试套件。
新增 `wrist_dex.py` 仅组合已存在的源专用适配器和 NativeVectorSolver；未改动其数值求解算法。腕旋转始终独立于手指优化，R 只重置腕零位。
