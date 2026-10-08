# 遥操作与重定向项目

项目包含 L21 手部重定向、TRON2A 双臂 IK，以及 H1-2、GR1-T2、G1 的 PyBullet 任务和回放。生产代码统一位于 `src/teleoperation`。

**阶段汇报与答辩材料**：[present 成果包](present/README.md)，完整汇报正文见 [项目阶段成果](present/项目阶段成果.md)。

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

## 常用命令

```powershell
# H5 对齐：另存输出，保留原文件
python -m teleoperation hand align --input raw.h5 --output aligned.h5

# 训练、恢复初始化和导出
python -m teleoperation hand train --input aligned.h5 --run-name experiment --device cpu
python -m teleoperation hand train --input aligned.h5 --run-name experiment2 --init-checkpoint model_best.pth
python -m teleoperation hand export --input aligned.h5 --checkpoint model_best.pth --output angles.h5 --device cpu
python -m teleoperation hand inspect --angle-h5 angles.h5

# MediaPipe：原始点 -> palm-local -> 三帧窗口 -> 18D 结果
python -m teleoperation hand realtime --frames 300
python -m teleoperation hand realtime --visualize --frames 300

# 原生机器人动作文件，以及 L21 角度到原装手的回放
python -m teleoperation replay actions --file actions.h5 --robot h1_2 --no-render
python -m teleoperation replay native-hand --file angles.h5 --robot h1_2 --hand both
python -m teleoperation replay l21-hand --file angles.h5 --headless-replay
python -m teleoperation replay mounted-hand --file angles.h5 --robot g1 --hand both

# 仿真任务与关节演示
python -m teleoperation sim run --task pushcube --robot h1_2 --no-render --no-record
python -m teleoperation sim demo-joints
```

双臂链路使用明确的 TRON2A 标定和 URDF：

```powershell
python -m teleoperation dual export --observations observation.h5 --angle-h5 angles.h5 --calibration configs/tron2a_dach_calibration.example.json --output commands.h5 --urdf robot.urdf
python -m teleoperation dual realtime --adapter my_input:factory --calibration calibration.json --urdf robot.urdf
python -m teleoperation dual replay --command-h5 commands.h5 --calibration calibration.json --urdf robot.urdf
```

`dual realtime` 的 factory 返回 `InputSource` 或原始组合观测迭代器，具体字段见 [contracts](docs/contracts.md)。示例标定文件需要结合实际设备填写，不代表已经完成实体标定。

检查与展示统一使用 `tools`：

```powershell
python -m teleoperation tools --help
python -m teleoperation tools show-robots
python -m teleoperation tools show-all-hands --file angles.h5
python -m teleoperation tools check-camera
python -m teleoperation tools check-environment
```

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

MediaPipe 需要相应依赖、摄像头与 `hand_landmarker.task`。`show-hand` 沿用外部 `linkerhand_sdk` 的展示资源。TRON2A 的外部 URDF/mesh 沿用 `third_party/tron2-robot-description/...` 默认位置，也可显式指定 `--urdf`。

手部网络仍输出 18D，去除 root placeholder 后才是 L21 的 17 个执行关节。TRON2A/L21 命令是 48D，H1/GR1/G1 原生动作分别为 38/36/28D。当前没有经过验证的跨机器人手臂转换；原生手回放保持手臂中性姿态。完整实时整机闭环、硬件驱动与录制性能优化仍属于后续功能工作。

## 手动验证

本轮按要求直接迁移，没有使用冻结基线进行比较，也没有执行迁移后的训练、推理或物理回归。已完成的检查范围是语法、模块导入、CLI 帮助、包构建和静态依赖边界。数值一致性及硬件行为需要后续验证。

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.architecture.test_boundaries -v
python -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render
python -m teleoperation replay actions --dummy --steps 120 --robot gr1_t2 --no-render
python -m teleoperation replay actions --dummy --steps 120 --robot g1 --no-render
```

完整测试中的 TRON2A 测试需要外部 URDF。可用环境变量 `TRON2A_TEST_URDF` 指定位置；资源缺失仍明确失败，不新增跳过规则。然后检查 CPU 小规模训练、初始化恢复、导出、H5 对齐事务、相机失效恢复和实际回放。

旧根脚本与旧 Python 包入口已退役。旧文档保存在 [docs/archive](docs/archive)，其中的历史命令仅供追溯。
