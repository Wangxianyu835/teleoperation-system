# 相机手部链路阅读说明 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让读者能沿相机 NN 手部链路追踪数据、识别对象状态，并判断修改影响。

**Architecture:** 保留现有文件、公共接口、数值计算和状态生命周期。新增一份面向使用者的中文说明，在关键封装和库调用处补注释，并纠正 forward 输出说明。

**Tech Stack:** Python、NumPy、PyTorch、unittest、Git。

**Spec:** 本次会话确认的 Q2、Q4—Q7：先讲清相机到关节命令的链路；解释库调用和类；保留历史状态类；第一轮只修改说明。

## Global Constraints

- 修改前遵循 AGENTS.md 创建 annotated checkpoint；完成后本地提交，不推送。
- 公共输入、窗口、结果接口及数值算法保持兼容；本次不修改可执行语句。
- 不将相机处理解释为上游 TransHandR VR 算法，不更改权重元数据。
- 读者能读 Python 基础，但需要明确库调用的作用、类的状态及调用关系。

## Task 1: 一条可对照源码的阅读链路

**Files:**
- Create: `docs/hand-camera-reading-guide.md`
- Modify: `README.md`，增加阅读入口。
- Modify: `applications/hand_control.py`、`applications/hand_realtime.py`、`applications/hand_processing.py`（均位于 `src/teleoperation/`）。
- Modify: `src/teleoperation/retargeting/hand/` 下的 `temporal.py`、`normalization.py`、`interface.py`、`joint_commands.py`、`transformer.py`；仅注释和 docstring。
- Test: `tests/test_unified_hand_inputs.py`、`tests/test_joint_commands.py`、`tests/test_hand_control_cli.py`。

**Interfaces:**
- Consumes: 既有 `RawHandFrame`、`CanonicalHandFrame`、`HandWindow`、`HandJointResult`、`HandJointCommand`。
- Produces: 阅读说明与准确注释；无新运行时接口。

- [x] 核对现有源码：入口、shape/单位、缓冲与缺失状态、NN 与命令处理、相关测试。
- [x] 检查工作区并创建修改前检查点。
- [x] 写阅读说明，包含真实调用图、状态表、库函数解释、修改影响和验证命令；示意值明确标记。
- [x] 补关键中文注释并纠正 `PoseTransformer.forward` 的输出描述。
- [x] 对比检查点与当前 Python AST，忽略 docstring 后完全一致；检查差异空白与文档链接。
- [x] 使用已安装环境运行相关 unittest；只改注释，不新增镜像实现的测试。
- [x] 复核改动范围；按下述结果随文档一并本地提交，提交标识见包含本计划的 Git 提交。

本会话已有实施授权，直接在当前会话执行以上任务。

## 验证记录

- 基线：`9c65b6bc218d1bb79282c5fc84778df7396c7692`。
- 检查点：`checkpoint-20261107-2333-hand-reading-guide`。
- 8 个 Python 文件去掉 docstring 后 AST 一致；说明中的 12 个相对链接均能找到目标。
- `D:\Anaconda\envs\teleoperation\python.exe -m unittest tests.test_unified_hand_inputs tests.test_joint_commands tests.test_hand_control_cli -v`：21 项，20 通过，1 项跳过（需要 dex 独立环境），耗时 29.573 秒。
- 只变更注释、docstring 与文档；未运行全套测试，未进行真实摄像头/VR/实体设备验收。
