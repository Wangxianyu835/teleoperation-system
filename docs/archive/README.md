# 文档入口

**汇报与答辩**：[present 阶段成果包](../present/README.md)包含成果正文、10 张图片、4 段短视频、
GIF 预览和素材来源说明，可整体复制用于离线展示。

## 最终方向

完成“真实人类输入 → 手部/双臂重定向 → 明确的机器人命令与关节映射 → 仿真任务 →
同步记录 → 可复现评估”的双臂灵巧手遥操作闭环。手部 production 实现统一维护在
teleoperation-system；TRON2A + L21 命令与 H1-2 / GR1-T2 / G1 benchmark 动作通过
明确的消费端转换连接。模块存在、能显示模型、能播放 synthetic 动作，不等于真实闭环已验收。

## 建议阅读顺序

| 文档 | 内容与适用范围 |
|---|---|
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | **先读**：当前总体架构、重点端到端链路、逐文件职责、人工验证、已确认问题及后续方向 |
| [ENVIRONMENT.md](ENVIRONMENT.md) | 正式 Conda teleoperation 环境、重建方式、ENV-1 版本盘点与链路验收 |
| [HAND_RETARGETING.md](HAND_RETARGETING.md) | 正式 hand pipeline、数据/checkpoint 契约、安装分类、训练、导出、实时与 native-hand 回放 |
| [PR2_FOLLOW_UP.md](PR2_FOLLOW_UP.md) | command/order/validity、机器人映射和双臂整合的 PR2 工作边界 |
| [PR15_APPLICATION_ENTRY_RESULT.md](PR15_APPLICATION_ENTRY_RESULT.md) | PR1.5 入口修复、回放数值回归、正式环境 tests/smoke 与保留问题 |
| [PR1_HAND_CONSOLIDATION_RESULT.md](PR1_HAND_CONSOLIDATION_RESULT.md) | PR1 的固定源版本、迁移文件、测试和 smoke 记录；是当次验收报告 |
| [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) | 较早的系统接口说明；benchmark action 维度/顺序仍需参考，当前 hand 契约以 hand 文档为准 |
| [OFFLINE_PIPELINE.md](OFFLINE_PIPELINE.md) | 早期离线协作规划与契约 G/H；其中单手数据格式不是当前 canonical 两手 H5 |
| [TEAM_ONBOARDING.md](TEAM_ONBOARDING.md) | 早期团队协作、Git 和上手流程；环境/路径/文件数量以当前架构说明为准 |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | 2026-09-12 的历史背景、决策与论文阅读记录；保留历史，不作为当前完成状态的清单 |

文档放在本目录，仓库根 [README](../README.md) 提供总入口。当前状态更新写入
SYSTEM_ARCHITECTURE；hand 使用细节更新写入 HAND_RETARGETING；已经结束的 PR 验收报告
保留当时事实。新增人工验证结果应记录环境、输入/权重、命令、现象和是否通过。
