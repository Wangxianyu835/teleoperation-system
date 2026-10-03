# P0/P1 验证报告（2026-10-03）

本轮完成软件修复和只读验证，最终全量 **91 项测试通过**。没有重训、硬件操作或覆写原有权重/录制；模型架构、损失权重、双臂 48D/Dex 协议保持原配置。物理坐标、尺度和模型质量仍需 [待验证清单](UNRESOLVED_VERIFICATION_PLAN.md) 中的实验。

## 完成内容

| 项目 | 行为与证据 |
|---|---|
| T1 指尖距离 | MSE 已平均 batch，仅平均七对指尖；批量复制不再缩小 loss/累计梯度。3 项回归测试 |
| T2 拇指角度 | 避免 acos 在 ±1 的奇异梯度；覆盖共线、反向、近共线、极小、零长度、普通姿态和非有限输入，float32/float64。9 项测试 |
| 训练安全门 | backward 前拒绝非有限 total loss；step 前复用 max_norm=10 clipping 的 `error_if_nonfinite=True`，拒绝非有限梯度和 norm 溢出。错误含 batch/global_step；5 项测试 |
| 坐标诊断 | 基向量、正交性、行列式/反射、双侧有向体积、源/FK 尺度传播；只读报告未知单位。12 项新增测试 |
| H5 对齐事务 | 校验后写临时文件并原子替换；失败保留旧输出；覆盖 root/group、属性、硬链接及 Windows 占用目标。12 项新增测试 |
| 默认资源 | 仓库根目录解析 aligned 输入和 `my_run` checkpoint；显式参数可覆盖。4 项新增测试 |

入口：`model/losses.py`、`retargeting/training.py`、`scripts/align_h5_coordinates.py`、`scripts/diagnose_hand_coordinates.py`、`scripts/verify_p0.py`。

## 数学与无效几何约定

旧 thumb-angle 对四个 segment 使用 `F.normalize`（默认 eps=1e-12），点积 clamp 到 [-1,1] 后 acos。归一化避免直接除零，却将零段变成零向量；acos 端点仍有奇异导数，有限 forward 不保证有限 backward。

新路径在归一化前检查所有 segment 有限；每个 source/target 角度的两段均须长于 1e-12（坐标单位）。零段没有物理角度，低于数值分辨阈值的段也不参与该项；只对有效行使用原 MSE，全部无效返回与输入计算图相连的零。其他损失及样本 mask 不变。

角度改为 `atan2(norm(cross(a,b)), dot(a,b))`。没有新增角度 epsilon 或偏移；精确 0/π 被保留，norm 在共线点使用有限零次梯度。近边界测试验证有限梯度，普通姿态与 acos 的值和梯度在浮点误差内一致。1e-12 是沿用的数值 cutoff，不是物理手指尺寸阈值；非有限 segment 抛错，不替换 NaN 后继续训练。

## 实测证据

```powershell
python -m unittest discover -s tests -v
python scripts/diagnose_hand_coordinates.py --input input/visual_hand_data_20260912_153542.h5
python scripts/diagnose_hand_coordinates.py --input input/aligned_visual_hand_data_20260912_153542.h5
python scripts/verify_p0.py --input input/aligned_visual_hand_data_20260912_153542.h5 --checkpoint checkpoint/models/twohand_h5/linker/my_run/model_best.pth
python -m retargeting export --device cpu --output output/autonomous_verification/default_runtime_angles.h5
```

全量由 T1 后 49 → T2 后 63 → 坐标审计 75 → 对齐事务 87 → 默认配置 91。现有 nanobind 退出告警未造成测试失败。各专项和全量命令实际执行过；需要修复的问题先由红色回归复现。

现有 best checkpoint 大小 186036461 bytes，SHA256：

```text
73de5f2d9481ecceff0725af7c094b818a707bbf6cdc0131d57f1e62c90af3bf
```

strict load 后每侧抽取三个有效窗口，经真实模型、L21 FK、六项 loss 和 backward：

| 侧 | 帧索引 | 输入 / 输出 / FK 形状 | 梯度张量 |
|---|---|---|---:|
| 左 | 25,280,536 | `[3,3,25,3]` / `[3,18]` / `[3,23,3]` | 134 |
| 右 | 25,279,534 | 同上 | 134 |

两侧输出、FK、loss、梯度均有限，根占位为零且角度未超限；只读审计不执行 optimizer step。真实默认导出共 557 行，左 512、右 510 行有效，其余行按既有 hold 策略输出；无非有限/超限角度。新对齐结果与已有 aligned 录制的双手数组完全一致。10 个原有 checkpoint/H5 的 SHA256 与基线一致。

详细 JSON/日志保存在 Git 忽略的 `output/autonomous_verification/`。另一 checkout 可用上述命令和相同本地资源复现；录制和权重不随本次提交上传。

## 历史指标解释

旧 best loss 约 1020.56（epoch 80）不能与修复后 total 直接比较：T1 原来又除以有效侧 batch 大小，修复只改变这一分量，mask 和最后 batch 使倍数变化。不能用固定乘数转换旧总损失。T2 在普通有效姿态等价，边界和退化行为明确改变。抽样 loss 不是物理精度；重训前需按新目标重评旧 checkpoint，并完成独立录制与坐标/尺度验证。
