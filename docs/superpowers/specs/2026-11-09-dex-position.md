# 固定掌根 Dex Position 接入规格

用户要求：将 dex-retargeting 的 position-retargeting 加入当前重定向接口。
2026-11-09 已确认：固定掌根，只输出手指关节。本次交付为接入计划，功能尚未实现。

## 接口与范围

- 新增 `--backend dex-position`；现有 `nn`、`dex`、`transhandr` 的含义和默认值保持兼容。
- 输入仍为 `HandWindow`，每侧为已声明掌长归一化的 `(3,25,3)` 窗口或 `None`；优化使用最新一帧。
- 输出仍为 `dict[HandKey, HandJointResult]`，严格使用 `HandSpec.independent_joint_names` 顺序、有效上下限、spec_hash 和窗口时间戳。
- 支持已注册的 l21、g1、h1_2、gr1_t2，左右手及调用方选择的精确 model/side 集合。
- 覆盖 hand control/export、hand vr 的统一输入路由、robot control 及正式 Windows 启动器。
- 不新增自由根关节、手腕位姿字段、输入格式推断、自动后端回退或求解重试。
- 使用现有命令层做滤波、限速、失效保持和碰撞处理；Position 内部不再低通。
- 相机及统一 VR 使用各自既有来源适配；显式 TransHandR VR 数值流程不变。VR 不隐式套用相机预处理。

## 数值与坐标

复用 `build_hand_targets(window, spec).positions_native_m`，即：

```python
p_native = ((window.hands[spec.side][-1, spec.human_tip_indices, :]
             - window.hands[spec.side][-1, :1, :])
            @ spec.native_to_canonical.T) * spec.reference_palm_length_m
```

这是 spec.root_link 局部坐标中的米制指尖位置。上游 PositionOptimizer 比较的是
RobotWrapper 的 FK 基准坐标中的 link 平移，必须显式转换：

```python
p_fk = p_native @ root_pose[:3, :3].T + root_pose[:3, 3]
```

root_pose 从同一 RobotWrapper、同一固定关节配置计算。初始化时核对 root_link
之前没有参与优化的活动关节；若不满足固定掌根契约则明确报错。
用非零平移、非单位旋转的测试 URDF 验证转换，避免当前资产根变换为单位阵时掩盖错误。

直接调用未修改的 `PositionOptimizer`、`SeqRetargeting`、`MimicJointKinematicAdaptor`。
target links 使用 spec.tip_frames，human indices 使用 spec.human_tip_indices（一维 hand25 索引）。
不可照搬上游 hand21 的 `[4,8,12,16,20]`；不可照搬 offline YAML 的 dummy free joints。

首版参数：L 为 reference_palm_length_m；huber_delta = .02 * L；norm_delta = 0；
ftol_abs = 1e-5 * L；xtol_abs = 1e-6 rad；maxeval = 100；关节限位 epsilon = 0；内部低通关闭；
初值及 reset 使用 spec.q_reference。以上属于项目参数配置，不能宣称是上游默认参数。
位置损失为所有坐标分量的 SmoothL1 均值；Vector 对每条向量的欧氏范数做 SmoothL1，二者不可混同。
位置损失同样满足 `SmoothL1(L*x; beta=L*b) = L*SmoothL1(x; beta=b)`，故绝对停止阈值按 L 缩放。

源码中 norm_delta 项只进入梯度而未计入返回的目标函数值。首版通过配置为 0 避免该不一致，
不修改上游实现、不将其描述为已验证的时序正则；平滑由已有 JointCommandPipeline 完成。
不继承 NativeVectorSolver 专用于 G1 的多初值恢复逻辑。

## 失败语义与记录

输入 profile/形状等契约错误沿用现有边界校验并抛出；缺手返回 invalid 结果并 reset 对应 solver。
求解前保存 last_qpos；NLopt 非正状态、非有限值、越有效限位输出均返回 invalid 并恢复状态。
上游 Optimizer.retarget 会捕获 RuntimeError 并返回上一解，必须额外检查 NLopt 状态，不能把旧值当成功。
捕获异常限于可恢复数值失败；初始化、缺依赖及配置错误保留原始异常上下文。
微小浮点误差可按现有 1e-6 rad 容差裁剪，超过容差不能裁剪伪装成功。

沿用现有 H5 schema；metadata.backend = dex-position，algorithm_parameters.type = position，
记录实际参数、固定掌根、坐标系、缩放规则、上游版本和源哈希。现有 target_vectors 仍为归一化指标，
不是 PositionOptimizer 的米制输入。robot control 记录同样加入真实算法参数。
保留 compare_hand_methods 的既有 NN/Vector 契约；本期 Position 用独立数值验收，不改旧比较器的语义。

## 来源与核验记录

当前项目基线：`1c9a8f85420154af5e029565df99faa4e4a7486b`。
本地参考库：`D:/2026/code/dex_retargeting`，HEAD `05581fb9673e26c78eb5ac008843d662341b3593`；
存在未跟踪输出，本次未修改参考库。以下为实际文件 SHA256：

|路径|SHA256|
|---|---|
|D:/2026/code/dex_retargeting/dex-retargeting/dex_retargeting/optimizer.py|13d9e3fd3c3525b13836bd9cc33aa67aecfb82c564d0a282aa4b5f5398f31638|
|D:/2026/code/dex_retargeting/dex-retargeting/dex_retargeting/seq_retarget.py|0e7eddf301bd108053537020a0fc0d187bae7f46a3c235d9f5ca1743384679b1|
|D:/2026/code/dex_retargeting/dex-retargeting/dex_retargeting/retargeting_config.py|5092a8475af8a0bf23a9ab957cd4f6bb8696746b7982905120335044dc34fb7e|
|D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/lib/site-packages/dex_retargeting/optimizer.py|6b816d65c81f2e959ea6b76876bb03c2b768be95064a1b4b9e1563d36262cdf8|
|D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py|f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f|

运行环境版本为 `dex_retargeting 0.5.0+winconda`。安装包和参考 optimizer.py 的整文件哈希不同，
但 ast.dump(include_attributes=False) 对 Optimizer、PositionOptimizer 的 SHA256 分别一致：
`3e64c3179c4394b303b5c878bd33c3f4de60d7035d4edd42e6416e634b992b42`、
`1973b573c4b1abf4f3243edbedbfabf3d441b3d0c07163d397350bb8f1311f3f`。
这是源码结构核验，运行时数值等价性仍须实施阶段测试。
TransHandR 文件仅作为 VR 保护基线记录；本期不移植或调整其算法。

官方参考：[RetargetingConfig](https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/retargeting_config.py)、
[项目说明](https://github.com/dexsuite/dex-retargeting)。实施以记录的本地源码和实际安装包为准，不能用移动的 main 代替固定基线。

## 验收

同参数、初值、URDF、mimic 和输入序列下，封装结果与直接调用上游的独立关节输出误差 <= 1e-6 rad。
每模型/侧使用确定性可达 FK 样本，重建指尖最大误差 <= 5 mm；关节解非唯一，不用原始关节恢复误差作为精度判据。
不可达样本检查有效状态、限位和有限性并报告残差，不要求虚假的精确恢复。
报告 raw result 和执行后 command 指标，避免滤波、限速影响被算作优化器误差。
全套 unittest 在正式环境通过；dex-native 环境中的 Position 测试必须实际运行，不能全部 skip。
真实相机/VR 时延、手感和精度只有实际录制验证后才可宣称通过。
