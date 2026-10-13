# TransHandR 的限位、碰撞与平滑：本地实现核对

检查日期：2026-10-14。参考目录：`D:/2026/code/TransHandR/TransHandR`。
结论只适用于本机这份代码，不推断其他版本或论文中未提供的实现。

## 原版采用的方式

1. `config/variables_define.py:980` 设置 `receptive_field=3`，用连续三帧输入 Transformer。
   时序输入能提供运动上下文，不等同于显式角速度或加速度约束。
2. `model/model_poseformer.py:199` 构造 `AngleClamper(..., soft_clip=True)`；
   其 `forward` 从约第 316 行通过 tanh 和关节范围映射限制角度。
   每个关节合法不代表多个手指的组合姿态没有自碰撞。
3. `main_train.py:147` 构造碰撞损失。`model/loss.py:537` 的 `CollisionLoss`
   对 FK 节点计算两两距离，排除自身、基点和指定点对，处罚低于阈值的距离；
   约第 609 行使用 `exp(-(distance/threshold)^2)` 的均值。
   `config/variables_define.py:1006` 的阈值是 0.010；第 1001 行碰撞项权重是 10。
4. 对这份原版 Python 源码检索 `Kalman`、`filterpy`、`LPFilter`、`smoothing`、
   `lowpass`、`moving_average` 等，没有找到相应在线输出平滑实现。
   `main_hand_r_s.py:270` 将当前角度直接组成仿真动作。

## 当前工程已经继承的部分

`src/teleoperation/retargeting/hand/transformer.py` 保留角度范围映射。
`src/teleoperation/learning/losses.py:90` 左右的 `CollisionLoss` 使用相同的点距离软惩罚形式，
配置阈值仍是 0.010、碰撞权重仍是 10。当前 world 模型采用共享右侧 FK 和原生侧角度适配，
不能通过套用原版权重或改写 checkpoint 元数据来切换成原版输入坐标。

## 为什么还会碰撞，以及这次怎样处理

FK 点距离损失在训练中是软惩罚；它检查少量节点，不覆盖整个手指外壳。
不同关节角度可以都在合法范围内，却让中间连杆或外壳相交。
单纯增大损失权重或套用滤波器，都不能保证所有推理姿态可执行。

这次保持训练、FK 和 checkpoint 契约，加入共享输出命令层：
低通或常速度卡尔曼滤波 -> 角速度限制 -> L21 仿真指间碰撞路径检查。
默认低通参考 dex-retargeting 的输出 `LPFilter`；卡尔曼复用本工程已有的
`postprocessing.KalmanSmoother`，并非从原版 TransHandR 找到的实现。
碰撞检查是额外的仿真执行层措施，不冒充训练损失的硬保证。

用实际共享权重和原对齐录制的 120 个有效窗口测试时，原始命令全部触发配置的
碰撞/最小距离判据；这不意味着它们全部是肉眼可见或硬件上的严重穿插。
修改后的命令在同一检查范围内通过，但其中有被投影、受限的动作。
这份片段测试不能替代全数据集、左手、外部物体或真实机械手验收。

参数、命令和适用范围见 [实时手部输出处理](hand-command-safety.md)。
