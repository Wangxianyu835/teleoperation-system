# mytrans 核心说明

## 项目定位

本项目用于手部动作重定向：把 Vision Pro、SLAM/SLAMHR 或 H5 数据中的人手 3D 关键点，转换为机器人灵巧手的关节角，并支持离线推理、PyBullet 仿真和 LinkerHand 真实硬件控制。

## 核心流程

1. 数据输入  
   `dataset/custom_dataset.py` 从 H5 文件读取右手 3D 关键点序列，主要字段为 `r_glove_pos`，并按配置做缩放和 subject 选择。

2. 数据切片  
   `dataset/generators.py` 的 `ChunkedGenerator` 按时间窗口组织数据，默认给模型提供连续多帧输入。

3. 角度预测  
   `model/model_poseformer.py` 中的 `PoseTransformer` 先做帧内关节空间 Transformer，再做跨帧时间 Transformer，输出机器人手关节角，并通过 `AngleClamper` 限制到合法角度范围。

4. 运动学约束  
   `model/angle2real.py` 解析 URDF 和手部关节配置，构建正向运动学模型，把预测角度还原为机器人手 3D 关节位置。

5. 损失计算  
   `model/loss.py` 将机器人手 FK 结果与人手关键点对齐，组合向量方向、指尖位置、碰撞、拇指约束、指尖距离等损失训练模型。

6. 输出控制  
   推理结果可保存为 H5，也可送入 `yumi_gym` 做 PyBullet 仿真，或经 `main_h5_realtime.py` / `main_visionpro_realtime.py` 转成 0-255 驱动值控制 LinkerHand。

## 主要模块

- `config/`：命令行参数、全局变量、手型配置、关节索引、角度范围、URDF 路径、训练超参数和工具函数。
- `dataset/`：H5 数据读取、骨架定义、数据切片、数据清洗、坐标对齐、采集与可视化脚本。
- `model/`：PoseFormer 模型、正向运动学、损失函数、碰撞约束和角度到位置转换逻辑。
- `yumi_gym/`：注册 `yumi-v0` 仿真环境，加载不同机器人手 URDF，并用 PyBullet 执行动作。
- `LinkerHand/`：LinkerHand SDK 封装，负责 CAN/485 初始化、速度/力矩设置、位置控制和状态读取。
- `checkpoint/`：训练日志、模型权重和 TensorBoard 记录。
- `output/`：模型推理角度、转换后位置和对比结果输出。

## 关键入口

- `main_train.py`：训练入口。读取 H5 数据，创建 `PoseTransformer` 和 FK 模型，按多项约束损失训练，最终保存 `model_final.pth`。
- `main_test.py`：离线推理入口。加载训练好的模型，把 H5 关键点批量转换为机器人关节角，并写入输出 H5。
- `main_h5_simulate.py`：读取推理 H5，把角度补齐后送入 PyBullet 仿真环境播放。
- `main_h5_realtime.py`：读取推理 H5，将弧度角映射到 LinkerHand 0-255 驱动序列，并通过 CAN 控制真实手。
- `main_visionpro_realtime.py`：实时多进程链路。Vision Pro 采集三帧人手点云，模型重定向输出机器人角度，同时驱动仿真和真实 LinkerHand。
- `main_hand_r_s.py`：从 H5 读取角度，同时分发给仿真环境和真实手控制进程。
- `data_angle2pos.py`：把模型输出的关节角通过 FK 转成机器人手 3D 关节位置并保存。
- `check_gym.py`：检查 PyBullet/Gym 环境是否能正常加载和执行动作。
- `check_hand_connect.py`：检查 LinkerHand CAN 通道、硬件初始化和基础控制。
- `check_model.py`：检查模型权重加载、参数统计、NaN/Inf 和异常数值。

## 数据与控制链路

离线链路：`H5 3D关键点 -> ChunkedGenerator -> PoseTransformer -> 关节角H5 -> 仿真/真实手/位置转换`

实时链路：`Vision Pro -> 三帧关键点缓存 -> PoseTransformer -> robot_angles -> PyBullet / LinkerHand`
