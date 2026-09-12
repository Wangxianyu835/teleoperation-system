# 使用说明
使用前需分别运行

check_gym.py 测试仿真环境 

check_hand_connect.py 测试手部连接 

check_model.py 测试模型参数是否正常 

check_visionpro_connect.py 测试头显连接是否正常
## 文件设置
### config
放置一些不常修改的参数
### dataset
放置数据集 机器手模型
### model
模型文件和相关算法模块
### output
推理结果和指标输出


目前的情况：vr作为数据输入，模型推理后输出动作，作为输出控制机器手的运动；
待实现的内容：1.扩展输入，使其支持视觉的方面 2.扩展到两只手 3.增加手臂的逆运动求解 4.加入更多的手部类型



  

