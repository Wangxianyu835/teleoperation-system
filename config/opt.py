import argparse
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # 模型架构参数
        self.parser.add_argument('--layers', default=3, type=int)  # Transformer层数
        self.parser.add_argument('--channel', default=512, type=int)  # 通道数
        self.parser.add_argument('--d_hid', default=1024, type=int)  # 隐藏层维度
        
        # 数据集相关参数
        self.parser.add_argument('--dataset', type=str, default='h5')  # 使用的数据集名称
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)  # 关键点检测器类型
        self.parser.add_argument('--data_augmentation', type=bool, default=True)  # 是否使用数据增强
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)  # 是否使用反向增强
        self.parser.add_argument('--test_augmentation', type=bool, default=True)  # 测试时是否使用增强
        self.parser.add_argument('--crop_uv', type=int, default=0)  # UV裁剪参数
        self.parser.add_argument('--root_path', type=str, default='dataset/')  # 数据集根目录路径
        
        # 动作和数据处理参数
        self.parser.add_argument('-a', '--actions', default='*', type=str)  # 要训练/测试的动作类型(*表示所有动作)
        self.parser.add_argument('--downsample', default=1, type=int)  # 下采样因子
        self.parser.add_argument('--subset', default=1, type=float)  # 训练子集比例
        self.parser.add_argument('-s', '--stride', default=1, type=int)  # 步长
        
        # 硬件和训练模式参数
        self.parser.add_argument('--gpu', default='0,1,2,3', type=str, help='')  # 使用的GPU编号
        self.parser.add_argument('--train', default=1)  # 是否为训练模式(1为训练,0为测试)
        self.parser.add_argument('--test', action='store_true')  # 测试模式标志
        
        # 训练超参数
        self.parser.add_argument('--nepoch', type=int, default=15)  # 训练总轮数
        self.parser.add_argument('--batch_size', type=int, default=256, help='can be changed depending on your machine')  # 批次大小
        self.parser.add_argument('--lr', type=float, default=1e-3)  # 初始学习率
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)  # 大幅度学习率衰减因子
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)  # 大幅度衰减间隔轮数
        self.parser.add_argument('--workers', type=int, default=8)  # 数据加载进程数
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)  # 学习率衰减因子
        
        # 模型输入输出参数
        self.parser.add_argument('--frames', type=int, default=351)  # 输入帧数
        self.parser.add_argument('--pad', type=int, default=175)  # 填充大小(通常为(frames-1)//2)
        self.parser.add_argument('--n_joints', type=int, default=25)  # 关节点数量
        self.parser.add_argument('--out_joints', type=int, default=17)  # 输出关节点数量
        self.parser.add_argument('--out_all', type=int, default=1)  # 是否输出所有帧
        self.parser.add_argument('--in_channels', type=int, default=3)  # 输入通道数(通常是x,y,z坐标)
        self.parser.add_argument('--out_channels', type=int, default=1)  # 输出通道数(1)关节角度
        
        # 模型保存和加载参数
        self.parser.add_argument('--checkpoint', type=str, default='./checkpoint')  # 检查点保存路径
        self.parser.add_argument('--previous_dir', type=str, default='./saved/model')  # 预训练模型路径
        self.parser.add_argument('-previous_best_threshold', type=float, default=math.inf)  # 最佳模型阈值
        self.parser.add_argument('-previous_name', type=str, default='')  # 上一个最佳模型名称

    def parse(self):
        self.init()
        
        self.opt = self.parser.parse_args()

        if self.opt.test:
            self.opt.train = 0
            
        self.opt.pad = (self.opt.frames-1) // 2

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')
            self.opt.checkpoint = 'checkpoint/' + logtime + '%d'%(self.opt.frames)
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
       
        return self.opt





        
