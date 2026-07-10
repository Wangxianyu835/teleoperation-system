## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
from timm.models import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoseTransformer(nn.Module):
    # 在__init__函数中调整以下参数：
    def __init__(self, 
                num_frame=3, 
                in_num_joints=25, 
                in_chans=3,
                out_num_joint=17,
                out_chans=1,
                embed_dim_ratio=64,  # 增大空间嵌入维度，从32改为64
                spatial_depth=6,     # 增加空间Transformer层数
                temporal_depth=2,    # 减少时序Transformer层数
                spatial_mlp_ratio=4., # 增大空间MLP比例
                temporal_mlp_ratio=1., # 减小时序MLP比例
                num_heads=8, 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.2,  
                norm_layer=None,
                angle_limit_rad=None):
        """
        3D人体姿态估计的PoseTransformer模型
        Args:
            num_frame (int): 输入帧数，默认3帧
            in_num_joints (int): 关节点数量，默认25个关节
            in_chans (int): 输入通道数，3D关节点有3个通道：(x,y,z)坐标
            out_num_joints (int): 输出关节点数量，默认17个关节
            out_chans (int): 输出通道数，一维向量，即输出关节角度值
            embed_dim_ratio (int): 空间嵌入维度比率
            depth (int): Transformer块的深度
            num_heads (int): 注意力头的数量
            mlp_ratio (int): MLP隐藏层维度与嵌入维度的比例
            qkv_bias (bool): QKV线性变换是否使用偏置项
            qk_scale (float): QK缩放因子，如果不设置则默认为head_dim**-0.5
            drop_rate (float): Dropout概率
            attn_drop_rate (float): 注意力Dropout概率
            drop_path_rate (float): 随机深度路径Dropout概率
            norm_layer: 归一化层类型，默认LayerNorm
            angle_limit_rad: 角度限制，单位为弧度，默认无
        """
        #先独立帧内做关节位置编码，再做时序编码，再做关节角度预测
        super().__init__()
        # 设置归一化层，默认为LayerNorm，eps=1e-6
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # 计算时序嵌入维度：关节数量 × 空间嵌入维度比率
        embed_dim = embed_dim_ratio * in_num_joints
        
        # 输出维度：关节数量 × 1 角度向量
        out_dim = out_num_joint * out_chans    
        self.out_dim = out_dim
        self.out_chans = out_chans
        # 空间补丁嵌入：将3D关节点坐标(x,y,z)映射到高维空间
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        
        # 空间位置嵌入：为每个关节点添加位置信息
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, in_num_joints, embed_dim_ratio))

        # 时序位置嵌入：为每帧添加位置信息
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        
        # 位置Dropout层
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则：创建从0到drop_path_rate的线性序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, spatial_depth)]

        # 空间Transformer块列表：分别处理每个关节的空间关系
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, 
                num_heads=num_heads, 
                mlp_ratio=spatial_mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(spatial_depth)])
        # 随机深度衰减规则：创建从0到drop_path_rate的线性序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, temporal_depth)]
        # 时序Transformer块列表：处理关节在时间维度上的关系
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=temporal_mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(temporal_depth)])

        # 空间归一化层
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        
        # 时序归一化层
        self.Temporal_norm = norm_layer(embed_dim)

        # 加权平均卷积层：实现对不同帧的加权平均，提取中心帧信息
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        # 输出头：将特征映射到一维关节角度向量空间
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),  # 特征归一化
            nn.Linear(embed_dim, out_dim),  # 映射到输出维度
        )
        # 添加角度限制层
        self.angle_limit_rad = angle_limit_rad
        if angle_limit_rad is not None:
            self.angle_clamper = AngleClamper(angle_limit_rad, out_num_joint, out_chans,soft_clip=True)
        else:
            # 默认使用tanh激活，输出范围[-1, 1]，然后映射到[-π, π]
            self.angle_clamper = nn.Tanh()  # 输出范围[-1, 1]
    def Spatial_forward_features(self, x):
        """
        空间特征提取：对每个时间步独立处理各关节的空间关系
        
        Args:
            x: 输入张量,shape [batch_size, channels, frames, joints]
            
        Returns:
            处理后的特征,shape [batch_size, frames, embed_dim]
        """
        b, c , f, p = x.shape  # 提取批次大小、通道数、帧数、关节数
        # 重排张量：将批次和帧合并处理，shape [batch_size*frames, joints, channels]
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        # 将3D关节点坐标映射到高维空间
        x = self.Spatial_patch_to_embedding(x)
        # 添加空间位置嵌入
        x += self.Spatial_pos_embed
        # 应用位置Dropout
        x = self.pos_drop(x)
        # 通过多个空间Transformer块处理
        for blk in self.Spatial_blocks:
            x = blk(x)

        # 最终空间归一化
        x = self.Spatial_norm(x)
        # 重新排列张量形状，合并空间维度，shape [batch_size, frames, embed_dim]
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        """
        时序特征提取：处理关节在时间维度上的变化
        
        Args:
            x: 输入张量，shape [batch_size, frames, embed_dim]
            
        Returns:
            处理后的特征，shape [batch_size, 1, embed_dim]
        """
        b = x.shape[0]  # 获取批次大小
        
        # 添加时序位置嵌入
        x += self.Temporal_pos_embed
        # 应用位置Dropout
        x = self.pos_drop(x)
        
        # 通过多个时序Transformer块处理
        for blk in self.blocks:
            x = blk(x)

        # 时序归一化
        x = self.Temporal_norm(x)
        # x shape [batch_size, frames, embed_dim]
        # 使用加权平均提取中心帧特征，shape [batch_size, 1, embed_dim]
        x = self.weighted_mean(x)
        # 调整视图形状
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入张量,shape [batch_size,frames, joints,channels]
            
        Returns:
            3D姿态预测结果,shape [batch_size, 1, joints, 3]
        """
        # 调整输入张量维度顺序
        # print('forward重组前x',x.shape)
        x = x.permute(0, 3, 1, 2)  # [batch_size, channels, frames, joints]
        b, _, _, p = x.shape
        # print('forward重组后x',x.shape)
        # 首先进行空间特征提取：处理各关节的空间关系
        x = self.Spatial_forward_features(x)  # [batch_size, frames, embed_dim]
        # print('空间特征提取',x.shape)
        # 然后进行时序特征提取：处理时间维度上的变化
        x = self.forward_features(x)  # [batch_size, 1, embed_dim]
        # print('时序特征提取',x.shape)
        # 通过输出头得到最终角度向量预测
        x = self.head(x)  # [batch_size, 1, joints]
        # print('输出头',x.shape)
        # 重塑输出形状为[batch_size, frame, joints, 1]，表示每个关节的角度预测
        #x = x.view(b, 1, p, -1)
        # x = x.view(b, 1, self.out_dim, self.out_chans)
        # 应用角度限制
        if hasattr(self, 'angle_clamper') and self.angle_clamper is not None:
            x = self.angle_clamper(x)
        x = x.view(b, self.out_dim)
        # print('reshape',x.shape)
        return x

class AngleClamper(nn.Module):
    def __init__(self, angle_limits, num_joints, num_channels, soft_clip=True):
        super().__init__()
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.soft_clip = soft_clip
        
        if angle_limits is not None:
            limits_tensor = torch.tensor(angle_limits, dtype=torch.float32)
            self.register_buffer('angle_min', limits_tensor[:, 0])
            self.register_buffer('angle_max', limits_tensor[:, 1])
            self.register_buffer('angle_range', limits_tensor[:, 1] - limits_tensor[:, 0])
            self.register_buffer('angle_center', (limits_tensor[:, 1] + limits_tensor[:, 0]) / 2)
        else:
            self.angle_min = None
            self.angle_max = None
    
    def forward(self, x):
        if self.angle_min is not None:
            if x.dim() == 3:
                x = x.view(x.shape[0], x.shape[1], self.num_joints, self.num_channels)
                
                if self.soft_clip:
                    # 使用更温和的激活函数，避免完全饱和
                    x = torch.tanh(x * 0.5)  # 减少饱和
                    x = x * (self.angle_range / 2).unsqueeze(0).unsqueeze(-1) + \
                        self.angle_center.unsqueeze(0).unsqueeze(-1)
                
                x = x.view(x.shape[0], x.shape[1], -1)
            else:
                x = x.view(x.shape[0], self.num_joints, self.num_channels)
                
                if self.soft_clip:
                    x = torch.tanh(x * 0.5)  # 减少饱和
                    x = x * (self.angle_range / 2).unsqueeze(0).unsqueeze(-1) + \
                        self.angle_center.unsqueeze(0).unsqueeze(-1)
                
                x = x.view(x.shape[0], -1)
        else:
            x = torch.tanh(x) * math.pi
        
        return x
    
class AngleClamper01(nn.Module):
    def __init__(self, angle_limits, num_joints, num_channels, soft_clip=False):
        """
        角度限制层
        
        Args:
            angle_limits: 角度限制 [[min1, max1], [min2, max2], ...] 或 None
            num_joints: 关节数量
            num_channels: 通道数量
            soft_clip: 是否使用软限制
        """
        super().__init__()
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.soft_clip = soft_clip  # 添加此行
        
        if angle_limits is not None:
            # 将角度限制转换为张量
            limits_tensor = torch.tensor(angle_limits, dtype=torch.float32)
            self.register_buffer('angle_min', limits_tensor[:, 0])
            self.register_buffer('angle_max', limits_tensor[:, 1])
            
            # 计算范围和中点用于缩放
            self.register_buffer('angle_range', limits_tensor[:, 1] - limits_tensor[:, 0])
            self.register_buffer('angle_center', (limits_tensor[:, 1] + limits_tensor[:, 0]) / 2)
        else:
            # 默认使用tanh，映射到[-π, π]
            self.angle_min = None
            self.angle_max = None
    
    def forward(self, x):
        """
        前向传播，应用角度限制
        
        Args:
            x: 输入张量 [batch_size, seq_len, features] 或 [batch_size, features]
        """
        if self.angle_min is not None:
            # 将tanh输出[-1, 1]映射到指定范围
            if x.dim() == 3:  # [batch_size, seq_len, features]
                # reshape为[batch_size, seq_len, num_joints, num_channels]
                x = x.view(x.shape[0], x.shape[1], self.num_joints, self.num_channels)
                
                # 使用软限制或硬限制
                if self.soft_clip:
                    # 使用soft clamp，保持梯度流动
                    x = torch.tanh(x) * 0.9  # 留出一些空间
                    x = x * (self.angle_range / 2).unsqueeze(0).unsqueeze(-1) + \
                        self.angle_center.unsqueeze(0).unsqueeze(-1)
                else:
                    # 应用tanh激活
                    x = torch.tanh(x)
                    
                    # 映射到指定范围: tanh_output * range/2 + center
                    x = x * (self.angle_range / 2).unsqueeze(0).unsqueeze(-1) + \
                        self.angle_center.unsqueeze(0).unsqueeze(-1)
                
                # reshape回原形状
                x = x.view(x.shape[0], x.shape[1], -1)
            else:  # [batch_size, features]
                # reshape为[batch_size, num_joints, num_channels]
                x = x.view(x.shape[0], self.num_joints, self.num_channels)
                
                # 使用软限制或硬限制
                if self.soft_clip:
                    # 使用soft clamp，保持梯度流动
                    x = torch.tanh(x) * 0.9  # 留出一些空间
                    x = x * (self.angle_range / 2).unsqueeze(0).unsqueeze(-1) + \
                        self.angle_center.unsqueeze(0).unsqueeze(-1)
                else:
                    # 应用tanh激活
                    x = torch.tanh(x)
                    
                    # 映射到指定范围
                    x = x * (self.angle_range / 2).unsqueeze(0).unsqueeze(-1) + \
                        self.angle_center.unsqueeze(0).unsqueeze(-1)
                
                # reshape回原形状
                x = x.view(x.shape[0], -1)
        else:
            # 使用默认的tanh激活，映射到[-π, π]
            x = torch.tanh(x) * math.pi  # 输出范围[-π, π]
        
        return x

class SigmoidAngleClamper(nn.Module):
    def __init__(self, angle_limits, num_joints, num_channels):
        """
        使用sigmoid函数的角度限制层
        """
        super().__init__()
        self.num_joints = num_joints
        self.num_channels = num_channels
        
        if angle_limits is not None:
            limits_tensor = torch.tensor(angle_limits, dtype=torch.float32)
            self.register_buffer('angle_min', limits_tensor[:, 0])
            self.register_buffer('angle_max', limits_tensor[:, 1])
            self.register_buffer('angle_range', limits_tensor[:, 1] - limits_tensor[:, 0])
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.shape[0], x.shape[1], self.num_joints, self.num_channels)
            x = torch.sigmoid(x)  # 输出范围[0, 1]
            x = x * self.angle_range.unsqueeze(0).unsqueeze(-1) + self.angle_min.unsqueeze(0).unsqueeze(-1)
            x = x.view(x.shape[0], x.shape[1], -1)
        else:
            x = x.view(x.shape[0], self.num_joints, self.num_channels)
            x = torch.sigmoid(x)  # 输出范围[0, 1]
            x = x * self.angle_range.unsqueeze(0).unsqueeze(-1) + self.angle_min.unsqueeze(0).unsqueeze(-1)
            x = x.view(x.shape[0], -1)
        
        return x

