"""
基础Transformer模块
提供标准的Transformer编码器组件，包括MLP、Attention和Block
"""
import math
import torch
import torch.nn as nn
from functools import partial
from timm.layers import DropPath
# from timm.models.layers import DropPath

class Mlp(nn.Module):
    """
    多层感知机(MLP)模块
    用于Transformer中的前馈网络(Feed-Forward Network)
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度，默认为in_features
        out_features: 输出特征维度，默认为in_features
        act_layer: 激活函数层，默认为GELU
        drop: Dropout比率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]
            
        Returns:
            输出张量 [B, N, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    多头自注意力机制(Multi-Head Self-Attention)
    实现标准的Scaled Dot-Product Attention
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数，默认为8
        qkv_bias: 是否在QKV线性层中使用偏置，默认为False
        qk_scale: 缩放因子，如果为None则使用head_dim ** -0.5
        attn_drop: 注意力Dropout比率
        proj_drop: 投影层Dropout比率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 缩放因子，用于稳定注意力计算
        self.scale = qk_scale or head_dim ** -0.5

        # 将Q、K、V合并为一个线性层以提高效率
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]，B为batch size，N为序列长度，C为特征维度
            
        Returns:
            输出张量 [B, N, C]
        """
        B, N, C = x.shape
        # 计算Q、K、V并重塑为多头形式 [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离Q、K、V
        
        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 应用softmax归一化

        attn = self.attn_drop(attn)

        # 应用注意力权重到V，然后重塑并投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    Transformer编码器块
    包含一个多头自注意力层和一个MLP前馈网络
    使用残差连接和层归一化
    
    Args:
        dim: 特征维度
        num_heads: 注意力头数
        mlp_hidden_dim: MLP隐藏层维度
        qkv_bias: 是否在QKV中使用偏置
        qk_scale: 注意力缩放因子
        drop: Dropout比率
        attn_drop: 注意力Dropout比率
        drop_path: Stochastic depth的drop path比率
        act_layer: 激活函数层
        norm_layer: 归一化层
    """
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # Stochastic depth：随机深度，用于正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        前向传播
        使用Pre-Norm架构：先归一化，再应用注意力/MLP，最后残差连接
        
        Args:
            x: 输入张量 [B, N, C]
            
        Returns:
            输出张量 [B, N, C]
        """
        # 第一个残差块：自注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 第二个残差块：MLP前馈网络
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Transformer(nn.Module):
    """
    标准Transformer编码器
    由多个Transformer块堆叠而成，用于序列到序列的特征提取
    
    Args:
        depth: Transformer块的层数，默认为3
        embed_dim: 嵌入维度，默认为512
        mlp_hidden_dim: MLP隐藏层维度，默认为1024
        h: 注意力头数，默认为8
        drop_rate: Dropout比率，默认为0.1
        length: 序列长度（位置编码的长度），默认为27
    """
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=27):
        super().__init__()
        drop_path_rate = 0.2  # Stochastic depth比率
        attn_drop_rate = 0.  # 注意力Dropout比率
        qkv_bias = True  # 启用QKV偏置
        qk_scale = None  # 注意力缩放因子

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 位置编码：为序列中的每个位置学习位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth的线性衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        # 堆叠多个Transformer块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # 最终归一化层
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]
                B: batch size
                N: 序列长度
                C: 特征维度（应与embed_dim相同）
                
        Returns:
            输出张量 [B, N, embed_dim]
        """
        # 添加位置编码
        x += self.pos_embed
        x = self.pos_drop(x)

        # 通过所有Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 最终归一化
        x = self.norm(x)

        return x




